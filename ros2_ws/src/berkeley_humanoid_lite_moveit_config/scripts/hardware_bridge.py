#!/usr/bin/env python3
"""Drive the real arm motors from MoveIt, over recoil CAN.

Same FollowJointTrajectory interface as sim_trajectory_server.py, so nothing on
the MoveIt side changes -- only which node is listening.  This one talks to
actuators, so it is built to refuse rather than to guess.

  * **Dry run is the default.** Without --live no bus is opened at all; the
    bridge integrates commands internally and publishes them, which is how you
    check joint mapping, limits and trajectory flow with the robot unpowered.
  * **Output starts disabled**, even with --live. Motors stay in IDLE until
    somebody calls the enable service. Trajectories before that are rejected,
    not silently dropped.
  * Commands are slew-rate limited, so a step goal becomes a ramp.
  * Commands are clamped into the URDF limits less a margin.
  * A joint that stops answering PDOs freezes, then auto-disables.
  * Idle joints are still read (one per bus per cycle), so /joint_states shows
    the real arm before anything is enabled -- back-drive a joint by hand to
    check `direction`/`offset` -- and enabling refuses a joint that is silent
    or sitting outside its URDF limits (an unset offset) instead of yanking it.
  * E-STOP is out of band: it sets a flag the control loop checks every cycle
    and puts every motor in IDLE regardless of what is queued.  It latches, and
    clearing it is a separate service so recovery cannot happen by accident.

Joints marked `enabled: false` in the map are read but never commanded, and a
trajectory naming a joint that is not in the map at all is rejected outright.

    ros2 run berkeley_humanoid_lite_moveit_config hardware_bridge.py
    ros2 run berkeley_humanoid_lite_moveit_config hardware_bridge.py --live

    ros2 service call /hardware_bridge/enable std_srvs/srv/SetBool '{data: true}'
    ros2 service call /hardware_bridge/estop std_srvs/srv/Trigger
    ros2 service call /hardware_bridge/reset_estop std_srvs/srv/Trigger
    ros2 service call /hardware_bridge/zero std_srvs/srv/Trigger    # pose = URDF zero (output off)

The V1 wrist (Feetech SCS2332/SCS15 serial servos, three per arm) is driven
from the same node over a half-duplex TTL adapter, so a v1arm trajectory --
shoulder, elbow AND wrist -- lands on one action server. Wrist joints follow the
same rules: mapped in `wrist_servos`, `enabled: false` until verified, dry run
unless --live, torque off until enabled, torque off on E-STOP. Per bus one
SYNC_WRITE carries every goal each cycle and present positions are polled one
servo per cycle round-robin, so a 3-servo bus reports each joint at ~33 Hz.

    ros2 run berkeley_humanoid_lite_moveit_config hardware_bridge.py --live --buses wrist

opens only the wrist serial ports and leaves CAN alone (useful before the arm
motors are wired), `--buses can` the reverse; joints on a bus that is not opened
are reported but refuse commands.

The hands are not handled here; dexhand_bringup's driver already speaks
FollowJointTrajectory. See config/hardware_joints.yaml.
"""

from __future__ import annotations

import argparse
import math
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rcl_interfaces.srv import GetParameters
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool, Trigger

PACKAGE = "berkeley_humanoid_lite_moveit_config"


def duration_to_seconds(duration: Duration) -> float:
    return duration.sec + duration.nanosec * 1e-9


class FeetechBus:
    """Minimal Feetech SCS (SCS15 / SCS2332) half-duplex serial protocol.

    Big-endian words, the "SCSCL" register map. Only what the bridge needs:
    ping, torque on/off, read present position, sync-write goal positions.
    Every call holds the port lock, so the control loop and the enable service
    cannot interleave bytes on the wire.
    """

    PING, READ, WRITE, SYNC_WRITE = 0x01, 0x02, 0x03, 0x83
    BROADCAST = 0xFE
    REG_TORQUE_ENABLE = 40
    REG_GOAL_POSITION = 42      # position(2) time(2) speed(2)
    REG_PRESENT_POSITION = 56

    def __init__(self, port: str, baud: int = 1_000_000, timeout: float = 0.02) -> None:
        import serial  # pyserial; only needed with --live
        self.serial = serial.Serial(port, baudrate=baud, timeout=timeout)
        self.lock = threading.Lock()

    def close(self) -> None:
        self.serial.close()

    # -- framing ---------------------------------------------------------
    @staticmethod
    def _checksum(payload: bytes) -> int:
        return (~sum(payload)) & 0xFF

    def _send(self, servo_id: int, instruction: int, params: bytes) -> None:
        body = bytes([servo_id, len(params) + 2, instruction]) + params
        self.serial.reset_input_buffer()
        self.serial.write(b"\xff\xff" + body + bytes([self._checksum(body)]))
        self.serial.flush()

    def _recv(self, servo_id: int) -> bytes | None:
        """Status packet: FF FF ID LEN ERR params CHK -> params, or None."""
        head = self.serial.read(4)
        if len(head) < 4 or head[0] != 0xFF or head[1] != 0xFF or head[2] != servo_id:
            return None
        length = head[3]
        rest = self.serial.read(length)
        if len(rest) < length:
            return None
        if self._checksum(head[2:] + rest[:-1]) != rest[-1]:
            return None
        error, params = rest[0], rest[1:-1]
        if error:
            # Servo answered but flags a fault (overload, voltage...). Treat as
            # a valid link; the caller can inspect it if it cares.
            pass
        return params

    # -- commands --------------------------------------------------------
    def ping(self, servo_id: int) -> bool:
        with self.lock:
            self._send(servo_id, self.PING, b"")
            return self._recv(servo_id) is not None

    def read_word(self, servo_id: int, address: int) -> int | None:
        with self.lock:
            self._send(servo_id, self.READ, bytes([address, 2]))
            params = self._recv(servo_id)
        if params is None or len(params) < 2:
            return None
        return (params[0] << 8) | params[1]

    def read_position(self, servo_id: int) -> int | None:
        return self.read_word(servo_id, self.REG_PRESENT_POSITION)

    def write_torque(self, servo_id: int, on: bool) -> bool:
        with self.lock:
            self._send(servo_id, self.WRITE, bytes([self.REG_TORQUE_ENABLE, 1 if on else 0]))
            return self._recv(servo_id) is not None

    def sync_write_goals(self, goals: dict[int, tuple[int, int]]) -> None:
        """goals: servo_id -> (position ticks, speed ticks/s); one packet, no reply."""
        if not goals:
            return
        params = bytearray([self.REG_GOAL_POSITION, 6])
        for servo_id, (position, speed) in goals.items():
            position = min(max(int(position), 0), 1023)
            speed = min(max(int(speed), 0), 1023)
            params += bytes([servo_id, position >> 8, position & 0xFF,
                             0, 0, speed >> 8, speed & 0xFF])
        with self.lock:
            self._send(self.BROADCAST, self.SYNC_WRITE, bytes(params))


class JointHandle:
    """One actuator: where it lives, which way it turns, and how it is doing.

    `kind` is "can" (recoil actuator, radians on the wire) or "wrist" (Feetech
    serial servo, 0..1023 ticks on the wire).
    """

    def __init__(self, name: str, spec: dict, kind: str = "can",
                 servo_models: dict | None = None) -> None:
        self.name = name
        self.kind = kind
        self.bus_key = spec["bus"]
        self.device_id = int(spec.get("device_id", spec.get("servo_id", 0)))
        self.direction = float(spec.get("direction", 1.0))
        self.offset = float(spec.get("offset", 0.0))
        # Actuator units per URDF radian. The recoil firmware reports whatever
        # its own gear_ratio parameter says; with the factory default of 1.0
        # that is ROTOR radians, so a 15:1 gearbox needs 15 here (or -15 --
        # sign folds into direction, keep it positive and use direction).
        self.gear_ratio = float(spec.get("gear_ratio", 1.0))
        if self.gear_ratio == 0.0:
            raise SystemExit(f"{name}: gear_ratio must not be 0")
        self.enabled = bool(spec.get("enabled", False))

        # Wrist servos: URDF radians <-> servo ticks. `range_deg` is the travel
        # covered by 0..1023 and `center_ticks` the tick count at URDF zero.
        model = (servo_models or {}).get(spec.get("model", ""), {})
        range_deg = float(spec.get("range_deg", model.get("range_deg", 200.0)))
        self.ticks_per_rad = 1024.0 / math.radians(range_deg)
        self.center_ticks = float(spec.get("center_ticks", model.get("center_ticks", 512)))
        self.max_speed_ticks = int(spec.get("speed_ticks", model.get("speed_ticks", 0)))

        self.lower = -3.14159
        self.upper = 3.14159
        self.command = 0.0        # slew-limited target, URDF convention [rad]
        self.measured = 0.0       # URDF convention [rad]
        self.raw = 0.0            # last value as the actuator reported it
        self.missed = 0
        self.fault: str | None = None
        self.live = False         # actually energised right now
        self.seen = False         # has ever answered on its bus

    def to_actuator(self, value: float) -> float:
        return (value + self.offset) * self.direction * self.gear_ratio

    def to_urdf(self, value: float) -> float:
        self.raw = value
        return value / self.gear_ratio * self.direction - self.offset

    def offset_for_zero_here(self) -> float:
        """The `offset` that would make the current raw reading URDF zero."""
        return self.raw / self.gear_ratio * self.direction

    def to_ticks(self, value: float) -> int:
        return int(round(self.center_ticks + self.to_actuator(value) * self.ticks_per_rad))

    def from_ticks(self, ticks: int) -> float:
        return self.to_urdf((ticks - self.center_ticks) / self.ticks_per_rad)

    def clamp(self, value: float, margin: float) -> float:
        return min(max(value, self.lower + margin), self.upper - margin)


class HardwareBridge(Node):
    def __init__(self, config: dict, live: bool, buses: str = "all") -> None:
        super().__init__("hardware_bridge")
        self.live = live
        self.config = config
        self.safety = config["safety"]
        self.gains = config["gains"]
        self.open_can = buses in ("all", "can")
        self.open_wrist = buses in ("all", "wrist")

        self.lock = threading.Lock()
        self.estop = threading.Event()
        self.output_enabled = False

        self.joints = {name: JointHandle(name, spec, "can")
                       for name, spec in (config.get("joints") or {}).items()}
        wrist = config.get("wrist_servos") or {}
        models = wrist.get("models") or {}
        for name, spec in (wrist.get("joints") or {}).items():
            if name in self.joints:
                raise SystemExit(f"{name} is listed under both joints and wrist_servos")
            self.joints[name] = JointHandle(name, spec, "wrist", models)
        self.wrist_speed_default = float(wrist.get("speed_scale", 1.0))
        # Round-robin cursors for read-only polling of idle joints.
        self._wrist_poll: dict[str, int] = {}
        self._can_poll: dict[str, int] = {}

        # In live mode a bus we were told not to open makes its joints
        # read-only: they stay in the map (MoveIt still sees them) but any
        # trajectory touching them is refused, exactly like enabled: false.
        if self.live:
            for handle in self.joints.values():
                wanted = self.open_can if handle.kind == "can" else self.open_wrist
                if not wanted and handle.enabled:
                    handle.enabled = False
                    self.get_logger().warn(
                        f"{handle.name}: its bus is not being opened (--buses); "
                        "left read-only")

        self._apply_urdf_limits()
        self.buses: dict[str, object] = {}
        self.wrist_buses: dict[str, FeetechBus] = {}
        if self.live:
            self._open_buses()

        callback_group = ReentrantCallbackGroup()
        self.publisher = self.create_publisher(JointState, "/joint_states", 10)
        rate = float(self.safety.get("control_rate_hz", 100.0))
        self.period = 1.0 / rate
        self.create_timer(self.period, self._tick, callback_group=callback_group)

        self.create_service(SetBool, "~/enable", self._on_enable,
                            callback_group=callback_group)
        self.create_service(Trigger, "~/estop", self._on_estop,
                            callback_group=callback_group)
        self.create_service(Trigger, "~/reset_estop", self._on_reset_estop,
                            callback_group=callback_group)
        self.create_service(Trigger, "~/zero", self._on_zero,
                            callback_group=callback_group)
        self.action = ActionServer(
            self, FollowJointTrajectory,
            "/arms_controller/follow_joint_trajectory",
            execute_callback=self._execute,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            cancel_callback=lambda _g: CancelResponse.ACCEPT,
            callback_group=callback_group,
        )

        commandable = [j.name for j in self.joints.values() if j.enabled]
        mode = "LIVE" if self.live else "dry run (no bus opened)"
        self.get_logger().info(
            f"hardware bridge up in {mode}; {len(commandable)} of "
            f"{len(self.joints)} mapped joints are commandable: {commandable}"
        )
        self.get_logger().warn(
            "output is DISABLED. enable with: "
            "ros2 service call /hardware_bridge/enable std_srvs/srv/SetBool '{data: true}'"
        )

    # ------------------------------------------------------------------ setup

    def _apply_urdf_limits(self) -> None:
        """Take each joint's travel from the robot description move_group loaded."""
        client = self.create_client(GetParameters, "/move_group/get_parameters")
        if not client.wait_for_service(timeout_sec=10.0):
            raise SystemExit(
                "/move_group/get_parameters unavailable; start the demo first so "
                "joint limits can be read from the running robot description"
            )
        request = GetParameters.Request(names=["robot_description"])
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        result = future.result()
        if result is None or not result.values or not result.values[0].string_value:
            raise SystemExit("could not read robot_description")

        root = ET.fromstring(result.values[0].string_value)
        found = 0
        for joint in root.findall("joint"):
            handle = self.joints.get(joint.get("name"))
            limit = joint.find("limit")
            if handle is None or limit is None:
                continue
            handle.lower = float(limit.get("lower", handle.lower))
            handle.upper = float(limit.get("upper", handle.upper))
            found += 1
        described = {j.get("name") for j in root.findall("joint")}
        missing = [n for n in self.joints if n not in described]
        for name in missing:
            # The map covers the fullest model (v1arm); a lesser model simply
            # lacks some of these joints. Dropping them keeps /joint_states and
            # the commandable set honest for whatever move_group loaded, and a
            # misspelt name still ends up rejected as "not in the hardware map".
            del self.joints[name]
        if missing:
            self.get_logger().warn(
                f"{len(missing)} mapped joints are not in this robot model and were "
                f"dropped: {missing}")
        self.get_logger().info(f"joint limits read from robot_description ({found})")

    def _open_buses(self) -> None:
        can_keys = {j.bus_key for j in self.joints.values() if j.kind == "can"}
        if self.open_can and can_keys:
            try:
                import berkeley_humanoid_lite_lowlevel.recoil as recoil
            except ImportError as error:
                raise SystemExit(
                    f"--live needs the lowlevel package on PYTHONPATH: {error} "
                    "(or pass --buses wrist to leave CAN alone)"
                )
            self.recoil = recoil
            for key, channel in (self.config.get("bus") or {}).items():
                if key not in can_keys:
                    continue
                self.get_logger().info(f"opening CAN bus '{key}' on {channel}")
                try:
                    self.buses[key] = recoil.Bus(channel)
                except Exception as error:  # noqa: BLE001 - socketcan raises OSError variants
                    # An arm whose adapter is not plugged in should not take
                    # the other arm down with it. Its joints stay mapped (so
                    # MoveIt still sees them) but cannot be commanded.
                    self.get_logger().error(
                        f"CAN bus '{key}' ({channel}) could not be opened: {error}; "
                        "its joints are left read-only")
                    for handle in self.joints.values():
                        if handle.kind == "can" and handle.bus_key == key:
                            handle.enabled = False
                    continue
                self._can_poll[key] = 0
            for handle in self.joints.values():
                if handle.kind != "can" or handle.bus_key not in self.buses:
                    continue
                measured = self._read_can_position(handle, timeout=0.05)
                if measured is None:
                    self.get_logger().warn(
                        f"{handle.name}: motor {handle.device_id} on "
                        f"'{handle.bus_key}' did not answer; reported as 0 until it does")
                    continue
                handle.seen = True
                handle.measured = measured
                handle.command = measured
                self.get_logger().info(
                    f"{handle.name}: motor {handle.device_id} at {measured:+.3f} rad "
                    f"(URDF limits {handle.lower:+.2f}..{handle.upper:+.2f})")

        wrist_keys = {j.bus_key for j in self.joints.values() if j.kind == "wrist"}
        if self.open_wrist and wrist_keys:
            ports = (self.config.get("wrist_servos") or {}).get("bus") or {}
            for key in sorted(wrist_keys):
                if key not in ports:
                    raise SystemExit(f"wrist_servos.bus has no entry for '{key}'")
                port = ports[key]
                device, baud = port["port"], int(port.get("baud", 1_000_000))
                self.get_logger().info(f"opening wrist servo bus '{key}' on {device} @ {baud}")
                try:
                    self.wrist_buses[key] = FeetechBus(device, baud)
                except Exception as error:  # noqa: BLE001 - surface the OS message
                    raise SystemExit(f"cannot open {device}: {error}")
                self._wrist_poll[key] = 0
                # Read once at start-up so /joint_states shows the real wrist
                # before anything is enabled, and so a missing servo shows now.
                for handle in self.joints.values():
                    if handle.kind != "wrist" or handle.bus_key != key:
                        continue
                    ticks = self.wrist_buses[key].read_position(handle.device_id)
                    if ticks is None:
                        self.get_logger().warn(
                            f"{handle.name}: servo id {handle.device_id} on '{key}' "
                            "did not answer; it will fault when enabled")
                    else:
                        handle.measured = handle.from_ticks(ticks)
                        handle.command = handle.measured

    def _read_can_position(self, handle: JointHandle, timeout: float = 0.002) -> float | None:
        """Measured position over SDO with a bounded wait. recoil's own
        read_position_measured() blocks forever when the motor is absent, which
        would freeze the control loop; this never waits more than `timeout`."""
        bus = self.buses.get(handle.bus_key)
        if bus is None:
            return None
        value = bus._read_parameter_f32(  # noqa: SLF001 - the public wrapper has no timeout
            handle.device_id,
            self.recoil.Parameter.POSITION_CONTROLLER_POSITION_MEASURED,
            timeout=timeout)
        return None if value is None else handle.to_urdf(value)

    def _observe_can(self, key: str) -> None:
        """Read-only pass over one CAN bus: one not-yet-energised motor per
        cycle, round-robin. This is what lets you back-drive a joint with the
        output disabled and watch /joint_states to settle `direction` and
        `offset`, and it means enabling starts from the real position instead
        of a stale zero."""
        handles = [h for h in self.joints.values()
                   if h.kind == "can" and h.bus_key == key and not h.live]
        if not handles:
            return
        cursor = self._can_poll.get(key, 0)
        self._can_poll[key] = cursor + 1
        handle = handles[cursor % len(handles)]
        # A motor that has never answered (not wired yet) costs a full timeout
        # per poll, so ask it only every 100th turn to keep the loop at rate.
        if not handle.seen and handle.missed >= 3 and cursor % 100:
            return
        measured = self._read_can_position(handle)
        if measured is None:
            handle.missed += 1
            return
        if handle.missed and not handle.seen:
            self.get_logger().info(f"{handle.name}: motor {handle.device_id} answering")
        handle.missed = 0
        handle.seen = True
        handle.measured = measured

    # --------------------------------------------------------------- services

    def _on_enable(self, request, response):
        if request.data:
            if self.estop.is_set():
                response.success = False
                response.message = "E-STOP latched; clear it before enabling"
                return response
            self._energise()
            response.message = f"output enabled ({'live' if self.live else 'dry run'})"
        else:
            self._deenergise()
            response.message = "output disabled"
        response.success = True
        self.get_logger().warn(response.message)
        return response

    def _on_estop(self, _request, response):
        self.estop.set()
        self._deenergise()
        response.success = True
        response.message = "E-STOP: all motors idle, output latched off"
        self.get_logger().error(response.message)
        return response

    def _on_reset_estop(self, _request, response):
        """Clear the latch. Separate from ~/enable on purpose: recovering from an
        E-STOP should be a deliberate act, not a side effect of re-enabling."""
        if not self.estop.is_set():
            response.success = True
            response.message = "no E-STOP was latched"
            return response
        self.estop.clear()
        for handle in self.joints.values():
            handle.fault = None
            handle.missed = 0
        response.success = True
        response.message = "E-STOP cleared; output is still disabled"
        self.get_logger().warn(response.message)
        return response

    def _on_zero(self, _request, response):
        """Take the arm's current pose as URDF zero: recompute `offset` for
        every CAN joint that is answering, apply it now, and print the yaml to
        make it permanent. Same idea as bimanual.py, which captures its offsets
        at start-up -- but here it is deliberate, with the output disabled and
        the arm physically posed at zero first. Refused while anything is live."""
        if self.output_enabled:
            response.success = False
            response.message = "disable the output first; zeroing a live joint would jump it"
            return response
        if not self.live:
            response.success = False
            response.message = "dry run: there is nothing to zero"
            return response
        lines, skipped = [], []
        with self.lock:
            for handle in self.joints.values():
                if handle.kind != "can":
                    continue
                if not handle.seen:
                    skipped.append(handle.name)
                    continue
                handle.offset = handle.offset_for_zero_here()
                handle.measured = handle.to_urdf(handle.raw)
                handle.command = handle.measured
                handle.fault = None if handle.fault == "OUT_OF_RANGE" else handle.fault
                lines.append(f"  {handle.name}: offset: {handle.offset:.4f}")
        text = ("applied in memory only -- paste into hardware_joints.yaml:\n"
                + "\n".join(lines))
        if skipped:
            text += f"\n(not answering, untouched: {skipped})"
        response.success = bool(lines)
        response.message = text
        self.get_logger().warn(text)
        return response

    def _energise(self) -> None:
        with self.lock:
            for handle in self.joints.values():
                if not handle.enabled or handle.fault:
                    continue
                # Start from where the joint actually is, so enabling cannot
                # snap it to a stale target.
                if self.live and handle.kind == "wrist":
                    bus = self.wrist_buses[handle.bus_key]
                    ticks = bus.read_position(handle.device_id)
                    if ticks is None:
                        handle.fault = "LOST"
                        self.get_logger().error(
                            f"{handle.name}: servo did not answer at enable; skipped")
                        continue
                    handle.measured = handle.from_ticks(ticks)
                    handle.command = handle.measured
                    if not bus.write_torque(handle.device_id, True):
                        handle.fault = "LOST"
                        self.get_logger().error(
                            f"{handle.name}: torque-on not acknowledged; skipped")
                        continue
                    handle.live = True
                    continue
                if self.live:
                    measured = self._read_can_position(handle, timeout=0.02)
                    if measured is None:
                        handle.fault = "LOST"
                        self.get_logger().error(
                            f"{handle.name}: motor did not answer at enable; skipped")
                        continue
                    handle.measured = measured
                    if not (handle.lower <= measured <= handle.upper):
                        # Enabling here would be legal for the motor but MoveIt
                        # will refuse to start from outside the URDF limits and
                        # the very first clamped target would yank the joint
                        # towards the limit. Almost always an unset offset.
                        handle.fault = "OUT_OF_RANGE"
                        self.get_logger().error(
                            f"{handle.name}: measured {measured:+.3f} rad is outside "
                            f"URDF limits {handle.lower:+.2f}..{handle.upper:+.2f}; "
                            "fix `offset` in hardware_joints.yaml before enabling")
                        continue
                handle.command = handle.measured
                if self.live:
                    bus = self.buses[handle.bus_key]
                    bus.set_mode(handle.device_id, self.recoil.Mode.IDLE)
                    bus.write_position_kp(handle.device_id, self.gains["position_kp"])
                    bus.write_position_kd(handle.device_id, self.gains["position_kd"])
                    bus.write_torque_limit(handle.device_id, self.gains["torque_limit"])
                    bus.feed(handle.device_id)
                    bus.set_mode(handle.device_id, self.recoil.Mode.POSITION)
                handle.live = True
            self.output_enabled = True

    def _deenergise(self) -> None:
        with self.lock:
            for handle in self.joints.values():
                if handle.live and self.live:
                    if handle.kind == "wrist":
                        self.wrist_buses[handle.bus_key].write_torque(handle.device_id, False)
                    else:
                        self.buses[handle.bus_key].set_mode(
                            handle.device_id, self.recoil.Mode.IDLE)
                handle.live = False
            self.output_enabled = False

    # ------------------------------------------------------------ control loop

    def _tick(self) -> None:
        if self.estop.is_set():
            return
        with self.lock:
            for handle in self.joints.values():
                if handle.kind == "wrist":
                    continue          # wrist buses are serviced per bus below
                if self.live and handle.live:
                    self._exchange(handle)
                elif not self.live:
                    # Dry run: the joint follows its command exactly.
                    handle.measured = handle.command
            if self.live:
                for key in self.buses:
                    self._observe_can(key)
                for key in self.wrist_buses:
                    self._exchange_wrist_bus(key)
            else:
                for handle in self.joints.values():
                    if handle.kind == "wrist":
                        handle.measured = handle.command
        self._publish()

    def _exchange_wrist_bus(self, key: str) -> None:
        """One cycle on one serial bus: every live goal in a single SYNC_WRITE,
        then one present-position read (round-robin), so the bus never blocks
        the 100 Hz loop for more than one reply."""
        bus = self.wrist_buses[key]
        handles = [h for h in self.joints.values()
                   if h.kind == "wrist" and h.bus_key == key]
        if not handles:
            return
        goals = {}
        for handle in handles:
            if handle.live:
                speed = handle.max_speed_ticks or int(
                    float(self.safety["max_joint_speed"]) * handle.ticks_per_rad
                    * self.wrist_speed_default)
                goals[handle.device_id] = (handle.to_ticks(handle.command), speed)
        bus.sync_write_goals(goals)

        index = self._wrist_poll.get(key, 0) % len(handles)
        self._wrist_poll[key] = index + 1
        handle = handles[index]
        ticks = bus.read_position(handle.device_id)
        if ticks is None:
            handle.missed += 1
            # Polling is 1-in-N, so scale the loss thresholds accordingly.
            if handle.live and handle.missed * len(handles) >= int(self.safety["pdo_loss_threshold"]):
                handle.fault = "LOST"
                handle.live = False
                bus.write_torque(handle.device_id, False)
                self.get_logger().error(
                    f"{handle.name}: no reply for {handle.missed} polls, disabled")
            return
        handle.missed = 0
        handle.measured = handle.from_ticks(ticks)

    def _exchange(self, handle: JointHandle) -> None:
        bus = self.buses[handle.bus_key]
        bus.transmit_pdo_2(handle.device_id,
                           position_target=handle.to_actuator(handle.command),
                           velocity_target=0.0)
        measured, _velocity = bus.receive_pdo_2(handle.device_id)
        if measured is None:
            handle.missed += 1
            if handle.missed >= int(self.safety["pdo_loss_threshold"]):
                handle.fault = "LOST"
                handle.live = False
                bus.set_mode(handle.device_id, self.recoil.Mode.IDLE)
                self.get_logger().error(
                    f"{handle.name}: no PDO reply for "
                    f"{handle.missed} cycles, disabled"
                )
            return
        handle.missed = 0
        handle.measured = handle.to_urdf(measured)

    def _publish(self) -> None:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        with self.lock:
            message.name = list(self.joints)
            message.position = [self.joints[n].measured for n in message.name]
        self.publisher.publish(message)

    # ---------------------------------------------------------------- action

    def _reject(self, goal_handle, code, reason):
        goal_handle.abort()
        result = FollowJointTrajectory.Result()
        result.error_code = code
        result.error_string = reason
        self.get_logger().error(f"trajectory rejected: {reason}")
        return result

    def _execute(self, goal_handle):
        trajectory = goal_handle.request.trajectory
        names = list(trajectory.joint_names)
        points = list(trajectory.points)
        result = FollowJointTrajectory.Result()

        unmapped = [n for n in names if n not in self.joints]
        if unmapped:
            return self._reject(
                goal_handle, FollowJointTrajectory.Result.INVALID_JOINTS,
                f"joints are not in the hardware map: {unmapped}")

        blocked = [n for n in names if not self.joints[n].enabled]
        if blocked:
            return self._reject(
                goal_handle, FollowJointTrajectory.Result.INVALID_JOINTS,
                f"joints are mapped but disabled in hardware_joints.yaml: {blocked}")

        faulted = [n for n in names if self.joints[n].fault]
        if faulted:
            return self._reject(
                goal_handle, FollowJointTrajectory.Result.INVALID_JOINTS,
                f"joints are faulted and need re-enabling: {faulted}")

        if self.estop.is_set():
            return self._reject(goal_handle, -1, "E-STOP is latched")
        if not self.output_enabled:
            return self._reject(goal_handle, -1,
                                "output is disabled; call ~/enable first")
        if not points:
            goal_handle.succeed()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            return result

        margin = float(self.safety["limit_margin"])
        max_step = float(self.safety["max_joint_speed"]) * self.period
        flaky = int(self.safety["link_flaky_cycles"])

        self.get_logger().info(
            f"executing {len(points)} points over "
            f"{duration_to_seconds(points[-1].time_from_start):.2f}s "
            f"for {len(names)} joints"
        )

        start = self.get_clock().now()
        rate = self.create_rate(1.0 / self.period)
        previous_time = 0.0
        with self.lock:
            previous = [self.joints[n].command for n in names]

        for point in points:
            target_time = duration_to_seconds(point.time_from_start)
            segment = target_time - previous_time

            while rclpy.ok():
                if self.estop.is_set():
                    return self._reject(goal_handle, -1, "E-STOP during execution")
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                    result.error_string = "cancelled"
                    return result

                elapsed = (self.get_clock().now() - start).nanoseconds * 1e-9
                if elapsed >= target_time:
                    break

                alpha = 1.0 if segment <= 0.0 else (elapsed - previous_time) / segment
                alpha = min(max(alpha, 0.0), 1.0)
                with self.lock:
                    for index, name in enumerate(names):
                        handle = self.joints[name]
                        if handle.fault:
                            return self._reject(
                                goal_handle, -1,
                                f"{name} faulted mid-trajectory ({handle.fault})")
                        # A flaky link freezes the goal rather than letting it
                        # walk away from where the joint actually is.
                        if handle.missed >= flaky:
                            continue
                        wanted = (previous[index]
                                  + alpha * (point.positions[index] - previous[index]))
                        wanted = handle.clamp(wanted, margin)
                        step = wanted - handle.command
                        step = min(max(step, -max_step), max_step)
                        handle.command += step
                rate.sleep()

            previous = list(point.positions)
            previous_time = target_time

        goal_handle.succeed()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result


def load_config(path: Path | None) -> dict:
    if path is None:
        path = Path(get_package_share_directory(PACKAGE)) / "config" / "hardware_joints.yaml"
    if not path.is_file():
        raise SystemExit(f"joint map not found: {path}")
    return yaml.safe_load(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="open the CAN buses and wrist serial ports. "
                             "Without this nothing is touched.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--buses", choices=("all", "can", "wrist"), default="all",
                        help="with --live: which buses to open. Joints on a bus "
                             "that stays closed are read-only.")
    args, _ = parser.parse_known_args()

    config = load_config(args.config)
    rclpy.init()
    node = HardwareBridge(config, live=args.live, buses=args.buses)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._deenergise()
        for bus in node.wrist_buses.values():
            bus.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
