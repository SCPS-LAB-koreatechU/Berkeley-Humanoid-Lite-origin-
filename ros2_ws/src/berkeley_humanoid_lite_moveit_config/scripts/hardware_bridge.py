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

The hands are not handled here; dexhand_bringup's driver already speaks
FollowJointTrajectory. See config/hardware_joints.yaml.
"""

from __future__ import annotations

import argparse
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


class JointHandle:
    """One actuator: where it lives, which way it turns, and how it is doing."""

    def __init__(self, name: str, spec: dict) -> None:
        self.name = name
        self.bus_key = spec["bus"]
        self.device_id = int(spec["device_id"])
        self.direction = float(spec.get("direction", 1.0))
        self.offset = float(spec.get("offset", 0.0))
        self.enabled = bool(spec.get("enabled", False))

        self.lower = -3.14159
        self.upper = 3.14159
        self.command = 0.0        # slew-limited target, URDF convention [rad]
        self.measured = 0.0
        self.missed = 0
        self.fault: str | None = None
        self.live = False         # actually energised right now

    def to_actuator(self, value: float) -> float:
        return (value + self.offset) * self.direction

    def to_urdf(self, value: float) -> float:
        return value * self.direction - self.offset

    def clamp(self, value: float, margin: float) -> float:
        return min(max(value, self.lower + margin), self.upper - margin)


class HardwareBridge(Node):
    def __init__(self, config: dict, live: bool) -> None:
        super().__init__("hardware_bridge")
        self.live = live
        self.config = config
        self.safety = config["safety"]
        self.gains = config["gains"]

        self.lock = threading.Lock()
        self.estop = threading.Event()
        self.output_enabled = False

        self.joints = {name: JointHandle(name, spec)
                       for name, spec in (config.get("joints") or {}).items()}
        if config.get("wrist_servos"):
            self.get_logger().warn(
                "wrist_servos is populated but this bridge only speaks recoil CAN; "
                "those joints stay unavailable"
            )

        self._apply_urdf_limits()
        self.buses: dict[str, object] = {}
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
        missing = [n for n in self.joints if n not in
                   {j.get("name") for j in root.findall("joint")}]
        if missing:
            raise SystemExit(
                f"joints in the map are not in the robot description: {missing}"
            )
        self.get_logger().info(f"joint limits read from robot_description ({found})")

    def _open_buses(self) -> None:
        try:
            import berkeley_humanoid_lite_lowlevel.recoil as recoil
        except ImportError as error:
            raise SystemExit(
                f"--live needs the lowlevel package on PYTHONPATH: {error}"
            )
        self.recoil = recoil
        for key, channel in self.config["bus"].items():
            if not any(j.bus_key == key for j in self.joints.values()):
                continue
            self.get_logger().info(f"opening CAN bus '{key}' on {channel}")
            self.buses[key] = recoil.Bus(channel)

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

    def _energise(self) -> None:
        with self.lock:
            for handle in self.joints.values():
                if not handle.enabled or handle.fault:
                    continue
                # Start from where the joint actually is, so enabling cannot
                # snap it to a stale target.
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
                if self.live and handle.live:
                    self._exchange(handle)
                elif not self.live:
                    # Dry run: the joint follows its command exactly.
                    handle.measured = handle.command
        self._publish()

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
                        help="open the CAN buses. Without this nothing is touched.")
    parser.add_argument("--config", type=Path, default=None)
    args, _ = parser.parse_known_args()

    config = load_config(args.config)
    rclpy.init()
    node = HardwareBridge(config, live=args.live)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._deenergise()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
