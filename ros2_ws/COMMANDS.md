# Command reference

Every command assumes the workspace is sourced on an isolated ROS domain:

```bash
cd ros2_ws && source /opt/ros/humble/setup.bash && source install/setup.bash && export ROS_DOMAIN_ID=77
```

`run_demo.sh` does that for you and refuses domain 0, which collides with the
dexhand session that normally runs there.

---

## First time on a machine

```bash
./fetch_vendor.sh
```

```bash
python3 src/berkeley_humanoid_lite_description/mirror_meshes.py
```

```bash
python3 src/berkeley_humanoid_lite_description/generate_urdf.py
```

```bash
colcon build --symlink-install
```

`fetch_vendor.sh` clones the two DexHand descriptions into `vendor/`. They are
not committed — both are CC BY-NC-SA 4.0 and this repo is MIT.

---

## Run

| Command | What it does |
| --- | --- |
| `./run_demo.sh` | MoveIt + RViz, `v1arm` model |
| `./run_demo.sh model:=dexhand` | right arm only, 5 DOF |
| `./run_demo.sh model:=stock` | original end effectors |
| `./run_demo.sh use_rviz:=false` | headless, for tests and hardware |

URDF only, one slider per joint, no MoveIt:

```bash
ros2 launch berkeley_humanoid_lite_description display.launch.py model:=dexhand
```

`model:=` there is `stock`, `dexhand` or `tuning`.

---

## Drive it

```bash
ros2 run berkeley_humanoid_lite_moveit_config motion_commander.py --list
```

| Command | What it does |
| --- | --- |
| `motion_commander.py right_arm ready` | named target from the SRDF |
| `motion_commander.py right_hand cylinder_grip` | grip preset |
| `motion_commander.py right_arm --position 0.375 -0.463 0.579` | tip to a point, orientation free |
| `motion_commander.py right_arm --pose X Y Z QX QY QZ QW` | full pose — usually fails, see below |
| `motion_commander.py --demo` | both arms to ready, both hands open + grip, back to home |

As a library:

```python
from motion_commander import MotionCommander
arm = MotionCommander()
arm.move_named("right_arm", "ready")
arm.move_position("right_arm", [0.375, -0.463, 0.579])
```

Prefer `--position`. The wrist has little travel, so only about 8% of arbitrary
palm orientations are reachable at a given point.

---

## Verify

```bash
ros2 run berkeley_humanoid_lite_moveit_config check_ik.py --trials 30 --model v1arm
```

Draws random in-limit configurations, runs FK for a known-reachable point,
solves IK back from a neutral seed, re-runs FK on the answer. `--model stock`
when running the stock robot: the right arm's tip differs.

---

## Regenerate

Run all of these after changing anything structural — `include_legs`, the mount,
the wrist chain. A collision matrix is tied to the links its URDF has.

```bash
python3 src/berkeley_humanoid_lite_description/generate_urdf.py
```

```bash
python3 src/berkeley_humanoid_lite_moveit_config/generate_srdf.py --model v1arm
```

```bash
python3 src/berkeley_humanoid_lite_moveit_config/generate_srdf.py --model dexhand
```

```bash
python3 src/berkeley_humanoid_lite_moveit_config/generate_srdf.py --model stock
```

---

## Fit the hand mount

The one transform still unmeasured. Load the slider model and drag the six
`dexhand_mount_*` sliders:

```bash
ros2 launch berkeley_humanoid_lite_description display.launch.py model:=tuning
```

```bash
ros2 run berkeley_humanoid_lite_description read_mount.py --watch
```

Slider values *are* the URDF origin — paste the printed block over `mount.right`
in `config/arm_attachment.yaml`, then regenerate.

---

## Hardware

Nothing below has driven a real arm motor or wrist servo yet; the DexHand
(right hand, `/dev/ttyACM0`) has. `hardware:=true` keeps MoveIt/RViz identical
and only swaps who executes: `hardware_bridge.py` for arms **and V1 wrist**
(recoil CAN + Feetech SCS serial), the dexhand drivers for the hands. Everything
starts in dry run.

```bash
./run_demo.sh model:=v1arm hardware:=true                          # all dry run, nothing opened
./run_demo.sh model:=v1arm hardware:=true right_hand:=serial       # hand real, arm+wrist dry
./run_demo.sh model:=v1arm hardware:=true live:=true buses:=wrist right_hand:=serial   # + wrist servos
./run_demo.sh model:=v1arm hardware:=true live:=true right_hand:=serial left_hand:=serial
```

| Arg | Values | Meaning |
| --- | --- | --- |
| `hardware` | false / **true** | real controllers instead of `sim_trajectory_server` |
| `live` | **false** / true | bridge opens CAN + wrist serial; false = dry run |
| `buses` | all / can / wrist | with `live`: which buses to open; the rest stay read-only |
| `right_hand`, `left_hand` | **none** / serial / off | dexhand driver in dry run / real port / not started |
| `right_hand_port`, `left_hand_port` | `/dev/ttyACM0`, `/dev/ttyACM1` | |
| `bridge_config` | path | alternative `hardware_joints.yaml` (e.g. a copy with more joints enabled, for a dry run) |

The bridge can also run by hand against a plain `demo.launch.py`; stop
`sim_trajectory_server` first, they share the action name:

```bash
ros2 run berkeley_humanoid_lite_moveit_config hardware_bridge.py                     # dry run
ros2 run berkeley_humanoid_lite_moveit_config hardware_bridge.py --live --buses wrist
```

| Command | What it does |
| --- | --- |
| `ros2 service call /hardware_bridge/enable std_srvs/srv/SetBool '{data: true}'` | energise arm motors + wrist servos |
| `ros2 service call /hardware_bridge/estop std_srvs/srv/Trigger` | E-STOP, latches (arm + wrist; hands are separate) |
| `ros2 service call /hardware_bridge/reset_estop std_srvs/srv/Trigger` | clear the latch |
| `ros2 service call /dexhand_driver/enable std_srvs/srv/SetBool '{data: true}'` | right hand servos |
| `ros2 service call /left_hand_adapter/enable std_srvs/srv/SetBool '{data: true}'` | left hand servos |

Bring-up order for the arm, one joint at a time, hand on the E-STOP:

```bash
./run_demo.sh model:=v1arm hardware:=true live:=true buses:=can      # nothing enabled yet
```

With output still disabled the bridge polls every mapped motor, so
`/joint_states` (and RViz) show the real arm. The actuators report rotor
radians (firmware gear_ratio 1.0; the reference config says -15), which
`gear_ratio: 15` in `hardware_joints.yaml` converts. Then:

1. back-drive one joint by hand; RViz must move the same way -- if not, flip
   that joint's `direction`;
2. pose the arm at URDF zero (the `home` state), then capture the offsets:

```bash
ros2 service call /hardware_bridge/zero std_srvs/srv/Trigger
```

   It applies them immediately and prints the lines to paste into
   `hardware_joints.yaml`; the firmware forgets its encoder position on power
   cycle, so re-zero (or persist offsets in the firmware) after each power-up;
3. enable and move only the shoulder:

```bash
ros2 service call /hardware_bridge/enable std_srvs/srv/SetBool '{data: true}'
ros2 run berkeley_humanoid_lite_moveit_config motion_commander.py left_shoulder home
```

`left_shoulder` / `right_shoulder` are joint-space groups of the three shoulder
motors only (no IK, no marker) -- the ones `hardware_joints.yaml` enables first.
`left_arm` names all eight joints and stays rejected until elbow and wrist are
enabled too. Enabling refuses a motor that does not answer or sits outside its
URDF limits (an unset `offset`) rather than driving it.

`config/hardware_joints.yaml` ships with only three left-shoulder motors
`enabled: true` and every wrist servo `enabled: false`; a v1arm plan touches all
eight joints of an arm, so it is rejected until the elbow and wrist entries are
verified and enabled (or a copy is passed via `bridge_config` for a dry run).
Wrist servo IDs, ports (`/dev/ttyUSB0/1`), directions and `range_deg` are
placeholders -- verify one servo at a time with torque off, watching
`/joint_states`.

Second hand on its own (needs the dexhand workspace sourced too):

```bash
source ~/Desktop/dexhand_moveit_ws/install/setup.bash
```

```bash
ros2 launch berkeley_humanoid_lite_moveit_config left_hand.launch.py serial_port:=/dev/ttyACM1
```

---

## Things that will bite

| Symptom | Cause |
| --- | --- |
| RViz aborts with `Variable 'R_Index_Yaw' is not known` | ROS domain collision with the dexhand session. Use `run_demo.sh`. |
| Links render empty | `fetch_vendor.sh` not run, or meshes not regenerated |
| Rotation rings do nothing | 5-DOF model (`stock`/`dexhand`) — position-only IK by design |
| Bridge rejects every trajectory | joints still `enabled: false` in `hardware_joints.yaml` (elbow, wrist), output not enabled, or E-STOP latched |
| `colcon build` builds four packages | `vendor/COLCON_IGNORE` missing; re-run `fetch_vendor.sh` |
