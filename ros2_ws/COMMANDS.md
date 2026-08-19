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

Nothing below has driven a real motor yet. Start with `model:=stock`: the V1
wrist does not exist, so a `v1arm` plan contains joints the bridge rejects.

```bash
ros2 run berkeley_humanoid_lite_moveit_config hardware_bridge.py
```

```bash
ros2 run berkeley_humanoid_lite_moveit_config hardware_bridge.py --live
```

Dry run is the default — without `--live` no CAN bus is opened at all.

| Command | What it does |
| --- | --- |
| `ros2 service call /hardware_bridge/enable std_srvs/srv/SetBool '{data: true}'` | energise the arm motors |
| `ros2 service call /hardware_bridge/estop std_srvs/srv/Trigger` | E-STOP, latches |
| `ros2 service call /hardware_bridge/reset_estop std_srvs/srv/Trigger` | clear the latch |
| `ros2 service call /dexhand_driver/enable std_srvs/srv/SetBool '{data: true}'` | right hand servos |
| `ros2 service call /left_hand_adapter/enable std_srvs/srv/SetBool '{data: true}'` | left hand servos |

Second hand (needs the dexhand workspace sourced too):

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
| Bridge rejects every trajectory | model has joints the hardware map lacks (use `stock`), or output not enabled |
| `colcon build` builds four packages | `vendor/COLCON_IGNORE` missing; re-run `fetch_vendor.sh` |
