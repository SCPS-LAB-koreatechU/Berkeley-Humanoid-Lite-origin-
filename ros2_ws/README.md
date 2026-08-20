# Berkeley Humanoid Lite — MoveIt 2 workspace

RViz + MoveIt inverse kinematics for the Berkeley Humanoid Lite arms, with
DexHand hardware grafted on. **Simulation only** — nothing in this workspace
opens a CAN bus, a serial port, or touches a motor.

Built and verified against ROS 2 Humble.

Every command in one place: **[COMMANDS.md](COMMANDS.md)**.

## Build and run

```bash
git submodule update --init source/berkeley_humanoid_lite_assets
cd ros2_ws
source /opt/ros/humble/setup.bash && colcon build --symlink-install
./run_demo.sh
```

**The submodule step is not optional.** `description/meshes` is a symlink into
the assets submodule, so on a fresh clone it dangles and the build stops at

```
ament_cmake_symlink_install_directory() can't find '.../meshes/'
```

which names the path and nothing else. CMake now checks the mesh directories up
front and prints the command that fixes each one, but the submodule still has to
be there. `.gitmodules` points at SSH; without keys on the machine, fetch it
over HTTPS instead:

```bash
git -c submodule.source/berkeley_humanoid_lite_assets.url=\
https://github.com/HybridRobotics/Berkeley-Humanoid-Lite-Assets.git \
    submodule update --init source/berkeley_humanoid_lite_assets
```

`vendor/` and the mirrored left-hand meshes *are* committed, so
`fetch_vendor.sh` and `mirror_meshes.py` are only for refreshing them from
upstream.

**This repository is not uniformly MIT.** The DexHand content under `vendor/`,
and the mirrored meshes derived from it, are CC BY-NC-SA 4.0: NonCommercial and
ShareAlike. Read **[THIRD_PARTY.md](THIRD_PARTY.md)** before using any of it.

`run_demo.sh` sources the workspace and pins `ROS_DOMAIN_ID` (default 77).
**Use it rather than calling `ros2 launch` directly.** A separate dexhand MoveIt
session normally runs on this machine on the default domain (0), driving real
hardware over `/dev/ttyACM0`. On a shared domain RViz picks up the dexhand
planning scene off `/monitored_planning_scene`, tries to apply dexhand joint
variables to the humanoid model, and aborts:

```
Setting the scene for model 'dexhandv2_right_8servo' but model 'berkeley-humanoid-lite' is loaded.
Variable 'R_Index_Yaw' is not known to model 'berkeley-humanoid-lite'
```

That is a domain collision, not a configuration error. The script refuses to run
on domain 0 for this reason.

**`v1arm` is the robot.** The build is a DexHand V1 forearm, wrist *and
palm*, carrying DexHand V2 digit assemblies — that is what "V1 with the fingers
swapped for V2" is as a kinematic chain — and `v1arm` is the only generated
model that is that.

The V1 palm is four bulk pieces bolted in a row plus a cover plate, and the
third wrist joint drives the first of them directly, as upstream intends. Each
V2 digit is bolted to the bulk that carried V1's own. `hand.palm` in
`description/config/arm_attachment.yaml` chooses between that and the V2 palm,
which is one printed part carrying all five digits; `dexhand` still uses the
latter. It is
the default everywhere now: the MoveIt demo, the RViz sanity check, and the
measurement scripts under `scripts/motion/`. They had drifted, with MoveIt
planning against `v1arm` while the scripts analysed the wristless `dexhand` and
the RViz launch did not offer `v1arm` at all. `models.py` in the motion package
is the single place that names it.

The other three are narrower views of the same robot, useful for particular
jobs and not for planning: `dexhand` has no wrist, so a plan made against it
cannot be executed; `stock` has no hand; `tuning` exists only to fit the mount.

Three robots are available; pass `model:=` through to the launch file:

| `model` | Robot | DOF/arm | IK | Planning groups |
| --- | --- | --- | --- | --- |
| `v1arm` (default) | V1 forearm + 3-DOF wrist + DexHand v2 fingers, **both arms** | 8 | **full 6-DOF pose** | `left_arm`, `right_arm`, `both_arms`, `left_hand`, `right_hand` |
| `dexhand` | DexHand v2 welded to the right arm | 5 | position only | `left_arm`, `right_arm`, `both_arms`, `right_hand` |
| `stock` | Original end effectors | 5 | position only | `left_arm`, `right_arm`, `both_arms` |

`v1arm` is the only model where the interactive marker's **rotation rings do
something**. Five joints cannot realise an arbitrary 6-DOF pose; the V1 wrist
adds two pitch joints with a yaw between them, taking each arm to eight.

That makes orientation goals *solvable*, not *free* — read the reachability note
under Limitations before planning around it.

In the MotionPlanning panel pick a group, drag the interactive marker, then
**Plan & Execute**. `right_hand` has no marker — it is a joint group; use the
imported grip presets (`open`, `fist`, `cylinder_grip`, `spread`, `count_1`…)
from the goal-state dropdown. `both_arms` plans in joint space.

URDF-only sanity check, a slider per joint and no MoveIt involved:

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch berkeley_humanoid_lite_description display.launch.py
```

Sourcing `install/setup.bash` is the step to remember. Without it `ros2` only
searches `/opt/ros/humble` and reports the package as not found, which reads
like a build failure rather than an unsourced shell. `run_demo.sh` does it for
you; the launch files here do not.

`model:=` selects `v1arm` (the default), `dexhand`, `stock` or `tuning`.

## Packages

| Package | Contents |
| --- | --- |
| `berkeley_humanoid_lite_description` | Generated URDFs, meshes (symlinked, not copied) |
| `berkeley_humanoid_lite_moveit_config` | SRDF, kinematics/limits/controller config, launch, simulated controllers |

Four URDFs are generated into `description/urdf/`:

| File | Purpose |
| --- | --- |
| `berkeley_humanoid_lite.urdf` | arms only, stock end effectors |
| `berkeley_humanoid_lite_dexhand.urdf` | DexHand welded to the right forearm |
| `berkeley_humanoid_lite_v1arm.urdf` | V1 forearm + wrist + DexHand fingers, both arms |
| `berkeley_humanoid_lite_tuning.urdf` | v1arm with the right mount as 6 sliders |

## Regenerating

Everything generated is reproducible. Run these after pulling a new upstream
URDF, after the DexHand description changes, or after editing the attachment
config:

```bash
python3 src/berkeley_humanoid_lite_description/mirror_meshes.py   # left-hand STLs
python3 src/berkeley_humanoid_lite_description/generate_urdf.py
python3 src/berkeley_humanoid_lite_moveit_config/generate_srdf.py --model v1arm
python3 src/berkeley_humanoid_lite_moveit_config/generate_srdf.py --model dexhand
python3 src/berkeley_humanoid_lite_moveit_config/generate_srdf.py --model stock
```

An SRDF's collision matrix is tied to the links its URDF has, so regenerate all
three after changing anything structural — `include_legs`, the mount, the wrist.

## Where the V1 forearm and wrist come from

`vendor/dexhand_v1_description/` holds a trimmed copy of
[iotdesignshop/dexhand_description](https://github.com/iotdesignshop/dexhand_description)
— the two URDFs and the meshes this build needs: the forearm, the two wrist
halves, and the palm (four bulk pieces and a cover plate, per side, separately
authored rather than mirrored). V1 *finger* meshes are omitted because the
digits are **V2**.

`generate_urdf.py` reads the wrist kinematics out of those URDFs rather than
hard-coding them, per side, because the left variant is separately authored and
is *not* a mirror of the right (`y = -0.0217` right vs `-0.0203` left, and the
yaw axis flips sign). The chain is:

```
forearm --wrist_pitch_lower--> wrist_lower --wrist_yaw--> wrist_upper --wrist_pitch_upper--> hand
```

Three printed parts, three joints, matching the Forearm / Wrist_joint / Hand
split in the OnShape assembly. There is no axial-rotation joint: the V1 README
lists that servo as optional and the upstream URDF does not model it.

**The upstream inertials are wrong for a print.** They are Fusion 360 defaults
at steel density — 7850–7860 kg/m³ across all four links, so `forearm_1` is
listed at 2.275 kg where its 289.7 cm³ of PLA is 359 g solid. `generate_urdf.py`
rescales mass and inertia by `structure_density_kgm3 / 7850`. The rescaled
figure covers plastic only; the servos, battery and electronics inside the
forearm are not modelled at all.

## Which DexHand this is, and why the fingers curl

The vendored description is the complete V2 hand: 16 independently actuated
joints plus 5 that mimic. This robot has 8 servos. Which joints those drive, and
what the other 13 do, is set in the `hand` block of
`description/config/arm_attachment.yaml`:

| | |
| --- | --- |
| **actuated** (8) | `R_{Index,Middle,Ring,Pinky}_{Pitch,Yaw}`, at the servos' travel -- pitch 0.95 rad against the mechanism's 1.309, yaw 0.30 against 0.349 |
| **coupled** (8) | each flexor follows its knuckle, each DIP follows its flexor, emitted as `<mimic>` |
| **frozen** (5) | the whole thumb, welded at the angles the config names, with the rotation baked into the joint origin |

**This used to come from a file on one developer's laptop.** `generate_urdf.py`
read `dexhand_moveit_ws/.../dexhandv2_right_8servo.urdf` when it existed and the
full 16-joint description when it did not, so regenerating anywhere else
silently produced a different robot -- one with no fingertip frames to aim at
and every finger joint free. `generate_srdf.py` had the same problem with the
matching SRDF, except that one failed outright. Both now read from the
repository, and the models are reproducible from a fresh clone.

The coupling is the part that changes behaviour. With the flexors and DIPs
welded, a finger was a rigid 75 mm rod on a single hinge: it could not close on
anything, and no retargeting could make it. Coupled, one servo curls the whole
finger through 163 degrees. Measured with `scripts/motion/analyze_hand.py`:

| | welded | coupled |
| --- | ---: | ---: |
| Fingertip travel | 101 mm | 137 mm |
| Closest a fingertip comes to the thumb, as configured | 85.5 mm | 51.8 mm |
| ... at the best thumb posture the mechanism allows | 43.3 mm | 0.0 mm |

So a pinch is now reachable -- but not at the posture the config ships, which is
joint zero: the fully extended, splayed pose, 52 mm short of anything. No servo
reaches the thumb, so that posture is decided when the hand is assembled.
`analyze_hand.py` prints the angles that close the gap; put them in
`hand.thumb` and regenerate.

**The coupling ratios are estimates.** Both ship at 1.0, which is the right
shape for a linkage-driven finger and the wrong precision, and every number
above moves with them.

Measuring them is a photograph, not a capture session. The ratios are angle
ratios, so read the angles: clicking the joint centres in a side view to a
couple of pixels pins the distal stage to +/-0.02, where half-millimetre
fingertip triangulation only manages +/-0.07. Each finger's three flexion axes
are exactly parallel, so one view along the palm's y axis reads all three at
once, within 0.4 degrees of projection error at worst, and no servo angle is
needed -- the knuckle angle comes from the same photo.
Better still, the joints need not be clicked at all: every one runs on a bearing
whose dark ring a circle detector finds, so
`scripts/motion/track_finger_joints.py` reads the angles straight out of a video
of the sweep. It measures `Flexor <- Pitch`, which is the ratio worth measuring
-- `DIP <- Flexor` already comes from the vendor's CAD. Clamp the finger and fix
the camera: in a handheld clip the proximal-to-middle phalanx length ratio, a
constant of the hardware, ranged over 0.63 to 0.88, which is the foreshortening
measuring itself. Aim by looking for the bearings -- their axes are the flexion
axes, so a view that shows them as circles is a view in which the finger flexes
in the image plane, and one that shows the finger's back is neither.

**A ratio only describes free space.** The fingers are tendon-driven, so under
contact the proximal joint stops at the object and the distal ones keep closing.
`<mimic>` holds the ratio rigidly and cannot conform, which makes these models
right for reach and retargeting and wrong for grasp physics. Model the tendon in
whichever simulator handles the contact.

While fixing this, one upstream defect surfaced: every `<mimic>` in
`dexhandv2_right.urdf` names its driver without the `R_` prefix the joints
carry, so all five point at joints that do not exist. The generated models
re-emit them with the right names.

## The left hand is generated, not shipped

`dexhandv2_description` provides `right` and `cobot_right` meshes and no left
variant, so `mirror_meshes.py` produces one: vertices mirrored across Y (the
fingers extend along +Z and spread along Y), triangle winding reversed so faces
still point outward, normals recomputed from the rewound vertices. It refuses to
write a mesh whose signed volume changed, which is what catches a missed rewind.

Real files rather than `<mesh scale="-1 1 1">` — negative scale is handled
inconsistently across RViz, FCL and the mesh loaders, and a silently inside-out
collision mesh is a bad thing to debug.

In the URDF the mirrored joint origins negate y, and the **axes** follow the
pseudovector rule `a' = (-ax, ay, -az)`. That is what makes a positive joint
angle curl a left finger the same way it curls the right one, so all twelve
imported grip presets work on both hands untouched.

Left-hand links and joints are renamed `R_` → `L_` (or prefixed `L_` where there
was no side marker). The right hand keeps upstream names so `dexhand_bringup`
still matches it.

## Fitting the DexHand mount

**The mount transform in `description/config/arm_attachment.yaml` is a guess, not
a measurement**, as is the entire V1 wrist chain beside it. The default puts the DexHand base exactly where the stock hand's
frame was with the fingers pointing distally — the stock hand body sits at
z = +0.054 in the `arm_right_elbow_roll` frame and the DexHand's fingers extend
along its own +Z, so an identity rotation already lines them up. What identity
does *not* fix is the standoff from the forearm and which way the palm faces.

To dial it in, load the tuning model and drag the six `dexhand_mount_*` sliders:

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch berkeley_humanoid_lite_description display.launch.py model:=tuning
```

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 run berkeley_humanoid_lite_description read_mount.py --watch
```

The tuning model orders those joints x, y, z, yaw, pitch, roll precisely so the
slider values *are* the URDF origin — no conversion. `read_mount.py` prints a
block to paste over the `mount.right:` section of `arm_attachment.yaml`; then
regenerate and rebuild. The left mount is derived from the right by
`left: mirror` unless you override it with explicit numbers.

## What had to be fixed to get here

**Mesh paths.** The upstream URDF writes `package://../meshes/foo.stl`. The ROS
resource retriever resolves `package://<name>/…` through the ament index, so a
relative `..` never resolves and every link renders empty. `generate_urdf.py`
rewrites the paths and adds a fixed `world` root link (MoveIt needs a fixed root
to plan against). The DexHand's `package://dexhandv2_description/…` paths are
repointed the same way at a symlinked copy, so only this workspace needs sourcing.

**No ros2_control.** Only `moveit_simple_controller_manager` is installed, so
execution goes to plain `FollowJointTrajectory` action servers —
`scripts/sim_trajectory_server.py`, which replays trajectories onto
`/joint_states`. Installing `ros-humble-ros2-control` is not required.

**Collision matrix.** The Setup Assistant is GUI-only, so
`generate_collision_matrix.py` reproduces its sampling: FK over 20 000 random
arm and finger configurations, separating-axis tests between bounding boxes
(primitives exactly, meshes via their vertex bounds), and a pair is disabled when
it collides in every sample or in none. Boxing a mesh is conservative — it can
claim a collision the real geometry would not have, never the reverse.

Intra-hand pairs are **not** recomputed: the DexHand ships an SRDF whose 194-pair
matrix the Setup Assistant already produced against real mesh geometry, and
`--import-srdf` folds those in verbatim. Of the 297 remaining pairs, 267 are
disabled; the 30 left enabled are the ones that matter — the hand and both elbows
against the torso and the same-side hip.

## Limitations worth knowing

**Orientation goals mostly fail, and that is the hardware, not the model.** V1
wrist travel is tight: ±30° lower pitch, ±25° yaw, ±20° upper pitch. Measured at
a mid-range reference point, IK solves only about **8% of uniformly random palm
orientations** (17/200). Position goals are essentially always reachable;
orientation is a narrow cone around whatever the arm posture already implies.
Plan positions freely, treat orientation as a preference rather than a
constraint, and watch the marker colour before relying on a pose goal.

**The mount transform is still a guess.** Everything else about the forearm and
wrist now comes from real upstream data, but where the V1 forearm bolts onto the
humanoid's `elbow_roll` is not a part that exists in any CAD. It is the one
number left to dial in; see Fitting the DexHand mount.

**5 DOF cannot reach an arbitrary pose** — on the `stock` and `dexhand` models.
Five joints against six pose degrees of freedom means KDL runs in
`position_only_ik` mode: the solver drives the tip to the requested *point* and
accepts whatever orientation falls out, so goal orientation in RViz is ignored.
`v1arm` does not have this problem.

**Arm self-collision checking is close to vacuous.** The upstream humanoid URDF
defines collision geometry on only 11 of 28 links, and on the arms only
`*_shoulder_yaw` and `*_elbow_roll` carry any. The arms cannot detect a collision
with each other because there is almost nothing there to collide. The DexHand is
the exception — every one of its 22 links has mesh collision geometry, so the
hand is checked properly against itself and against the body. Treat a successful
*arm* plan as "kinematically valid", not "will not hit itself".

**Only a right hand exists.** `dexhandv2_description` ships `right` and
`cobot_right` meshes and no left variant, so the left arm keeps its stock end
effector. A left hand would need a mirrored description upstream.

**DexHand names are unchanged.** Joints stay `R_Index_Pitch` and links stay
`base_link`, matching upstream, so the existing `dexhand_bringup` driver and
`fingertip_ik` node keep resolving them. That means `base_link` in this combined
robot is the *hand's* mount link, not the torso — the torso is `base`. It also
means a second hand could not be added without prefixing.

**The legs are gone.** `include_legs: false` in
`description/config/arm_attachment.yaml` strips all 12 leg links and their
joints from every model — nothing plans them, no motor drives them, and they
only ever contributed collision geometry. Set it true and the whole chain (URDF,
collision matrix, SRDF) regenerates with them back. With them off the torso
floats: `world` welds to `base` at the origin and the base mesh sits ~0.675 m up,
exactly where it always was.

## Driving it from code

`motion_commander.py` wraps the MoveGroup action so motions can be scripted
instead of dragged. Humble predates `moveit_py`, so it uses the action and
service interfaces directly. Named targets are read from the **running**
move_group's `robot_description_semantic` parameter, so it always matches
whichever model was launched rather than a file that may be stale.

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash && export ROS_DOMAIN_ID=77
```

```bash
ros2 run berkeley_humanoid_lite_moveit_config motion_commander.py --list
```

```bash
ros2 run berkeley_humanoid_lite_moveit_config motion_commander.py right_arm ready
```

```bash
ros2 run berkeley_humanoid_lite_moveit_config motion_commander.py right_hand cylinder_grip
```

```bash
ros2 run berkeley_humanoid_lite_moveit_config motion_commander.py right_arm --position 0.375 -0.463 0.579
```

```bash
ros2 run berkeley_humanoid_lite_moveit_config motion_commander.py --demo
```

Or as a library:

```python
from motion_commander import MotionCommander

arm = MotionCommander()
arm.move_named("right_arm", "ready")
arm.move_position("right_arm", [0.375, -0.463, 0.579])
arm.move_named("right_hand", "cylinder_grip")
```

**Use `move_position`, not `move_pose`.** `move_position` constrains the tip to
a sphere and leaves orientation free; `move_pose` adds an orientation
constraint and, given the wrist's ~8% orientation reachability, usually fails.
Every call returns a bool and exits non-zero on failure, so sequences can be
scripted safely.

## Verification

Start the demo headless in one terminal, then run the check in another **on the
same domain**:

```bash
./run_demo.sh use_rviz:=false
```

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash && ROS_DOMAIN_ID=77 ros2 run berkeley_humanoid_lite_moveit_config check_ik.py --trials 30
```

`check_ik.py` draws random in-limit configurations, runs FK to get a
known-reachable point, solves IK back to it from a neutral seed, and re-runs FK
on the solution. Pass `--model stock` when running the stock robot; the right
arm's tip differs between the two.

Last run on the DexHand model:

`v1arm` (9 DOF per arm, full pose):

| Check | Result |
| --- | --- |
| Wrist kinematics vs V1 source | all 6 joints match exactly, both sides |
| IK round trip, `left_arm` | 40/40 solved, position 0.00 mm, **orientation 0.001°** |
| IK round trip, `right_arm` | 39/40 solved, position 0.01 mm, **orientation 0.001°** |
| Random orientations at a fixed point | **17/200 (8%)** — the real limitation |
| After stripping the legs | IK 20/20 both arms, demo 8/8, zero mesh or TF warnings |
| `motion_commander.py --demo` | 8/8 steps (both arms to ready, both hands open + grip, both arms home) |
| `--position` goal, right arm | reached within 3.93 mm of a 5 mm tolerance sphere |
| `--position` goal, left arm | planned and executed, 14 trajectory points |
| Unreachable goal / bad group / bad target | all fail cleanly with a usable message and non-zero exit |
| Bridge: trajectory before `~/enable` | rejected, `output is disabled; call ~/enable first` |
| Bridge: trajectory naming disabled joints | rejected, offending joints listed |
| Bridge: limit clamp | 3.0 rad request clamped to 0.7354 = URDF upper 0.785398 − 0.05 margin |
| Bridge: E-STOP | latches, blocks trajectories and re-enable; `~/reset_estop` clears it |
| Left hand adapter (against a mock driver) | `L_` goal arrives at the driver as `R_`, values intact |
| Left hand adapter: joint states | only `L_` names reach `/joint_states`, no `R_` leakage |
| Left hand adapter: wrong-side names | rejected, offending joints listed |
| Left hand adapter: enable relay | forwarded to the left driver, its reply returned verbatim |
| Left/right mirror symmetry (FK, zero pose) | 1.6e-6 m — inherited from upstream CAD, see below |
| Controllers registered | `arms_controller` (18), `right_hand_controller` (8), `left_hand_controller` (8) |

The one unsolved right-arm sample is KDL failing to converge from the neutral
seed on a redundant chain, not a modelling error.

`dexhand` (5 DOF per arm, position only):

| Check | Result |
| --- | --- |
| IK round trip, both arms | 30/30 solved, worst position error 0.01 mm |
| `left_arm` plan & execute | success, worst joint error 0.0074 rad |
| `right_hand` plan & execute (`cylinder_grip`) | success, worst finger error 0.0097 rad |

**On symmetry:** the model is not perfectly left/right symmetric, by ~1.6 µm at
the fingertips. That is inherited: the upstream humanoid's own arms already
differ by ~4e-7 m at `elbow_roll` before anything is attached, and the
orientation component of that grows through the hand chain. The mirroring itself
is exact — every mirrored finger axis matches the pseudovector rule to 1e-12.

## Running against hardware

`hardware_bridge.py` replaces `sim_trajectory_server.py` behind the same
`/arms_controller/follow_joint_trajectory` action, so nothing on the MoveIt side
changes — only which node is listening. **It has not driven a real motor yet:**
no CAN interface is up on this machine and `/dev/ttyACM*` is absent, so
everything below was verified in dry run.

The hands are not in the bridge. `dexhand_bringup`'s `dexhand_driver` already
exposes `hand_controller/follow_joint_trajectory` and already uses the same
`R_<finger>_<Yaw|Pitch>` names this URDF does — which is exactly why the right
hand's names were left unprefixed. `moveit_controllers_hardware.yaml` routes
MoveIt straight to it.

### It is built to refuse

| Guard | Behaviour |
| --- | --- |
| Dry run is the default | without `--live` no bus is opened at all |
| Output starts disabled | motors stay IDLE until `~/enable`; trajectories before that are rejected, not dropped |
| Unmapped joints | trajectory rejected naming the offending joints |
| Mapped but `enabled: false` | rejected — this is how uncalibrated motors stay untouched |
| Slew limit | a step goal becomes a ramp at `max_joint_speed` |
| Limit clamp | commands clamped into the URDF limits less `limit_margin` |
| PDO loss | goal freezes after `link_flaky_cycles`, joint auto-disables after `pdo_loss_threshold` |
| E-STOP | latches, IDLEs every motor, and needs a separate `~/reset_estop` to clear |

```bash
ros2 service call /hardware_bridge/enable std_srvs/srv/SetBool '{data: true}'
```

```bash
ros2 service call /hardware_bridge/estop std_srvs/srv/Trigger
```

### Bring-up order

1. **Match the model to the metal.** `config/hardware_joints.yaml` maps only the
   ten CAN arm joints. Plan with `model:=stock`, not `v1arm`: the V1 wrist does
   not exist yet, so a `v1arm` plan includes wrist joints the bridge will reject.
2. **Only motors 1, 3 and 5 are `enabled: true`** — the left shoulder, the ones
   actually calibrated. Every other joint is mapped but disabled. Note this means
   a whole-group `left_arm` plan is still rejected, because the group needs five
   joints and three are live; command the three directly until the elbows are
   fitted.
3. **Verify sign and offset before enabling anything.** `direction` and `offset`
   are transcribed from `robot/bimanual.py`, which describes a robot whose joints
   are named differently. Bring a joint up disabled, back-drive it by hand, watch
   `/joint_states`, and only then set `enabled: true`.
4. **Then, and only then, `--live`.**

### The left hand

`dexhand_driver` is hardwired to one hand: joint names come from
`JOINT_NAMES = R_<finger>_<Yaw|Pitch>` and it publishes them on the **absolute**
`/joint_states`, so two instances would both claim to be the right hand.

Rather than fork it, `left_hand_adapter.py` sits in front of a second instance
and translates:

```
MoveIt --L_* trajectory--> adapter --R_* trajectory--> left driver
MoveIt <--L_* joint states-- adapter <--R_* joint states-- left driver
```

The left driver never learns it is a left hand; MoveIt never learns there is a
translation. `left_hand.launch.py` starts both, with the driver in the
`/left_hand` namespace and its `/joint_states` remapped to
`/left_hand/raw_joint_states` — a namespace alone would not separate them,
because the driver publishes to an absolute topic.

```bash
source ~/Desktop/dexhand_moveit_ws/install/setup.bash
```

```bash
ros2 launch berkeley_humanoid_lite_moveit_config left_hand.launch.py serial_port:=/dev/ttyACM1
```

```bash
ros2 service call /left_hand_adapter/enable std_srvs/srv/SetBool '{data: true}'
```

Physical mirroring is handled by the driver's own `yaw_dir` parameter, defaulted
to `-1,-1,-1,-1` here: a mirrored hand spreads the other way but curls the same
way. **Verify on the bench** — command a small spread and check the fingers move
apart, not together.

### Still missing

- **The V1 wrist.** Feetech SCS2332/SCS15 serial servos, not recoil CAN, and not
  built. `wrist_servos` in the joint map is deliberately empty.

### Before any of it is safe

1. **The mount transform must be real.** Reach, collision checking and where the
   palm ends up are all wrong by however wrong the mount guess in
   `arm_attachment.yaml` is.
2. **Check the load.** The V1 forearm is a self-contained unit: 11 servos, an
   Arduino, and 2x 18650 cells, estimated at 0.5–0.6 kg against the stock hand's
   0.292 kg, on a longer lever (220 mm forearm vs a 54 mm hand offset). The URDF
   declares 20 Nm per joint and the motors can make 1.79 Nm at 20 A, but
   `motor_configuration.json` currently caps `torque_limit` at 0.8.
2. **Joint mapping and sign.** MoveIt joint names are URDF names; the CAN side
   uses device IDs with per-joint direction and offset arrays (see
   `robot/bimanual.py`). These are not the same convention and must be mapped
   explicitly.
3. **Motor coverage.** `bimanual.py` maps CAN IDs 1/3/5 to the left shoulder
   pitch/roll/yaw. A 5-DOF plan needs the elbow motors (IDs 7, 9) present and
   calibrated too.
4. **Safety.** `scripts/motor/gui_arm_control.py` already implements slew-rate
   limiting, PDO-loss detection and out-of-band E-STOP. A hardware bridge should
   reuse that logic rather than command trajectory setpoints directly.
