# berkeley_humanoid_lite_motion

Volumetric capture to robot motion: retargeting, augmentation, and an honest
account of what the hardware drops on the way.

Pure numpy and scipy. Nothing here imports Isaac Lab or ROS, so it runs in CI and
on a laptop as well as inside a training environment.

```bash
pip install -e source/berkeley_humanoid_lite_motion
pytest source/berkeley_humanoid_lite_motion
```

## Start here

```bash
python3 scripts/motion/analyze_hand.py
```

It reports what the hand can reach, measured from the same URDF MoveIt plans
against. Read it before designing a grasp controller, a retargeting objective or
a reward term, because the headline numbers are worse than the joint count
suggests.

## What the hand actually is

The 8-servo DexHand variant, described in
`ros2_ws/src/berkeley_humanoid_lite_description/config/arm_attachment.yaml`.
Of the hand's 21 joints:

| | |
| --- | --- |
| **Actuated** (8) | `R_{Index,Middle,Ring,Pinky}_{Pitch,Yaw}` — flexion 0…0.95 rad, spread ±0.30 rad |
| **Coupled** (8) | each finger's flexor follows its knuckle, and its DIP follows the flexor |
| **Frozen** (5) | the whole thumb, welded at whatever angles the config names |

The coupling is what makes a finger curl: one servo carries the middle phalanx
and the tip with it, for 163° of total flexion at full travel. Without it a
finger is a rigid 75 mm rod on one hinge and no retargeting can close it.

Measured with `analyze_hand.py`, coupled versus rigid:

| | rigid | coupled |
| --- | ---: | ---: |
| Fingertip travel | 101 mm | **137 mm** |
| Index-to-pinky span | 9–133 mm | 11–146 mm |
| Gap to the thumb, as configured | 85.5 mm | **51.8 mm** |
| Gap to the thumb, best posture | 43.3 mm | **0.0 mm** |

That last row is the one that matters: with the fingers curling, a fingertip can
reach the thumb — **pinch grasps become possible**, where with rigid fingers no
thumb posture could produce one. But not at the posture the config ships, which
is joint zero: the fully extended, splayed pose, 52 mm short. No servo reaches
the thumb, so this is a build-time choice. `analyze_hand.py` prints the angles
that close it; set them under `hand.thumb` and regenerate.

### The ratios are not measured

`Flexor ← Pitch` and `DIP ← Flexor` both ship at 1.0. That is the right *shape*
for a linkage-driven finger and the wrong precision, and every number in the
table above depends on it. For scale, WIRobotics' ALLEX publishes measured
couplings for a comparable anthropomorphic finger — `DIP ← PIP` at **0.656** and
thumb `IP ← MCP` at **0.732**, linear fits through the origin of a quartic — so
ratios of 1.0 should be read as placeholders, not as agreement.

Measure them:

```bash
python3 scripts/motion/fit_finger_coupling.py measurements.csv
```

It fits both stages against the model's own forward kinematics and prints the
YAML to paste back. On synthetic data with 0.3 mm of noise it recovers a 0.83
ratio as 0.824.

**If the fingers on your build really are rigid**, set both multipliers to 0 and
regenerate — the model goes back to what it was, and the pinch conclusion goes
with it.

## Pipeline

```
32 mp4 + 4DS  ──▶  3D keypoints  ──▶  MotionSequence  ──▶  augmentation  ──▶  RL reference
   (capture)        (triangulate)      (retarget)          (this package)     (Isaac Lab)
```

| Module | Role |
| --- | --- |
| `urdf.py` | Minimal URDF parser and forward kinematics. Reads the actuated set off the URDF so a regenerated model cannot silently invalidate a retarget. |
| `dexhand.py` | The hand as this robot has it: actuated joints, fingertip frames, reachability, the frozen-thumb study. |
| `motion.py` | `MotionSequence` — the interchange format every stage reads and writes, as npz. |
| `hand_retarget.py` | 21 human keypoints → 8 servo angles, by fingertip position in a size-normalised palm frame. |
| `augment.py` | Mirroring, time warping, and a capture-noise model. |

### Retargeting is by fingertip, not by joint

Joint-to-joint would mean matching angles that have no counterpart. Instead both
hands get the *same* palm frame construction — wrist plus the two outer knuckles,
the landmarks a 32-view rig loses least often — the human fingertips are scaled
by the knuckle-span ratio, and the eight angles that come nearest are solved for.

The per-frame residual is returned in `extras["fingertip_error"]`, and it is as
much the point as the angles are: it is how far the hardware fell short on that
frame. Tens of millimetres on a curled hand is the hand, not the solver.

Two signals the robot has no joint for are carried alongside rather than
discarded, so a policy can still condition on them:

- `extras["aperture"]` — thumb-to-index pinch distance over knuckle span
- `extras["curl"]` — per-finger flexion in [0, 1]

### Augment the keypoints, not the joint trajectory

Perturbing eight solved angles produces poses the solver would never choose and,
on a hand this constrained, often ones no servo can hold. Perturbing the human
keypoints and re-solving keeps every sample inside the reachable set, because the
same bottleneck that limits the original limits the augmentation.

`feasible_rate_limit()` says how far a clip can be sped up before it exceeds a
joint speed cap — 3.0 rad/s for the fingers in `joint_limits_dexhand.yaml`, and
1.0 rad/s at the tighter clamp in `hardware_joints.yaml`. Past that the
augmentation is producing data the hardware cannot reproduce, which is worse than
having less of it.

Physical augmentation — rolling a clip out under randomised initial states and
disturbances — is not here. It needs the simulator, so it belongs with the RL
environment; this package is what feeds it.

## Not yet built

- **Ingest.** 32-view triangulation from the studio mp4s, and 4DS mesh sequences
  for the root trajectory and floor contact. `MotionSequence` is the target
  format; nothing writes it from video yet.
- **Body retargeting.** Only the hand is done. The 22 body joints in
  `configs/policy_humanoid.yaml` want an IK-based retarget; `scripts/teleop/`
  already carries a pink/pinocchio solver to build on.
- **Grasp reward.** The coupling makes pinch reachable; nothing yet decides when
  to use it.
- **The RL tracking task.** A motion-tracking environment alongside
  `tasks/locomotion/velocity`, reusing the existing rsl_rl plumbing.

## Caveats

- Untested against real capture data. Every test here runs on synthetic
  keypoints built backwards from robot poses, which validates the geometry and
  the solver but says nothing about what a real triangulation looks like.
- `CaptureNoise` defaults are placeholders. Measure them: reproject triangulated
  keypoints into the 32 views for `jitter`, and count per-keypoint view losses
  for `dropout`.
- The hand analysis assumes the mount and the URDF are right. Both carry known
  caveats — see the Limitations section of `ros2_ws/README.md`.
- The coupling ratios are estimates, and the reachability numbers move with
  them. See "The ratios are not measured" above.
- Isaac Lab's URDF importer has historically not honoured `<mimic>`. The
  coupling is resolved in `Chain.resolve()` for anything that goes through this
  package, but a policy trained on an imported USD may be commanding a hand
  whose fingers do not curl. Check the imported articulation before trusting it.
