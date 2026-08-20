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

### Measuring them

**Photograph the finger from the side and read the angles off the image.** Do
not use the capture rig: the ratios *are* angle ratios, and reading them from a
fingertip position dilutes the distal stage through an 18 mm lever. 1-sigma on
each parameter, from a 9-pose sweep:

| what you measure | `Flexor ← Pitch` | `DIP ← Flexor` |
| --- | ---: | ---: |
| fingertip markers, 2.0 mm | ±0.061 | ±0.273 |
| fingertip markers, 0.5 mm | ±0.015 | ±0.068 |
| angles read to ±2° | ±0.021 | ±0.030 |
| joint centres clicked, ±5 px | ±0.032 | ±0.049 |
| **joint centres clicked, ±2 px** | **±0.013** | **±0.022** |

Clicking joint centres in the image beats holding a protractor to it, and beats
half-millimetre triangulation on the parameter that is hard to see. Nothing is
calibrated, and the commanded servo angle is not needed either — every angle,
the knuckle's included, comes from the same photo.

Before anything else, check the coupling exists: hold the hand still, command
one knuckle across its travel, and watch whether the middle and distal segments
rotate relative to the segment before them. If they do not, this build's fingers
are rigid — set both multipliers to 0 and stop.

Then point a camera along the palm's **y axis**. Each finger's three flexion
axes are exactly parallel, so one view reads all three angles undistorted;
viewing along the palm's y rather than the finger's own axis costs at most 0.4°
of projection error (pinky, the worst — 0.1° for index and ring, and the middle
finger's axes lie exactly along y). Nine poses across the travel is the point of
diminishing returns. A video is easier than nine stills; pull frames afterwards
with `ffmpeg -i sweep.mp4 -vf fps=1 frame_%03d.png`.

**Clamp the finger and fix the camera.** Handheld does not work: the module
gets reoriented between poses, and each reorientation foreshortens the
projection differently. In one 57-second handheld clip the proximal-to-middle
phalanx length ratio — a constant of the hardware — ranged over 0.63 to 0.88,
which is the foreshortening measuring itself.

**Aiming the camera is the error that matters, and the only one you cannot fix
afterwards.** Looking along the finger rather than across it foreshortens it,
and a foreshortened finger still gives a perfectly self-consistent set of
angles, so the fit cannot tell. Pure geometry, no noise, on the proximal ratio:

| camera off by | ratio biased by |
| ---: | ---: |
| 5° | −0.003 |
| 10° | −0.014 |
| 20° | −0.054 |
| 30° | −0.119 |
| 60° | −0.428 |

against a ±0.013 error bar from clicking — so aim within about 10°. The tool
cross-checks clicked phalanx lengths against the model's and warns, but only
from roughly 30° up: below that the signal is smaller than the clicking noise,
so a quiet warning is not a clean bill. Filming across a desk at whatever angle
the hand happens to sit is the failure mode. Put the camera on the table, level
with the hand, perpendicular to the finger, and keep it still between poses.

Then:

Every joint runs on a bearing whose dark ring sits on white plastic, so the
joints can be found automatically and the clicking skipped entirely:

```bash
python3 scripts/motion/track_finger_joints.py sweep.mp4 \
    --base 240,610 --roi 300,80,700,400 --preview /tmp/check -o angles.csv
python3 scripts/motion/fit_finger_coupling.py angles.csv
```

That measures `Flexor ← Pitch` — the ratio actually worth measuring, since
`DIP ← Flexor` already comes from the vendor's CAD at 1.0. The fingertip has no
bearing, so the distal angle is left unmeasured rather than guessed. Look at the
`--preview` frames before believing the numbers.

To do it by hand instead, or to get the distal stage too:

```bash
python3 scripts/motion/fit_finger_coupling.py points.csv --from-points --template
# fill in five clicked points per photo, then
python3 scripts/motion/fit_finger_coupling.py points.csv --from-points
```

It fits both stages, prints an error bar per finger and the YAML to paste back,
and tests whether a single multiplier is even the right model — ALLEX needed a
quartic for the same relation. That test needs about ±1° of angle accuracy to
work; coarser measurements pin the ratios but cannot tell curvature from noise.
Angle triples and fingertip positions are also accepted.

### What a ratio cannot capture

These fingers are **tendon-driven** — a cable runs the length of each one — so
the ratio only holds in free space. Under contact the proximal joint stops at
the object and the distal ones keep closing: the finger conforms, which is the
point of an underactuated hand.

`<mimic>` cannot express that. It holds the ratio rigidly, so the model is right
for retargeting and free-space reach and **wrong for grasp physics** — a
simulated finger will push an object away where the real one would wrap it.
Measure the free-space ratio, use it for the kinematics, and model the tendon
properly in whatever simulator handles the contact (MuJoCo tendons plus equality
constraints; PhysX fixed tendons, which is how Isaac Lab's Shadow Hand does it).
ALLEX hits the same wall and says so: exact model in MJCF, linear approximation
in URDF and USD.

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
