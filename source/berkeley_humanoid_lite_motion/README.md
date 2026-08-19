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

`generate_urdf.py` builds against the 8-servo DexHand variant. Of the hand's 21
joints it leaves **8 actuated** and freezes **13** to `type="fixed"`:

| | |
| --- | --- |
| Actuated | `R_{Index,Middle,Ring,Pinky}_{Pitch,Yaw}` — flexion 0…0.95 rad, spread ±0.30 rad |
| Frozen | the entire thumb (5 joints), every flexor and every DIP |

Three consequences, all measured rather than assumed:

- **Fingers are rigid.** With the flexor and DIP fixed, a finger is one 75 mm
  link on one hinge. It cannot curl. A fist is not a reachable pose.
- **The hand cannot pinch.** The closest any fingertip comes to the frozen thumb
  tip is **85.5 mm** as the URDF ships. Searching every posture the thumb
  mechanism could be assembled in only gets that to **43.3 mm**. Precision
  pick-and-place is out; caging an object against the palm is what is left.
- **The frozen thumb posture is a free design choice, currently taken badly.**
  Joint zero is the fully extended, splayed pose — the worst of the available
  options. `best_thumb_posture()` reports the angles that halve the gap. No
  servo is involved; this is a matter of how the part is fitted, and then of
  freezing the URDF at those angles instead of at zero.

So finger detail lost between the studio and the robot is mostly lost at the
robot. Sharpening the capture side pays off in the arms and the body; on this
hand it does not.

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
