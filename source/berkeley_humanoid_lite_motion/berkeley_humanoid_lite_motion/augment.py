"""Augmenting a small capture set into a training set.

The useful place to augment is the *keypoints*, before retargeting, not the
joint trajectory after it. Perturbing eight solved angles produces poses the
solver would never have chosen and, on a hand this constrained, frequently ones
no servo can hold. Perturbing the human keypoints and re-solving keeps every
sample inside the reachable set by construction, because the same bottleneck
that limits the original limits the augmentation.

Three families are provided, and they compose:

* **Mirroring** doubles the set for free. The generated URDF already mirrors the
  hand's joint axes by the pseudovector rule, so a mirrored pose reuses the same
  joint values -- see the "left hand is generated" section of `ros2_ws/README.md`.
  The humanoid's own left/right joints are *not* mirrored that way and need the
  roll and yaw signs flipped.
* **Time warping** covers speed variation, which a studio capture of one take
  has none of.
* **Capture noise** -- triangulation jitter and per-view occlusion dropout --
  is what makes a policy robust to the estimator it will be fed at run time
  rather than to the offline-smoothed version it trained on.

None of this substitutes for physical augmentation (rolling the retargeted clip
out in simulation under randomised initial states and disturbances). That lives
with the RL environment because it needs the simulator; this module is what
feeds it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .motion import MotionSequence

#: Sagittal mirror for the humanoid's own joints: swap the side, then negate the
#: joints whose positive direction is defined about an axis that the mirror
#: reverses. Pitch is in the sagittal plane and survives; roll and yaw do not.
HUMANOID_MIRROR = ({"left": "right", "right": "left"}, re.compile(r"_(roll|yaw)_joint$"))

#: The DexHand needs no sign flips. `mirror_meshes.py` and `generate_urdf.py`
#: already negate the joint origins' y and apply a' = (-ax, ay, -az) to the axes,
#: which is exactly what makes a positive angle curl a left finger the way it
#: curls the right one.
DEXHAND_MIRROR = ({"R_": "L_", "L_": "R_"}, re.compile(r"(?!)"))


def mirror_joint_name(name: str, swaps: dict[str, str]) -> str:
    for old, new in swaps.items():
        if old in name:
            return name.replace(old, new, 1)
    return name


def mirror(motion: MotionSequence, convention=HUMANOID_MIRROR,
           keypoint_axis: int = 0) -> MotionSequence:
    """Reflect a motion across the sagittal plane.

    Joint columns are renamed and reordered so the result still matches
    `joint_names`, rather than renamed in place -- a mirrored clip has to be
    stackable with an unmirrored one.
    """
    swaps, negate = convention
    mirrored_names = [mirror_joint_name(n, swaps) for n in motion.joint_names]
    # Column j of the output is the mirrored partner's column in the input.
    lookup = {n: i for i, n in enumerate(motion.joint_names)}
    missing = [n for n in mirrored_names if n not in lookup]
    if missing:
        raise KeyError(f"mirror partner absent from joint_names: {missing}")
    order = [lookup[mirror_joint_name(n, swaps)] for n in motion.joint_names]
    sign = np.array([-1.0 if negate.search(n) else 1.0 for n in motion.joint_names])
    joint_pos = motion.joint_pos[:, order] * sign

    root_pos = None if motion.root_pos is None else _negate_axis(motion.root_pos, keypoint_axis)
    root_quat = motion.root_quat
    if root_quat is not None:
        # Reflecting a rotation across the plane normal to axis k gives M R M,
        # which in wxyz keeps the scalar part and the k-th vector component and
        # negates the other two: k=0 -> (w, x, -y, -z). Verified against
        # scipy for all three axes; the tempting extra sign on w is wrong.
        root_quat = root_quat.copy()
        for i in range(1, 4):
            if i != keypoint_axis + 1:
                root_quat[:, i] *= -1.0

    keypoints = motion.keypoint_pos
    if keypoints is not None:
        keypoints = _negate_axis(keypoints, keypoint_axis)

    return MotionSequence(
        fps=motion.fps,
        joint_names=motion.joint_names,
        joint_pos=joint_pos,
        root_pos=root_pos,
        root_quat=root_quat,
        keypoint_names=motion.keypoint_names,
        keypoint_pos=keypoints,
        keypoint_conf=motion.keypoint_conf,
        extras=dict(motion.extras),
    )


def _negate_axis(array: np.ndarray, component: int) -> np.ndarray:
    """Negate one component of the trailing xyz axis, whatever the leading shape."""
    out = np.array(array, copy=True)
    out[..., component] *= -1.0
    return out


def time_warp(motion: MotionSequence, rate: float) -> MotionSequence:
    """Play a clip at `rate` times its speed, keeping the frame rate.

    Implemented by resampling at `fps / rate` and relabelling, so the output has
    the same `fps` and a duration scaled by `1 / rate`. Rates far from 1 are not
    free: the robot's velocity limits do not scale with the clip, and a clip at
    rate 2 may simply be untrackable. `feasible_rate_limit` says where that is.
    """
    if rate <= 0:
        raise ValueError("rate must be positive")
    stretched = motion.resample(motion.fps / rate)
    stretched.fps = motion.fps
    return stretched


def feasible_rate_limit(motion: MotionSequence, max_joint_speed: float) -> float:
    """Largest `rate` whose joint speeds stay within `max_joint_speed` [rad/s].

    `joint_limits_dexhand.yaml` caps the fingers at 3.0 rad/s and the arms at
    2.0; `hardware_joints.yaml` clamps harder still, at 1.0. Speeding a clip up
    past this produces training data the hardware cannot reproduce, which is a
    worse failure than having less data.
    """
    if len(motion) < 2:
        return np.inf
    speed = np.abs(np.diff(motion.joint_pos, axis=0)).max() * motion.fps
    return np.inf if speed < 1e-12 else float(max_joint_speed / speed)


@dataclass
class CaptureNoise:
    """A model of what the capture stage gets wrong, applied to keypoints.

    Defaults are placeholders until measured against the rig: reproject the
    triangulated keypoints into the 32 views and take the residual spread for
    `jitter`, and the per-keypoint fraction of views that lose it for `dropout`.
    Fingers occlude far more often than the wrist, hence `fingertip_dropout`.
    """

    #: Standard deviation of per-keypoint position error [m].
    jitter: float = 0.004
    #: Probability a keypoint is lost on a frame.
    dropout: float = 0.02
    #: Probability a *fingertip* is lost on a frame; they self-occlude most.
    fingertip_dropout: float = 0.08
    #: Frames a lost keypoint stays lost, on average.
    dropout_frames: int = 3

    def apply(self, keypoints: np.ndarray, fingertip_indices=(4, 8, 12, 16, 20),
              seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Return noisy (T, K, 3) keypoints and their (T, K) confidence.

        Dropped keypoints are held at their last seen value with confidence 0
        rather than deleted, because that is what a tracker downstream does and
        what the retargeter's confidence weighting expects.
        """
        rng = np.random.default_rng(seed)
        keypoints = np.array(keypoints, dtype=float, copy=True)
        T, K = keypoints.shape[:2]
        keypoints += rng.normal(0.0, self.jitter, keypoints.shape)

        p = np.full(K, self.dropout)
        p[list(fingertip_indices)] = self.fingertip_dropout
        # Start a dropout with probability p/dropout_frames so the *occupancy*
        # comes out at p once each one is held for dropout_frames.
        starts = rng.random((T, K)) < (p / max(self.dropout_frames, 1))
        lost = np.zeros((T, K), dtype=bool)
        for t in range(T):
            for d in range(self.dropout_frames):
                if t + d < T:
                    lost[t + d] |= starts[t]

        confidence = np.where(lost, 0.0, 1.0)
        for t in range(1, T):
            keypoints[t][lost[t]] = keypoints[t - 1][lost[t]]
        return keypoints, confidence
