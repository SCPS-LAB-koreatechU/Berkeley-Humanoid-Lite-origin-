"""The interchange format, and the augmentations that widen a small capture set."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from berkeley_humanoid_lite_motion.augment import (
    DEXHAND_MIRROR, HUMANOID_MIRROR, CaptureNoise, feasible_rate_limit, mirror, time_warp,
)
from berkeley_humanoid_lite_motion.motion import MotionSequence

HUMANOID_JOINTS = tuple(
    f"{limb}_{side}_{joint}_joint"
    for limb, joints in (("arm", ("shoulder_pitch", "shoulder_roll", "shoulder_yaw")),
                         ("leg", ("hip_roll", "hip_pitch", "knee_pitch")))
    for side in ("left", "right")
    for joint in joints
)


def make_motion(frames=10, names=HUMANOID_JOINTS, seed=0):
    rng = np.random.default_rng(seed)
    quat = rng.normal(size=(frames, 4))
    return MotionSequence(
        fps=30.0,
        joint_names=names,
        joint_pos=rng.normal(size=(frames, len(names))),
        root_pos=rng.normal(size=(frames, 3)),
        root_quat=quat / np.linalg.norm(quat, axis=1, keepdims=True),
        keypoint_names=tuple(f"k{i}" for i in range(4)),
        keypoint_pos=rng.normal(size=(frames, 4, 3)),
        extras={"aperture": rng.random(frames)},
    )


def test_shape_mismatches_are_caught_at_construction():
    with pytest.raises(ValueError):
        MotionSequence(fps=30.0, joint_names=("a", "b"), joint_pos=np.zeros((5, 3)))
    with pytest.raises(ValueError):
        MotionSequence(fps=30.0, joint_names=("a",), joint_pos=np.zeros((5, 1)),
                       root_pos=np.zeros((4, 3)))


def test_round_trips_through_disk(tmp_path):
    original = make_motion()
    reloaded = MotionSequence.load(original.save(tmp_path / "clip.npz"))
    assert reloaded.joint_names == original.joint_names
    assert reloaded.keypoint_names == original.keypoint_names
    assert np.allclose(reloaded.joint_pos, original.joint_pos)
    assert np.allclose(reloaded.root_quat, original.root_quat)
    assert np.allclose(reloaded.extras["aperture"], original.extras["aperture"])


def test_select_reorders_by_name_not_by_position():
    motion = make_motion()
    wanted = list(reversed(motion.joint_names))
    assert np.allclose(motion.select(wanted), motion.joint_pos[:, ::-1])


def test_resample_preserves_duration_and_unit_quaternions():
    motion = make_motion(frames=30)
    resampled = motion.resample(50.0)
    assert abs(resampled.duration - motion.duration) < 2 / 50.0
    assert np.allclose(np.linalg.norm(resampled.root_quat, axis=1), 1.0, atol=1e-5)


def test_mirroring_twice_is_the_identity():
    motion = make_motion()
    twice = mirror(mirror(motion))
    assert np.allclose(twice.joint_pos, motion.joint_pos)
    assert np.allclose(twice.root_pos, motion.root_pos)
    assert np.allclose(twice.root_quat, motion.root_quat)


def test_mirroring_swaps_sides_and_flips_only_roll_and_yaw():
    motion = make_motion()
    swapped = mirror(motion)
    for i, name in enumerate(motion.joint_names):
        partner = name.replace("left", "right") if "left" in name else name.replace("right", "left")
        j = motion.joint_names.index(partner)
        sign = -1.0 if name.endswith(("_roll_joint", "_yaw_joint")) else 1.0
        assert np.allclose(swapped.joint_pos[:, i], sign * motion.joint_pos[:, j])


def test_mirrored_root_rotation_matches_the_reflected_matrix():
    """The sign pattern is easy to get wrong and silently produces a motion that
    leans the wrong way, so it is pinned against an explicit reflection."""
    motion = make_motion()
    M = np.diag([-1.0, 1.0, 1.0])
    swapped = mirror(motion, keypoint_axis=0)
    for before, after in zip(motion.root_quat, swapped.root_quat):
        R = Rotation.from_quat(np.roll(before, -1)).as_matrix()
        assert np.allclose(Rotation.from_quat(np.roll(after, -1)).as_matrix(), M @ R @ M, atol=1e-6)


def test_dexhand_mirroring_keeps_joint_values():
    """`generate_urdf.py` already mirrored the hand's joint axes, which is why
    the imported grip presets work on both hands untouched."""
    names = tuple(f"{side}_{f}_{a}" for side in ("R", "L") for a in ("Pitch", "Yaw")
                  for f in ("Index", "Pinky"))
    motion = make_motion(names=names)
    swapped = mirror(motion, DEXHAND_MIRROR)
    for i, name in enumerate(names):
        partner = ("L_" + name[2:]) if name.startswith("R_") else ("R_" + name[2:])
        assert np.allclose(swapped.joint_pos[:, i], motion.joint_pos[:, names.index(partner)])


def test_a_joint_with_no_mirror_partner_is_an_error():
    motion = make_motion(names=("arm_left_shoulder_pitch_joint",))
    with pytest.raises(KeyError):
        mirror(motion, HUMANOID_MIRROR)


@pytest.mark.parametrize("rate", [0.5, 2.0])
def test_time_warp_scales_duration_and_keeps_the_frame_rate(rate):
    motion = make_motion(frames=60)
    warped = time_warp(motion, rate)
    assert warped.fps == motion.fps
    assert abs(warped.duration - motion.duration / rate) < 2 / motion.fps


def test_rate_limit_says_where_speeding_a_clip_up_stops_being_physical():
    """`joint_limits_dexhand.yaml` caps the fingers at 3 rad/s. A clip already at
    that speed cannot be sped up at all."""
    motion = make_motion(frames=30, seed=1)
    speed = np.abs(np.diff(motion.joint_pos, axis=0)).max() * motion.fps
    assert np.isclose(feasible_rate_limit(motion, speed), 1.0)
    assert feasible_rate_limit(motion, 2 * speed) > 1.0
    still = MotionSequence(fps=30.0, joint_names=("a",), joint_pos=np.zeros((10, 1)))
    assert feasible_rate_limit(still, 3.0) == np.inf


def test_capture_noise_marks_what_it_dropped_and_holds_the_last_value():
    keypoints = np.tile(np.arange(21)[:, None] * 0.01, (40, 1, 3))
    noisy, confidence = CaptureNoise(jitter=0.0, dropout=0.0, fingertip_dropout=1.0).apply(
        keypoints, seed=0
    )
    assert confidence[:, 8].min() == 0.0                 # index fingertip goes
    assert confidence[:, 5].min() == 1.0                 # its knuckle does not
    lost = confidence[:, 8] == 0.0
    for t in np.flatnonzero(lost):
        if t > 0:
            assert np.allclose(noisy[t, 8], noisy[t - 1, 8])


def test_capture_noise_jitter_is_the_requested_size():
    keypoints = np.zeros((400, 21, 3))
    noisy, _ = CaptureNoise(jitter=0.005, dropout=0.0, fingertip_dropout=0.0).apply(keypoints, seed=1)
    assert 0.004 < noisy.std() < 0.006
