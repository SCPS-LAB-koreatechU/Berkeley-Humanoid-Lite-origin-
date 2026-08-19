"""Forward kinematics, and what the 8-servo hand can actually reach."""

import numpy as np
import pytest

from berkeley_humanoid_lite_motion import dexhand as dh
from berkeley_humanoid_lite_motion.dexhand import FINGERS
from berkeley_humanoid_lite_motion.urdf import axis_angle_to_matrix, rpy_to_matrix


def test_rpy_matches_sequential_axis_rotations():
    rpy = np.array([0.3, -0.7, 1.1])
    expected = (axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), rpy[2])
                @ axis_angle_to_matrix(np.array([0.0, 1.0, 0.0]), rpy[1])
                @ axis_angle_to_matrix(np.array([1.0, 0.0, 0.0]), rpy[0]))
    assert np.allclose(rpy_to_matrix(rpy), expected)


def test_axis_angle_is_a_rotation():
    R = axis_angle_to_matrix(np.array([1.0, -2.0, 0.5]), 0.9)
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(axis_angle_to_matrix(np.zeros(3), 1.0), np.eye(3))


def test_unreachable_link_raises_rather_than_returning_identity(hand):
    with pytest.raises(KeyError):
        hand.chain.forward("R_Index_tip_frame", "R_Pinky_tip_frame")


def test_generated_and_upstream_agree_on_frames(hand, upstream):
    """The two descriptions must be interchangeable, since the thumb study
    reads axes from one and finger geometry from the other."""
    for finger, link in dh.UPSTREAM_TIP_LINKS.items():
        generated = hand.chain.position(dh.tip_frame(finger), dh.PALM_FRAME)
        legacy = upstream.forward(link, dh.PALM_FRAME) @ np.append(dh.TIP_FRAME_OFFSET, 1.0)
        assert np.allclose(generated, legacy[:3], atol=1e-9)


def test_only_eight_joints_are_actuated(hand):
    """The whole pipeline is built on this. If a URDF regeneration ever unfreezes
    the thumb or the flexors, the retargeting objective has to change with it."""
    assert hand.joint_names == tuple(
        f"R_{f}_{a}" for a in ("Pitch", "Yaw") for f in FINGERS
    )
    frozen = {n for n, j in hand.chain.joints.items()
              if not j.actuated and n.startswith("R_") and "tip_fixed" not in n}
    assert len(frozen) == 13
    assert {n for n in frozen if "Thumb" in n} == {
        "R_Thumb_Yaw", "R_Thumb_Roll", "R_Thumb_Pitch", "R_Thumb_Flexor", "R_Thumb_DIP"
    }


def test_servo_travel_is_tighter_than_upstream(hand, upstream):
    """The 8-servo build is not upstream with joints removed; it moves less."""
    lower, upper = hand.limits
    for i, name in enumerate(hand.joint_names):
        assert upper[i] - lower[i] < (upstream.joints[name].upper - upstream.joints[name].lower) + 1e-9
    assert np.isclose(upper[0], 0.95)      # pitch, vs 1.309 upstream
    assert np.isclose(upper[4], 0.30)      # yaw,   vs 0.349 upstream


def test_fingers_are_rigid_beyond_the_knuckle(hand):
    """Flexor and DIP are fixed, so a finger is one rigid link on one hinge --
    the reason fingertip position, not joint angle, is the retargeting target."""
    tip_open = hand.chain.position(dh.tip_frame("Index"), dh.PALM_FRAME, {"R_Index_Pitch": 0.0})
    knuckle = hand.chain.position("Index_Knuckle_1", dh.PALM_FRAME, {"R_Index_Pitch": 0.0})
    for pitch in (0.2, 0.6, 0.95):
        moved_tip = hand.chain.position(dh.tip_frame("Index"), dh.PALM_FRAME, {"R_Index_Pitch": pitch})
        moved_knuckle = hand.chain.position("Index_Knuckle_1", dh.PALM_FRAME, {"R_Index_Pitch": pitch})
        assert np.isclose(np.linalg.norm(moved_tip - moved_knuckle),
                          np.linalg.norm(tip_open - knuckle), atol=1e-12)


def test_the_hand_cannot_pinch(hand, upstream):
    """No fingertip reaches the frozen thumb, at any posture the thumb could be
    fixed in. Pick-and-place has to be a caging grasp, not a pinch."""
    at_zero = dh.opposition_gap(hand, upstream, np.zeros(4))
    _, best = dh.best_thumb_posture(hand, upstream, restarts=4)
    assert at_zero > 0.08                      # 85 mm as the URDF ships
    assert best > 0.03                         # 43 mm at the most opposed posture
    assert best < at_zero                      # ... but the posture choice matters
