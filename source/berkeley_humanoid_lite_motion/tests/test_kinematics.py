"""Forward kinematics, and what the 8-servo hand can actually reach."""

import numpy as np
import pytest

from berkeley_humanoid_lite_motion import dexhand as dh
from berkeley_humanoid_lite_motion.dexhand import FINGERS
from berkeley_humanoid_lite_motion.urdf import axis_angle_to_matrix, rpy_to_matrix

from conftest import ATTACHMENT_CONFIG, UPSTREAM_URDF


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
    """The two descriptions must be interchangeable, since the thumb study reads
    axes from one and finger geometry from the other.

    Measured from each finger's own root rather than from a palm: the generated
    model mounts the digits on the V1 bulk chain, so palm-frame positions differ
    by where the bracket puts them. What has to match is the finger itself.
    """
    for finger, link in dh.UPSTREAM_TIP_LINKS.items():
        root = f"{finger}_Knuckle_Cross_1"
        generated = hand.chain.position(dh.tip_frame(finger), root)
        legacy = upstream.forward(link, root) @ np.append(dh.TIP_FRAME_OFFSET, 1.0)
        assert np.allclose(generated, legacy[:3], atol=1e-9)


def test_only_eight_joints_are_actuated(hand):
    """Eight servos, and the retargeting objective is built on exactly these.
    Coupled joints move but cannot be commanded; frozen ones do neither."""
    assert hand.joint_names == tuple(
        f"R_{f}_{a}" for a in ("Pitch", "Yaw") for f in FINGERS
    )
    coupled = {n for n, j in hand.chain.joints.items()
               if j.mimic is not None and n.startswith("R_")}
    frozen = {n for n, j in hand.chain.joints.items()
              if not j.movable and n.startswith("R_") and "tip_fixed" not in n}
    assert len(coupled) == 8                    # flexor and DIP, both per finger
    assert frozen == {"R_Thumb_Yaw", "R_Thumb_Roll", "R_Thumb_Pitch",
                      "R_Thumb_Flexor", "R_Thumb_DIP"}


def test_one_servo_curls_a_whole_finger(hand):
    """The point of the coupling. Commanding only the knuckle must carry the
    middle phalanx and the tip with it, or the finger is a rigid rod again."""
    chain = hand.chain
    for finger in FINGERS:
        resolved = chain.resolve({f"R_{finger}_Pitch": 0.8})
        assert resolved[f"R_{finger}_Flexor"] == pytest.approx(0.8)
        assert resolved[f"R_{finger}_DIP"] == pytest.approx(0.8)


def test_the_coupling_chain_resolves_transitively(hand):
    """DIP follows the flexor, which follows the knuckle. A single pass over the
    joints would leave the DIP at zero and quietly half-straighten the finger."""
    chain = hand.chain
    dip = chain.joints["R_Index_DIP"]
    assert dip.mimic[0] == "R_Index_Flexor"
    assert chain.joints["R_Index_Flexor"].mimic[0] == "R_Index_Pitch"
    assert chain.resolve({"R_Index_Pitch": 0.5})["R_Index_DIP"] == pytest.approx(0.5)


def test_a_curled_finger_reaches_further_than_a_rigid_one(hand):
    """Fingertip travel is what the coupling buys: it sweeps an arc into the
    palm instead of pivoting about one knuckle.

    Measured in the finger's own root frame, so it says the same thing whichever
    palm the digit is bolted to.
    """
    root = "Index_Knuckle_Cross_1"
    tips = np.stack([
        hand.chain.position(dh.tip_frame("Index"), root, {"R_Index_Pitch": p})
        for p in np.linspace(0.0, 0.95, 20)
    ])
    travel = np.linalg.norm(tips[-1] - tips[0])
    assert travel > 0.09                        # ~110 mm curled, ~65 mm rigid
    # ... and it ends up nearer the knuckle than it started, not further out.
    assert np.linalg.norm(tips[-1]) < np.linalg.norm(tips[0]) - 0.02


def test_upstream_mimic_references_are_broken_and_repairable(upstream):
    """Every `<mimic>` upstream ships names a joint that does not exist -- it
    drops the `R_` the joints carry. Loading it strictly must fail."""
    from berkeley_humanoid_lite_motion.urdf import Chain

    with pytest.raises(KeyError):
        Chain.from_urdf(UPSTREAM_URDF)          # no repair: broken, and says so
    assert upstream.joints["R_Index_DIP"].mimic[0] == "R_Index_Flexor"


def test_frozen_thumb_carries_its_configured_angle(hand, upstream):
    """A frozen joint's angle is baked into its origin, so setting a non-zero
    angle in the config actually moves the thumb rather than being ignored."""
    import yaml

    config = yaml.safe_load(ATTACHMENT_CONFIG.read_text())
    configured = config["hand"]["thumb"]
    assert set(configured) == {"R_Thumb_Yaw", "R_Thumb_Roll", "R_Thumb_Pitch",
                               "R_Thumb_Flexor", "R_Thumb_DIP"}
    # The shipped config is all zeros, so the frozen thumb must sit exactly
    # where upstream's zero pose puts it. Compared against upstream rather than
    # against a written-down number, and from the thumb's own root, since where
    # the palm puts that root is a separate question.
    assert all(angle == 0.0 for angle in configured.values())
    root = "Thumb_Prox_KC_1"
    assert np.allclose(hand.chain.position(dh.THUMB_TIP_FRAME, root),
                       upstream.position(dh.THUMB_TIP_FRAME, root), atol=1e-9)


def test_servo_travel_is_tighter_than_upstream(hand, upstream):
    """The 8-servo build is not upstream with joints removed; it moves less."""
    lower, upper = hand.limits
    for i, name in enumerate(hand.joint_names):
        assert upper[i] - lower[i] < (upstream.joints[name].upper - upstream.joints[name].lower) + 1e-9
    assert np.isclose(upper[0], 0.95)      # pitch, vs 1.309 upstream
    assert np.isclose(upper[4], 0.30)      # yaw,   vs 0.349 upstream


def test_pinch_needs_the_thumb_refrozen(hand, upstream):
    """With the fingers curling, opposition becomes reachable -- but not at the
    thumb posture the config currently ships, which is joint zero."""
    at_zero = dh.opposition_gap(hand, upstream, np.zeros(4))
    _, best = dh.best_thumb_posture(hand, upstream, restarts=6)
    assert at_zero > 0.03                      # ~52 mm: nothing opposes as shipped
    assert best < 0.005                        # ~0 mm: a fingertip can meet the thumb
    assert best < at_zero
