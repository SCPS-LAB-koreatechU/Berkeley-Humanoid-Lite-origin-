"""Retargeting human keypoints onto the eight servos."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from berkeley_humanoid_lite_motion.hand_retarget import (
    FINGER_MCP, FINGER_PIP, FINGER_TIP, THUMB_TIP, WRIST,
    HandRetargeter, RetargetWeights, aperture, curl, palm_basis,
)
from berkeley_humanoid_lite_motion.dexhand import FINGERS


def synthetic_hand(retargeter, q, human_width=0.085, seed=0, forearm_ratio=1.05):
    """A 21-keypoint human hand that the given robot pose `q` exactly explains.

    Built backwards: take the robot's fingertips, undo the size normalisation,
    and place a human hand around them, then move the whole thing somewhere
    arbitrary. Anything the retargeter recovers from this it recovered from
    geometry, not from the frame it happened to be handed.
    """
    # Every length scales with the hand, or the size-invariance the retargeter
    # relies on would be broken by the fixture rather than tested by it.
    forearm = human_width * forearm_ratio
    scale = human_width / retargeter.robot_frame.width
    local_tips = retargeter.robot_frame.to_local(retargeter.hand.fingertips(q)) * scale

    kp = np.zeros((21, 3))
    kp[WRIST] = 0.0
    half = human_width / 2.0
    # Knuckles laid out so palm_basis() of this hand is the identity.
    for finger, offset in zip(FINGERS, np.linspace(half, -half, len(FINGERS))):
        kp[FINGER_MCP[finger]] = [offset, 0.0, forearm]
    for i, finger in enumerate(FINGERS):
        kp[FINGER_TIP[finger]] = local_tips[i]
        kp[FINGER_PIP[finger]] = kp[FINGER_MCP[finger]] + (local_tips[i] - kp[FINGER_MCP[finger]]) / 3.0
        kp[FINGER_TIP[finger] - 1] = kp[FINGER_MCP[finger]] + (local_tips[i] - kp[FINGER_MCP[finger]]) * 0.7
    kp[THUMB_TIP] = [half * 1.2, half * 0.5, forearm * 0.6]

    rng = np.random.default_rng(seed)
    R = Rotation.random(random_state=seed).as_matrix()
    return kp @ R.T + rng.normal(0.0, 0.3, 3)


@pytest.fixture
def retargeter(hand):
    return HandRetargeter(hand=hand, weights=RetargetWeights(smoothness=0.0, rest=0.0))


def test_palm_basis_is_orthonormal_and_right_handed():
    R = palm_basis(np.array([0.0, 0.0, 0.0]), np.array([0.04, 0.01, 0.09]),
                   np.array([-0.04, -0.02, 0.08]))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_robot_palm_width_matches_the_hardware(retargeter):
    """65 mm across the knuckles. The scale every capture is normalised to."""
    assert 0.06 < retargeter.robot_frame.width < 0.07


@pytest.mark.parametrize("seed", range(8))
def test_reachable_poses_are_recovered(retargeter, seed):
    lower, upper = retargeter.hand.limits
    q = np.random.default_rng(seed).uniform(lower, upper)
    solved, error = retargeter.solve_targets(retargeter.targets(synthetic_hand(retargeter, q, seed=seed)))
    assert error.max() < 1e-3          # 1 mm
    assert np.abs(solved - q).max() < 1e-2


def test_retarget_ignores_where_the_hand_was_and_how_big_it_was(retargeter):
    """Two captures of the same gesture -- different person, different place --
    must give the same servo command."""
    lower, upper = retargeter.hand.limits
    q = np.random.default_rng(3).uniform(lower, upper)
    a, _ = retargeter.solve_targets(retargeter.targets(synthetic_hand(retargeter, q, 0.085, seed=1)))
    b, _ = retargeter.solve_targets(retargeter.targets(synthetic_hand(retargeter, q, 0.110, seed=2)))
    assert np.abs(a - b).max() < 1e-2


def test_a_human_pose_the_hand_cannot_hold_reports_its_shortfall(retargeter):
    """A fist is not reachable -- the fingers are rigid. The residual is the
    measure of that, and it must not be quietly zero."""
    lower, upper = retargeter.hand.limits
    kp = synthetic_hand(retargeter, upper, seed=5)
    for finger in FINGERS:                     # curl every fingertip into the palm
        kp[FINGER_TIP[finger]] = kp[WRIST] + (kp[FINGER_MCP[finger]] - kp[WRIST]) * 0.3
    _, error = retargeter.solve_targets(retargeter.targets(kp))
    assert error.max() > 0.02                  # centimetres, not millimetres


def test_sequence_carries_the_signals_the_robot_cannot_hold(retargeter):
    lower, upper = retargeter.hand.limits
    rng = np.random.default_rng(0)
    frames = np.stack([synthetic_hand(retargeter, rng.uniform(lower, upper), seed=s) for s in range(6)])
    motion = retargeter.solve_sequence(frames, fps=30.0)
    assert motion.joint_pos.shape == (6, 8)
    assert motion.joint_names == retargeter.hand.joint_names
    assert set(motion.extras) == {"fingertip_error", "aperture", "curl"}
    assert motion.extras["curl"].shape == (6, 4)
    assert np.isfinite(motion.extras["aperture"]).all()


def test_aperture_and_curl_are_invariant_to_pose_and_size(retargeter):
    lower, upper = retargeter.hand.limits
    q = np.random.default_rng(7).uniform(lower, upper)
    small = synthetic_hand(retargeter, q, 0.080, seed=11)[None]
    large = synthetic_hand(retargeter, q, 0.120, seed=12)[None]
    assert np.isclose(aperture(small)[0], aperture(large)[0], rtol=1e-6)
    assert np.allclose(curl(small)[0], curl(large)[0], atol=1e-6)


def test_confidence_zero_lets_a_lost_fingertip_go(retargeter):
    """A dropped keypoint must stop influencing the solve entirely.

    Not that the servo lands on the truth -- with the keypoint gone there is
    nothing to land on, and it falls back on smoothness and the rest pose. What
    must hold is that whatever garbage the tracker left behind changes nothing.
    """
    weighted = HandRetargeter(hand=retargeter.hand)
    lower, upper = retargeter.hand.limits
    q = np.random.default_rng(2).uniform(lower, upper)
    targets = retargeter.targets(synthetic_hand(retargeter, q, seed=2))

    def solve(index_target, confidence):
        corrupted = targets.copy()
        corrupted[0] = index_target
        return weighted.solve_targets(corrupted, q_init=q, confidence=confidence)[0]

    a, b = targets[0] + 0.5, targets[0] - 0.5
    assert np.allclose(solve(a, [0, 1, 1, 1]), solve(b, [0, 1, 1, 1]), atol=1e-6)
    assert not np.allclose(solve(a, [1, 1, 1, 1]), solve(b, [1, 1, 1, 1]), atol=1e-3)
