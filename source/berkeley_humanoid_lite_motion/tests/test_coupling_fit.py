"""The coupling measurement tool.

Its statistics went through two wrong versions, so they are pinned here. Every
case is synthetic: a finger built from the model's own segment lengths, bent by
known ratios, projected, and clicked at with a known error.
"""

import importlib.util

import numpy as np
import pytest

from conftest import REPO

FINGERS = ("Index", "Middle", "Ring", "Pinky")
TRUE_FLEXOR, TRUE_DIP = 0.83, 0.656
#: Palm stub plus the three phalanges, matching the URDF's 45.26/30.29/18.20 mm.
SEGMENTS_MM = np.array([40.0, 45.26, 30.29, 18.20])


@pytest.fixture(scope="module")
def tool():
    path = REPO / "scripts/motion/fit_finger_coupling.py"
    spec = importlib.util.spec_from_file_location("fit_finger_coupling", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clicked(tilt_deg=0.0, click_px=2.0, px_per_mm=9.0, poses=9, seed=0):
    """{finger: (N, 10)} clicked joint centres for a finger obeying the ratios.

    `tilt_deg` swings the finger's flexion plane away from the camera, which is
    the error the fit cannot see for itself.
    """
    rng = np.random.default_rng(seed)
    t = np.radians(tilt_deg)
    R = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    out = {}
    for finger in FINGERS:
        rows = []
        for pitch in np.linspace(0.05, 0.95, poses):
            turns = np.cumsum([0.0, pitch, TRUE_FLEXOR * pitch,
                               TRUE_DIP * TRUE_FLEXOR * pitch])
            position, points = np.zeros(3), []
            for length, angle in zip(SEGMENTS_MM * px_per_mm, turns):
                points.append(position.copy())
                position = position + length * np.array([np.cos(angle), 0.0, np.sin(angle)])
            points.append(position)
            image = (np.array(points) @ R.T)[:, [0, 2]] * [1, -1] + [200.0, 900.0]
            rows.append((image + rng.normal(0, click_px, (5, 2))).ravel())
        out[finger] = np.array(rows)
    return out


def fitted(tool, points):
    data = {f: tool.angles_from_points(v) for f, v in points.items()}
    fit = tool.fit_from_angles(data, np.pi / 180.0)
    return (np.mean([fit[f]["flexor"][0] for f in fit]),
            np.mean([fit[f]["dip"][0] for f in fit]))


def test_angles_come_out_of_clicked_points(tool):
    """Three angles per pose, each between a segment and the one before it."""
    angles = tool.angles_from_points(clicked(click_px=0.0)["Middle"])
    assert angles.shape == (9, 3)
    pitch = np.linspace(0.05, 0.95, 9)
    assert np.allclose(angles[:, 0], np.degrees(pitch), atol=1e-6)
    assert np.allclose(angles[:, 1], np.degrees(TRUE_FLEXOR * pitch), atol=1e-6)


def test_camera_roll_does_not_change_the_angles(tool):
    """Angles are between segments, so tipping the camera about its own axis --
    which happens on every handheld shot -- must not move them."""
    points = clicked(click_px=0.0)["Index"].reshape(9, 5, 2)
    turn = np.array([[np.cos(0.6), -np.sin(0.6)], [np.sin(0.6), np.cos(0.6)]])
    rolled = (points @ turn.T).reshape(9, 10)
    assert np.allclose(tool.angles_from_points(points.reshape(9, 10)),
                       tool.angles_from_points(rolled), atol=1e-9)


def test_a_square_view_recovers_the_ratios(tool):
    flexor, dip = fitted(tool, clicked())
    assert abs(flexor - TRUE_FLEXOR) < 0.02
    assert abs(dip - TRUE_DIP) < 0.03


def test_foreshortening_biases_the_ratios_low(tool):
    """Why the check below exists: the fit is confidently wrong, not noisy."""
    flexor, _ = fitted(tool, clicked(tilt_deg=60.0, click_px=0.0))
    assert flexor < TRUE_FLEXOR - 0.35


def test_foreshortening_check_is_quiet_on_a_square_view(tool, hand):
    """A false alarm here sends someone reshooting good data, so it is checked
    at sloppy clicking and at a small image too."""
    for click, scale in ((2.0, 9.0), (5.0, 9.0), (2.0, 4.0)):
        for seed in range(5):
            skew = tool.check_foreshortening(
                hand, clicked(click_px=click, px_per_mm=scale, seed=seed))
            assert skew <= tool.FORESHORTENING_TOLERANCE


def test_foreshortening_check_catches_a_badly_aimed_camera(tool, hand):
    for tilt in (45.0, 60.0):
        skew = tool.check_foreshortening(hand, clicked(tilt_deg=tilt))
        assert skew > tool.FORESHORTENING_TOLERANCE


def test_linearity_test_does_not_cry_curvature_over_noise(tool):
    """The first version compared how much a quadratic improved the fit, and
    called clean linear data curved: an extra term always fits noise a little
    better."""
    data = {f: tool.angles_from_points(v) for f, v in clicked(click_px=3.0).items()}
    assert tool.pooled_diagnostics(data, np.pi / 180.0)["curvature_p"] > tool.CURVATURE_ALPHA


def test_linearity_test_sees_a_real_curve(tool):
    """ALLEX's measured DIP-follows-PIP quartic, sampled finely enough to resolve.
    The second version tested against a quadratic and missed this entirely --
    the curve is odd-symmetric about the neutral pose."""
    pitch = np.linspace(0.05, 0.95, 24)
    rng = np.random.default_rng(0)
    data = {}
    for finger in FINGERS:
        flexor = TRUE_FLEXOR * pitch
        dip = 0.4269 * flexor + 0.0659 * flexor**2 + 0.136 * flexor**3 - 0.0462 * flexor**4
        noise = np.radians(0.5)
        data[finger] = np.degrees(np.stack([
            pitch + rng.normal(0, noise, len(pitch)),
            flexor + rng.normal(0, noise, len(pitch)),
            dip + rng.normal(0, noise, len(pitch)),
        ], axis=1))
    assert tool.pooled_diagnostics(data, np.pi / 180.0)["curvature_p"] < tool.CURVATURE_ALPHA


def test_a_bad_zero_pose_is_reported_as_an_offset(tool):
    pitch = np.linspace(0.05, 0.95, 9)
    data = {f: np.degrees(np.stack([pitch, TRUE_FLEXOR * pitch,
                                    TRUE_DIP * TRUE_FLEXOR * pitch + np.radians(6.0)], axis=1))
            for f in FINGERS}
    assert tool.pooled_diagnostics(data, np.pi / 180.0)["offset_deg"] > 4.0
