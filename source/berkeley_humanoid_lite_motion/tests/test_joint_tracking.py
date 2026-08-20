"""Reading joint angles out of video by finding the bearings.

Frames are drawn rather than filmed: a finger of known proportions bent by a
known ratio, with a dark ring at each joint. That makes the recovered ratio
checkable, which is the only way to know the tracker is measuring rather than
producing plausible numbers.
"""

import importlib.util

import numpy as np
import pytest

from conftest import REPO

cv2 = pytest.importorskip("cv2", reason="joint tracking needs opencv")

TRUE_FLEXOR = 0.83
#: Base stub, proximal and middle phalanx in pixels. The last two are in the
#: URDF's 45.26 : 30.29 proportion, which is what the tracker keys on.
SEGMENTS = np.array([160.0, 260.0, 174.0])
BASE = np.array([120.0, 400.0])


@pytest.fixture(scope="module")
def tracker():
    path = REPO / "scripts/motion/track_finger_joints.py"
    spec = importlib.util.spec_from_file_location("track_finger_joints", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render(pitch, decoys=(), blur=0):
    """One frame of a finger bent to `pitch` radians, and its true joint centres."""
    image = np.full((600, 900, 3), 245, np.uint8)
    turns = np.cumsum([0.0, pitch, TRUE_FLEXOR * pitch])
    position, points = BASE.copy(), [BASE.copy()]
    for length, angle in zip(SEGMENTS, turns):
        position = position + length * np.array([np.cos(angle), -np.sin(angle)])
        points.append(position.copy())
    for a, b in zip(points[:-1], points[1:]):
        cv2.line(image, tuple(a.astype(int)), tuple(b.astype(int)), (255, 255, 255), 26)
        cv2.line(image, tuple(a.astype(int)), tuple(b.astype(int)), (210, 210, 210), 22)
    for point in points[1:]:
        cv2.circle(image, tuple(point.astype(int)), 13, (40, 40, 40), 4)
        cv2.circle(image, tuple(point.astype(int)), 7, (150, 150, 150), -1)
    for centre, radius in decoys:
        cv2.circle(image, centre, radius, (60, 60, 60), 4)
    if blur:
        image = cv2.GaussianBlur(image, (blur, blur), 0)
    return image, np.array(points[1:])


def test_finds_the_three_bearings(tracker):
    image, truth = render(0.5)
    found = tracker.find_joints(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    assert found is not None
    assert np.allclose(found, truth, atol=3.0)


def test_ignores_circles_that_are_not_finger_shaped(tracker):
    """A room is full of circles -- wheels on a shirt, a fan grille. The triple
    has to be picked by shape, not by being the only one."""
    decoys = [((800, 80), 16), ((820, 140), 14), ((760, 200), 18), ((60, 60), 15)]
    image, truth = render(0.5, decoys=decoys)
    found = tracker.find_joints(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    assert found is not None
    assert np.allclose(found, truth, atol=3.0)


def test_a_frame_with_no_finger_is_rejected(tracker):
    image = np.full((600, 900, 3), 245, np.uint8)
    for centre, radius in (((300, 300), 15), ((500, 300), 15), ((700, 300), 15)):
        cv2.circle(image, centre, radius, (60, 60, 60), 4)
    # Evenly spaced circles are a 1.0 length ratio, nothing like a finger's 0.67.
    assert tracker.find_joints(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) is None


def test_chain_order_survives_a_folded_finger(tracker):
    """At full flexion the tip comes back towards the knuckle, and a nearest
    neighbour ordering would put the joints in the wrong sequence."""
    _, truth = render(0.95)
    scrambled = truth[[2, 0, 1]]
    ordered, lengths = tracker.chain_order(scrambled)
    assert np.allclose(ordered, truth, atol=1e-9) or np.allclose(ordered, truth[::-1], atol=1e-9)
    assert lengths.min() / lengths.max() == pytest.approx(SEGMENTS[2] / SEGMENTS[1], abs=0.02)


def test_which_end_is_the_knuckle_comes_from_the_palm_reference(tracker):
    _, truth = render(0.6)
    for joints in (truth, truth[::-1]):
        pitch, flexor, ordered = tracker.measure(joints, BASE)
        assert np.allclose(ordered[0], truth[0])
        # Signed, and which sign means "curled" depends on how the camera was
        # held; the caller normalises that across the whole sequence.
        assert abs(pitch) == pytest.approx(np.degrees(0.6), abs=1.0)
        assert abs(flexor) == pytest.approx(np.degrees(TRUE_FLEXOR * 0.6), abs=1.0)


@pytest.mark.parametrize("blur", [0, 5])
def test_a_swept_sequence_recovers_the_ratio(tracker, blur):
    """The end to end claim: frames in, coupling ratio out."""
    pitches, flexors = [], []
    for pitch in np.linspace(0.05, 0.95, 12):
        image, _ = render(pitch, blur=blur)
        found = tracker.find_joints(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        if found is None:
            continue
        measured_pitch, measured_flexor, _ = tracker.measure(found, BASE)
        pitches.append(measured_pitch)
        flexors.append(measured_flexor)
    # Blur costs detections -- a real sweep should be shot slowly enough to
    # avoid it -- but it must not move the ratio of the frames that survive.
    assert len(pitches) >= 6
    p, f = np.radians(pitches), np.radians(flexors)
    assert float(p @ f / (p @ p)) == pytest.approx(TRUE_FLEXOR, abs=0.03)
