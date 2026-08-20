#!/usr/bin/env python3
"""Read finger joint angles straight out of a video, by finding the bearings.

Each joint of the DexHand finger runs on a bearing whose dark ring sits on white
plastic, which a Hough circle transform picks out reliably. That removes the
clicking from the coupling measurement: point a fixed camera at a clamped
finger, sweep the knuckle, and this writes the angle table that
`fit_finger_coupling.py` fits.

WHAT IT CAN AND CANNOT MEASURE
------------------------------
Three bearings -- knuckle, PIP, DIP -- give two directions and so one angle
directly:

    flexor   between the proximal and middle phalanges   (knuckle, PIP, DIP)

The knuckle angle needs a palm reference, and `--base` supplies one as a fixed
pixel. That only holds if the module never moves in frame, which in practice it
does. **Without `--base` the palm reference is not needed at all**: the tool
writes the proximal phalanx's absolute direction, and since

    flexor = ratio * (proximal_direction - palm_direction)

a straight line through those points has the ratio as its slope and the unknown
palm direction folded into its intercept. `fit_finger_coupling.py` solves for
both. The module may then translate freely; only its *orientation* has to hold
still, which is a far weaker thing to ask of a bench setup than a fixed pixel.

Either way this measures `Flexor <- Pitch`, which is the ratio worth measuring:
`DIP <- Flexor` already comes from the vendor's own CAD at 1.0, where nothing
states the proximal one. The distal angle would need the fingertip, which has no
bearing and would have to be clicked per frame.

AIMING THE CAMERA: LOOK ALONG THE BEARING AXES
----------------------------------------------
The bearings sit on the finger's *sides*, and their axes are the flexion axes.
So the view that shows them as circles is exactly the view in which the finger
flexes in the image plane -- the one the measurement needs. That makes aiming
checkable by eye, with no protractor:

    you can see the bearing faces as circles  ->  the camera is right
    you see the back of the finger, no rings  ->  turn 90 degrees

A top-down shot of a finger lying on a desk fails this: the bearings are edge-on
and invisible, and the finger folds towards the camera rather than across it,
which is the worst case for foreshortening. Either put the camera at desk level
looking horizontally at the finger's side, or lay the module on its side and
keep the camera above.

If no frame yields three bearings, that is almost always what happened.

THE RIG
-------
The measurement is only as good as how still everything is:

* **Clamp the finger module**, and clamp it to something the camera can also
  see against. This is the step that gets skipped. In one clip the camera was
  rock steady -- 0.0 px of background motion between frames, 7 px of drift over
  32 seconds -- and the measurement still failed, because the module itself was
  being slid around the desk between poses. A fixed camera does not help if the
  thing it is pointed at moves.
* **Fix the camera** on a tripod or against something solid, square to the
  finger's flexion plane within about 10 degrees, and do not touch it.
* With both fixed, `--base` works and the palm reference is exact. Without a
  clamped module, omit `--base` and let the fit solve for the palm direction --
  but that still assumes the module's *orientation* holds, so it rescues
  translation, not sliding and turning.
* **Sweep slowly**, or step the knuckle and pause. Motion blur costs the
  detector both circles and precision.
* Plain background behind the finger. `--roi` also helps: the detector
  cheerfully finds wheels on a T-shirt.

USAGE
-----
    python3 scripts/motion/track_finger_joints.py sweep.mp4 \\
        --base 240,610 --roi 300,80,700,400 --finger Middle -o angles.csv
    python3 scripts/motion/fit_finger_coupling.py angles.csv

`--base` is any point on the palm along the finger's axis, proximal of the
knuckle, in pixels. With a fixed camera it is the same in every frame, so it is
given once. `--preview` writes annotated frames so the detections can be checked
before the numbers are believed.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source/berkeley_humanoid_lite_motion"))

from berkeley_humanoid_lite_motion.dexhand import FINGERS  # noqa: E402

#: Middle phalanx over proximal, from the URDF's 30.29 / 45.26 mm. Used to tell
#: the finger's bearings apart from every other circle in the room.
PHALANX_RATIO = 30.29 / 45.26

#: Hough settings to try, coarse to fine. How large a bearing appears depends
#: entirely on framing -- 13 px across a whole hand, 40 px for one finger filling
#: the frame -- and guessing wrong finds nothing at all, so the bands are swept
#: rather than configured. `param2` is the accumulator threshold: lower finds
#: more circles and more rubbish, which the shape filter then has to reject.
HOUGH_BANDS = [
    dict(dp=1.2, minDist=30, param1=120, param2=35, minRadius=7, maxRadius=22),
    dict(dp=1.2, minDist=50, param1=120, param2=35, minRadius=15, maxRadius=45),
    dict(dp=1.2, minDist=70, param1=120, param2=30, minRadius=30, maxRadius=80),
    dict(dp=1.2, minDist=20, param1=120, param2=25, minRadius=4, maxRadius=12),
]

#: How far the proximal-to-middle phalanx length ratio may stray from the
#: model's before a triple of circles is rejected as not being the finger.
#: Scale-free, so it works whether the hand fills the frame or sits in a corner,
#: which an absolute pixel spread does not. Loose, because it is identifying
#: circles here, not judging the view -- foreshortening moves this ratio too,
#: and `fit_finger_coupling.py` is what looks at that. Tight enough to reject
#: three evenly spaced circles, which is what most incidental triples look like.
PHALANX_RATIO_TOLERANCE = 0.25

#: The three bearings are the same part at the same distance, so they image at
#: nearly the same size -- 39, 40, 40 px in one clip. Background clutter that
#: happens to have finger-like spacing rarely also has matched radii, and this
#: is what stops a chair leg and a mouse from being mistaken for a finger.
RADIUS_CONSISTENCY = 1.6

#: A finger does not teleport. Once found, triples whose centroid is further
#: than this many bearing-radii from the last one are penalised, which keeps the
#: detector on the finger when the background offers a better-shaped triple.
CONTINUITY_RADII = 6.0

#: How far a frame's knuckle-to-DIP span may sit from the clip's median before
#: it is dropped. The shape test is scale-free on purpose, so it happily accepts
#: a correctly proportioned triple a third the right size; this catches those.
SCALE_TOLERANCE = 0.25


def chain_order(q):
    """Three points ordered along their chain, with the two segment lengths.

    The chain is the arrangement with the shortest total path: for three points
    that is always the true chain, however far the finger is bent.
    """
    order = min(itertools.permutations(range(3)),
                key=lambda o: (np.linalg.norm(q[o[0]] - q[o[1]])
                               + np.linalg.norm(q[o[1]] - q[o[2]])))
    p = q[list(order)]
    return p, np.array([np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[2] - p[1])])


def _finger_triple(circles, expected_ratio, previous=None):
    """The triple of circles best shaped like a finger, or None.

    A detector pointed at a room finds plenty of circles -- a mouse, a chair leg,
    the rim of a desk -- and enough of them that some triple will have
    finger-like spacing by chance. Three tests together are what separate the
    finger from the furniture:

    * the two segment lengths match the finger's own proportions,
    * the three bearings image at nearly the same size, as one part at one
      distance must,
    * and the triple has not jumped across the frame since the last one.

    Spacing alone is not enough; it was tried, and the tracker spent a clip
    following the office behind the hand.
    """
    best = None
    for triple in itertools.combinations(range(len(circles)), 3):
        chosen = circles[list(triple)]
        radii = chosen[:, 2]
        if radii.min() <= 0 or radii.max() / radii.min() > RADIUS_CONSISTENCY:
            continue
        ordered, lengths = chain_order(chosen[:, :2])
        if lengths.min() < 5.0:
            continue
        # Either end may be the knuckle, so score whichever reading is closer.
        error = min(abs(lengths[1] / lengths[0] - expected_ratio),
                    abs(lengths[0] / lengths[1] - expected_ratio))
        if error >= PHALANX_RATIO_TOLERANCE:
            continue
        if previous is not None:
            jump = np.linalg.norm(ordered.mean(axis=0) - previous.mean(axis=0))
            error += max(0.0, jump / (radii.mean() * CONTINUITY_RADII) - 1.0)
        if best is None or error < best[0]:
            best = (error, ordered)
    return None if best is None else best[1]


def find_joints(gray, roi=None, hough=None, expected_ratio=None, previous=None):
    """The three bearing centres in one frame, ordered along the chain, or None.

    Sweeps the radius bands unless `hough` pins one, and returns the settings
    that worked alongside the joints so a caller can try them first next time.
    `previous` is the last frame's joints, used to keep the detector on the
    finger rather than on whatever the background offers.
    """
    import cv2

    offset = np.zeros(2)
    if roi is not None:
        x, y, w, h = roi
        gray = gray[y:y + h, x:x + w]
        offset = np.array([x, y], dtype=float)
    gray = cv2.medianBlur(gray, 5)
    expected_ratio = PHALANX_RATIO if expected_ratio is None else expected_ratio

    # A pinned setting is tried first, then the rest -- framing can change
    # mid-clip, and re-sweeping only when the fast path fails costs nothing.
    bands = ([hough] if hough else []) + [b for b in HOUGH_BANDS if b != hough]
    for band in bands:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, **band)
        if circles is None or len(circles[0]) < 3:
            continue
        found = _finger_triple(circles[0].astype(float), expected_ratio, previous)
        if found is not None:
            return found + offset, band
    return None, None


def angle_between(a, b, c) -> float:
    """Signed angle at `b` between segment a->b and segment b->c, in degrees."""
    u, v = b - a, c - b
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    u, v = u / nu, v / nv
    return float(np.degrees(np.arctan2(u[0] * v[1] - u[1] * v[0], u @ v)))


def measure(joints, base=None) -> tuple[float, float, np.ndarray]:
    """(pitch, flexor) in degrees, and the joints reordered knuckle-first.

    With a `base`, pitch is the knuckle angle measured from it, and which end of
    the chain is the knuckle is decided by which is nearer. Without one, pitch is
    the proximal phalanx's absolute direction in the image -- the knuckle angle
    plus an unknown constant, which the fit removes -- and the chain keeps the
    order it came in, since nothing distinguishes its ends.
    """
    if base is not None:
        if np.linalg.norm(joints[0] - base) > np.linalg.norm(joints[-1] - base):
            joints = joints[::-1]
    else:
        # The proximal phalanx is half again as long as the middle one, so the
        # longer segment names the knuckle end. Without this the two ends swap
        # from frame to frame and the direction jumps by 180 degrees, which no
        # amount of fitting recovers from.
        if np.linalg.norm(joints[1] - joints[0]) < np.linalg.norm(joints[2] - joints[1]):
            joints = joints[::-1]
    knuckle, pip, dip = joints
    if base is not None:
        return angle_between(base, knuckle, pip), angle_between(knuckle, pip, dip), joints
    direction = pip - knuckle
    return (float(np.degrees(np.arctan2(-direction[1], direction[0]))),
            angle_between(knuckle, pip, dip), joints)


def frames_from(source: Path, step: int):
    """(index, BGR frame) from a video file or a directory of images."""
    import cv2

    if source.is_dir():
        for i, path in enumerate(sorted(source.iterdir())):
            if i % step == 0:
                image = cv2.imread(str(path))
                if image is not None:
                    yield i, image
        return
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {source}")
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            yield index, frame
        index += 1
    capture.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="video file, or a directory of frames")
    parser.add_argument("--base",
                        help="palm reference point 'u,v' in pixels. Omit unless the "
                             "module is genuinely fixed in frame -- without it the "
                             "palm direction is solved for during the fit")
    parser.add_argument("--roi", help="restrict detection to 'x,y,w,h'")
    parser.add_argument("--finger", default="Middle", choices=list(FINGERS))
    parser.add_argument("--step", type=int, default=6,
                        help="use every Nth frame (default: 6)")
    parser.add_argument("-o", "--output", type=Path, default=Path("angles.csv"))
    parser.add_argument("--preview", type=Path,
                        help="directory to write annotated frames into")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        raise SystemExit("this needs opencv: pip install opencv-python-headless")

    def numbers(text, count, flag):
        try:
            values = [float(v) for v in text.split(",")]
        except ValueError:
            values = []
        if len(values) != count:
            raise SystemExit(f"{flag} wants {count} comma-separated numbers, got {text!r}")
        return values

    base = np.array(numbers(args.base, 2, "--base")) if args.base else None
    roi = [int(v) for v in numbers(args.roi, 4, "--roi")] if args.roi else None
    if args.preview:
        args.preview.mkdir(parents=True, exist_ok=True)

    rows, seen, missed, band, previous = [], 0, 0, None, None
    detections = []
    for index, frame in frames_from(args.source, args.step):
        seen += 1
        joints, band = find_joints(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), roi,
                                   band, previous=previous)
        if joints is None:
            missed += 1
            continue
        previous = joints
        detections.append((index, joints, frame if args.preview else None))

    # The camera does not change distance mid-sweep, so the finger's apparent
    # size does not either. A triple a third the usual size is a different set
    # of circles, whatever its shape -- this is what the per-frame shape test
    # cannot see, since it is scale-free by design.
    if detections:
        spans_px = np.array([np.linalg.norm(j[2] - j[0]) for _, j, _ in detections])
        typical = np.median(spans_px)
        keep = np.abs(spans_px - typical) <= SCALE_TOLERANCE * typical
        missed += int((~keep).sum())
        detections = [d for d, ok in zip(detections, keep) if ok]

    for index, joints, frame in detections:
        pitch, flexor, ordered = measure(joints, base)
        rows.append((index, pitch, flexor))
        if frame is not None:
            marked = frame.copy()
            for point in ordered:
                cv2.circle(marked, tuple(point.astype(int)), 14, (0, 0, 255), 2)
            path = ordered if base is None else np.vstack([base, ordered])
            if base is not None:
                cv2.circle(marked, tuple(base.astype(int)), 6, (255, 0, 0), -1)
            cv2.polylines(marked, [path.astype(np.int32)], False, (0, 255, 0), 2)
            cv2.imwrite(str(args.preview / f"{index:05d}.jpg"), marked)

    if not rows:
        raise SystemExit(
            f"no frame in {args.source} gave three bearings.\n"
            "\n"
            "Most likely the camera is not looking along the bearing axes. The\n"
            "bearings are on the finger's sides; if the shot shows the back of\n"
            "the finger they are edge-on and invisible -- and that same view is\n"
            "the one where the finger folds towards the camera instead of across\n"
            "it, so it could not be measured even if they were found. Turn 90\n"
            "degrees: you should see the bearing faces as circles.\n"
            "\n"
            "Failing that, check --roi, and re-run with --preview to look.")

    # Flexion is one direction; whichever sign the most-bent frame came out as is
    # the one that means curled. An absolute direction carries the palm's
    # orientation too, so it must not be flipped -- only the joint angle is.
    sign = np.sign(sum(f for _, _, f in rows)) or 1.0
    reference = "palm" if base is not None else "absolute"
    with args.output.open("w") as handle:
        handle.write("finger,pitch,flexor,dip,pitch_reference,frame\n")
        for index, pitch, flexor in rows:
            written = sign * pitch if base is not None else pitch
            handle.write(f"{args.finger},{written:.3f},{sign * flexor:.3f},,"
                         f"{reference},{index}\n")

    print(f"{len(rows)} of {seen} sampled frames gave three bearings "
          f"({missed} rejected) -> {args.output}")
    spans = [p if base is None else sign * p for _, p, _ in rows]
    label = "proximal phalanx direction" if base is None else "knuckle angle"
    print(f"{label} spans {min(spans):.1f} to {max(spans):.1f} deg")
    print()
    print("Look at the preview frames before trusting this. Then:")
    print(f"  python3 scripts/motion/fit_finger_coupling.py {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
