#!/usr/bin/env python3
"""Read finger joint angles straight out of a video, by finding the bearings.

Each joint of the DexHand finger runs on a bearing whose dark ring sits on white
plastic, which a Hough circle transform picks out reliably. That removes the
clicking from the coupling measurement: point a fixed camera at a clamped
finger, sweep the knuckle, and this writes the angle table that
`fit_finger_coupling.py` fits.

WHAT IT CAN AND CANNOT MEASURE
------------------------------
Three bearings -- knuckle, PIP, DIP -- plus one fixed reference point on the
palm give two of the three angles:

    pitch    between the palm and the proximal phalanx   (base, knuckle, PIP)
    flexor   between the proximal and middle phalanges   (knuckle, PIP, DIP)

which is exactly what `Flexor <- Pitch` needs, and that is the ratio actually
worth measuring: `DIP <- Flexor` already comes from the vendor's own CAD at 1.0,
where nothing states the proximal one. The distal angle would need the
fingertip, which has no bearing and would have to be clicked per frame.

THE RIG
-------
The measurement is only as good as how still everything is:

* **Clamp the finger module.** Handheld does not work. A finger held in the hand
  is reoriented between poses, and every reorientation foreshortens the
  projection differently -- in one 57 s handheld clip the proximal-to-middle
  phalanx length ratio, which is a constant of the hardware, ranged over
  0.63 to 0.88.
* **Fix the camera** on a tripod or against something solid, square to the
  finger's flexion plane within about 10 degrees, and do not touch it.
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

#: Hough parameters. `param2` is the accumulator threshold -- lower finds more
#: circles and more rubbish. These were set against 1280x720 footage of the hand
#: filling about a third of the frame; scale the radii if yours differs.
HOUGH = dict(dp=1.2, minDist=30, param1=120, param2=35, minRadius=7, maxRadius=22)

#: How far the proximal-to-middle phalanx length ratio may stray from the
#: model's before a triple of circles is rejected as not being the finger.
#: Scale-free, so it works whether the hand fills the frame or sits in a corner,
#: which an absolute pixel spread does not. Loose, because it is identifying
#: circles here, not judging the view -- foreshortening moves this ratio too,
#: and `fit_finger_coupling.py` is what looks at that. Tight enough to reject
#: three evenly spaced circles, which is what most incidental triples look like.
PHALANX_RATIO_TOLERANCE = 0.25


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


def find_joints(gray, roi=None, hough=None, expected_ratio=None):
    """The three bearing centres in one frame, ordered along the chain, or None.

    Chooses the triple of detected circles whose two segment lengths best match
    the finger's own proportions. A detector pointed at a room finds plenty of
    circles -- wheels on a T-shirt, a fan grille -- and picking the triple that
    is shaped like a finger rejects them without needing to know how large the
    finger appears or where in the frame it sits.
    """
    import cv2

    offset = np.zeros(2)
    if roi is not None:
        x, y, w, h = roi
        gray = gray[y:y + h, x:x + w]
        offset = np.array([x, y], dtype=float)

    circles = cv2.HoughCircles(cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT,
                               **(hough or HOUGH))
    if circles is None or len(circles[0]) < 3:
        return None
    points = circles[0][:, :2].astype(float)
    if expected_ratio is None:
        expected_ratio = PHALANX_RATIO

    best = None
    for triple in itertools.combinations(range(len(points)), 3):
        ordered, lengths = chain_order(points[list(triple)])
        if lengths.min() < 5.0:
            continue
        # Either end may be the knuckle, so score whichever reading is closer.
        error = min(abs(lengths[1] / lengths[0] - expected_ratio),
                    abs(lengths[0] / lengths[1] - expected_ratio))
        if error < PHALANX_RATIO_TOLERANCE and (best is None or error < best[0]):
            best = (error, ordered)
    return None if best is None else best[1] + offset


def angle_between(a, b, c) -> float:
    """Signed angle at `b` between segment a->b and segment b->c, in degrees."""
    u, v = b - a, c - b
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    u, v = u / nu, v / nv
    return float(np.degrees(np.arctan2(u[0] * v[1] - u[1] * v[0], u @ v)))


def measure(joints, base) -> tuple[float, float, np.ndarray]:
    """(pitch, flexor) in degrees, and the joints reordered knuckle-first.

    Which end of the chain is the knuckle is decided by which is nearer the palm
    reference, so the caller does not have to know which way the finger points.
    """
    if np.linalg.norm(joints[0] - base) > np.linalg.norm(joints[-1] - base):
        joints = joints[::-1]
    knuckle, pip, dip = joints
    return angle_between(base, knuckle, pip), angle_between(knuckle, pip, dip), joints


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
    parser.add_argument("--base", required=True,
                        help="palm reference point 'u,v' in pixels")
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

    base = np.array([float(v) for v in args.base.split(",")])
    roi = [int(v) for v in args.roi.split(",")] if args.roi else None
    if args.preview:
        args.preview.mkdir(parents=True, exist_ok=True)

    rows, seen, missed = [], 0, 0
    for index, frame in frames_from(args.source, args.step):
        seen += 1
        joints = find_joints(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), roi)
        if joints is None:
            missed += 1
            continue
        pitch, flexor, ordered = measure(joints, base)
        rows.append((index, pitch, flexor))
        if args.preview:
            marked = frame.copy()
            for point in ordered:
                cv2.circle(marked, tuple(point.astype(int)), 14, (0, 0, 255), 2)
            cv2.circle(marked, tuple(base.astype(int)), 6, (255, 0, 0), -1)
            cv2.polylines(marked, [np.vstack([base, ordered]).astype(np.int32)],
                          False, (0, 255, 0), 2)
            cv2.imwrite(str(args.preview / f"{index:05d}.jpg"), marked)

    if not rows:
        raise SystemExit(f"no frame in {args.source} gave three bearings; "
                         "check --roi, and look at --preview output")

    # Flexion is one direction; whichever sign the most-bent frame came out as is
    # the one that means curled.
    sign = np.sign(sum(f for _, _, f in rows)) or 1.0
    with args.output.open("w") as handle:
        handle.write("finger,pitch,flexor,dip,frame\n")
        for index, pitch, flexor in rows:
            handle.write(f"{args.finger},{sign * pitch:.3f},{sign * flexor:.3f},,{index}\n")

    print(f"{len(rows)} of {seen} sampled frames gave three bearings "
          f"({missed} rejected) -> {args.output}")
    print(f"pitch spans {min(sign * p for _, p, _ in rows):.1f} to "
          f"{max(sign * p for _, p, _ in rows):.1f} deg")
    print()
    print("Look at the preview frames before trusting this. Then:")
    print(f"  python3 scripts/motion/fit_finger_coupling.py {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
