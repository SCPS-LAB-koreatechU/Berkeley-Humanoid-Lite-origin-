#!/usr/bin/env python3
"""Measure the DexHand's finger coupling ratios.

`hand.coupled` in `config/arm_attachment.yaml` says how far a finger curls for a
given servo command, and it ships as an estimate. Everything downstream moves
with it -- fingertip reach, whether the hand can oppose its thumb, what a
retargeted grasp actually does -- so it is worth measuring.

WHAT TO MEASURE, AND WITH WHAT
------------------------------
Measure the **joint angles**, not the fingertip position, and a side-view
photograph beats the motion capture rig. The ratios *are* angle ratios; reading
them off a fingertip position dilutes the distal stage through an 18 mm lever
and loses most of the precision. Per-parameter 1-sigma from a 9-pose sweep:

    fingertip markers, 2.0 mm        Flexor +/-0.061    DIP +/-0.273
    fingertip markers, 0.5 mm        Flexor +/-0.015    DIP +/-0.068
    angles read to +/-2 deg          Flexor +/-0.021    DIP +/-0.030
    joint centres clicked, +/-5 px   Flexor +/-0.032    DIP +/-0.049
    joint centres clicked, +/-2 px   Flexor +/-0.013    DIP +/-0.022

Clicking joint centres in an image beats holding a protractor to it, so that is
the mode to use (`--from-points`). Nothing needs to be calibrated and the
commanded servo angle is not needed either -- every angle, including the
knuckle's, is measured from the same photo.

THE PROCEDURE
-------------
0.  **First check the coupling exists at all.** Hold the hand still and command
    one knuckle from 0 to its limit while you watch the middle and distal
    segments. If they do not rotate relative to the segment before them, the
    fingers are rigid on this build: set both multipliers to 0 and stop here.

1.  Point a camera along the hand's **y axis** -- straight across the palm, so a
    finger's flexion is in the image plane. Each finger's three flexion axes are
    exactly parallel, so one view reads all three angles undistorted. Viewing
    along the palm's y rather than the finger's own axis costs at most 0.4 deg
    of projection error (pinky, the worst; 0.1 deg index and ring; the middle
    finger's axes are exactly along y, so use it if you measure only one).

    **This is the error that matters most, and the only one you cannot fix
    afterwards.** Looking along the finger instead of across it foreshortens it,
    and a foreshortened finger still produces a perfectly self-consistent set of
    angles -- the fit cannot tell. Pure geometry, no noise, on the proximal
    ratio:

        camera off by   5 deg    ratio biased by  -0.003
                       10 deg                     -0.014
                       15 deg                     -0.031
                       20 deg                     -0.054
                       30 deg                     -0.119
                       60 deg                     -0.428

    against a +/-0.013 error bar from clicking. So aim within about 10 degrees.
    The tool cross-checks clicked phalanx lengths against the model's and warns,
    but only from roughly 30 degrees up -- below that the signal is smaller than
    the clicking noise, so the warning staying quiet is not a clean bill.

    Filming across a desk at whatever angle the hand happens to sit is the
    failure mode: put the camera on the table, level with the hand, looking
    perpendicular to the finger, and keep it still between poses.

2.  Command the knuckle to a handful of angles across its travel -- 0 to 0.95
    rad. Nine poses gives DIP +/-0.030 at 2 deg reading error; three gives
    +/-0.047; seventeen only gets to +/-0.022, so there is little point going
    past nine or so.

3.  Read three angles off each image: the knuckle, then each segment relative to
    the one before it. Angles between *segments*, not to any fixed axis.

    A video is fine, and easier than shooting nine stills -- pull frames out of
    it afterwards:

        ffmpeg -i sweep.mp4 -vf fps=1 frame_%03d.png

4.  Start a measurement file, one row per photo:

        python3 scripts/motion/fit_finger_coupling.py points.csv --from-points --template

    Open each image and click five points along one finger: somewhere on the
    palm behind the knuckle, then the knuckle, both interphalangeal joints, and
    the fingertip. Paste the pixel coordinates into the row. Then:

        python3 scripts/motion/fit_finger_coupling.py points.csv --from-points

Two other input modes exist. `--radians`/degrees angle triples
(`finger,pitch,flexor,dip`) if you measured with a protractor or read the
angles some other way, and `--positions` for fingertip coordinates in the palm
frame (`finger,pitch,x,y,z`, metres) fitted through the model's forward
kinematics -- with the wider error bars above.

WHAT A RATIO CANNOT CAPTURE
---------------------------
If the finger is tendon-driven -- a cable along its length rather than a rigid
four-bar -- then the ratio only holds in free space. Under contact the proximal
joint stops at the object and the distal ones keep closing: the finger conforms,
which is the whole point of an underactuated hand.

A URDF `<mimic>` cannot do that. It enforces the ratio rigidly, so the model is
right for retargeting and free-space reach and wrong for grasp physics -- a
simulated finger will push an object away where the real one would wrap it.
Measure the free-space ratio here, use it for the kinematics, and model the
tendon properly in whatever simulator does the contact (MuJoCo tendons and
equality constraints, PhysX fixed tendons). This is the same wall ALLEX
describes: exact model in MJCF, linear approximation in URDF and USD.

A NOTE ON LINEARITY
-------------------
A single multiplier assumes the coupling is linear. It may not be: WIRobotics'
ALLEX models the same relation on its hand as a quartic and carries a linear
fit in URDF only because URDF cannot express the polynomial. This tool tests for
curvature -- pooled across fingers, against a cubic, by F-test -- and reports the
p-value whether or not it is significant.

That test needs better data than the ratios do. Against a simulated
ALLEX-shaped curve, reading angles to:

    +/-2 deg   p ~ 0.07   cannot tell curvature from noise
    +/-1 deg   p ~ 0.004  resolves it
    +/-0.5 deg p < 1e-6   resolves it decisively

So +/-2 deg is enough to pin the ratios and not enough to know whether a single
ratio is the right model. Read to a degree if that matters to you. More poses
does not substitute -- the noise floor is what limits this, not the sample count.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source/berkeley_humanoid_lite_motion"))

from berkeley_humanoid_lite_motion.dexhand import (  # noqa: E402
    FINGERS, PALM_FRAME, HandModel, tip_frame,
)

GENERATED = REPO / "ros2_ws/src/berkeley_humanoid_lite_description/urdf/berkeley_humanoid_lite_dexhand.urdf"

#: Significance level for the linear-vs-quadratic test. A quadratic always fits
#: noisy data a little better, so the question is whether it fits enough better
#: to be worth believing -- a magnitude threshold flags noise as curvature.
CURVATURE_ALPHA = 0.05


def read_csv(path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    """{finger: (N, len(columns)) float array}, validated and grouped."""
    rows = defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in ["finger", *columns] if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path}: missing column(s) {missing}; "
                             f"expected finger,{','.join(columns)}")
        for line, row in enumerate(reader, start=2):
            finger = row["finger"].strip().capitalize()
            if finger not in FINGERS:
                raise SystemExit(f"{path}:{line}: unknown finger {row['finger']!r}")
            try:
                # Blank means "not measured": track_finger_joints.py leaves the
                # distal angle empty because the fingertip has no bearing to
                # find. NaN here rather than a zero, which would fit as data.
                rows[finger].append([np.nan if not (row[c] or "").strip()
                                     else float(row[c]) for c in columns])
            except (TypeError, ValueError):
                raise SystemExit(f"{path}:{line}: non-numeric value")
    if not rows:
        raise SystemExit(f"{path} has no rows")
    return {finger: np.array(v) for finger, v in rows.items()}


#: Joint centres to click, proximal to distal. `base` is any point on the palm
#: along the finger's own axis, proximal of the knuckle -- it only sets the
#: reference the knuckle angle is measured from.
POINT_NAMES = ("base", "mcp", "pip", "dip", "tip")
POINT_COLUMNS = [f"{name}_{axis}" for name in POINT_NAMES for axis in "uv"]


def angles_from_points(points: np.ndarray) -> np.ndarray:
    """(N, 10) clicked pixel coordinates to (N, 3) joint angles in degrees.

    Each angle is between one segment and the one before it, so the result does
    not depend on how the camera was rolled, only that the finger's flexion
    plane is roughly facing it. Image v axis points down; that flips the sign of
    a cross product but not of an angle between segments, so it is left alone.

    Pixel coordinates rather than a protractor because they are easier to read
    accurately: a couple of pixels' error over a 45 mm segment imaged 400 px
    long is under half a degree, which is where the linearity test starts to
    work.
    """
    p = points.reshape(len(points), len(POINT_NAMES), 2)
    segments = np.diff(p, axis=1)                       # base->mcp, mcp->pip, ...
    lengths = np.linalg.norm(segments, axis=2)
    if np.any(lengths < 1e-6):
        raise SystemExit("two clicked points coincide; every joint needs its own point")
    unit = segments / lengths[..., None]
    # Signed angle in the image plane, so a finger that hyperextends reads
    # negative instead of folding back onto the same positive value.
    cross = unit[:, :-1, 0] * unit[:, 1:, 1] - unit[:, :-1, 1] * unit[:, 1:, 0]
    dot = np.sum(unit[:, :-1] * unit[:, 1:], axis=2)
    signed = np.degrees(np.arctan2(cross, dot))
    # Flexion is one consistent direction; whichever sign the first pose came
    # out as is the one that means "curled".
    reference = signed[np.argmax(np.abs(signed).sum(axis=1))]
    return signed * np.sign(reference.sum() or 1.0)


#: Median phalanx-proportion error above which the view is called foreshortened.
#:
#: Calibrated against simulated tilts. The 95th-percentile floor on a square
#: view is 0.3% when clicking to 2 px on a well-filled frame and 0.8% when
#: clicking to 5 px or imaging the hand small; the signal is 0.5-0.8% at 20
#: degrees, 0.9% at 25 and 1.1-1.3% at 30. So 1% fires from about 30 degrees
#: whatever the conditions, and below 20 the check simply cannot separate tilt
#: from clicking noise.
#:
#: That is a backstop against a badly aimed camera, not a substitute for aiming
#: it: at 30 degrees the proximal ratio is already biased by -0.11 against a
#: +/-0.013 error bar. Aim within 10 degrees, where the bias is under the noise.
FORESHORTENING_TOLERANCE = 0.01


def phalanx_ratios(hand: HandModel, finger: str) -> np.ndarray:
    """Proximal, middle and distal segment lengths, normalised by the proximal.

    Read from the URDF rather than stated, so they follow the model. These are
    the same for all four fingers as it ships, but nothing guarantees that.
    """
    lengths = np.array([
        np.linalg.norm(hand.chain.joints[f"R_{finger}_{joint}"].origin[:3, 3])
        for joint in ("Flexor", "DIP", "tip_fixed")
    ])
    return lengths / lengths[0]


def check_foreshortening(hand: HandModel, points: dict[str, np.ndarray]) -> float:
    """Typical disagreement between clicked and modelled phalanx proportions.

    A finger flexes in one plane, so a camera square to that plane images every
    segment at its true length whatever the pose, and the clicked proportions
    match the model's. Point the camera along the finger instead and the
    segments foreshorten by different amounts -- exactly the error the ratios
    are most sensitive to, and one the fit itself cannot see, because a
    foreshortened finger still gives a perfectly consistent set of angles.

    The median across poses and segments, not the maximum: clicking to a couple
    of pixels puts a 3-4% floor under the worst single reading, which buries the
    signal from anything short of a 45 degree tilt. The median sits near zero on
    a square view and tracks the tilt cleanly.
    """
    deviations = []
    for finger, rows in points.items():
        expected = phalanx_ratios(hand, finger)
        p = rows.reshape(len(rows), len(POINT_NAMES), 2)
        lengths = np.linalg.norm(np.diff(p, axis=1), axis=2)[:, 1:]     # skip base->mcp
        for row in lengths:
            if row[0] < 1e-6:
                continue
            deviations.extend(np.abs(row / row[0] - expected))
    return float(np.median(deviations)) if deviations else 0.0


def write_template(path: Path, mode: str, poses: int = 9) -> None:
    """Write an empty measurement file with the right header and row skeleton."""
    if path.exists():
        raise SystemExit(f"{path} already exists; move it aside or pick another name")
    columns = {"angles": ["pitch", "flexor", "dip"],
               "points": POINT_COLUMNS,
               "positions": ["pitch", "x", "y", "z"]}[mode]
    lines = [",".join(["finger", *columns])]
    for finger in FINGERS:
        for _ in range(poses):
            lines.append(",".join([finger, *[""] * len(columns)]))
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}: {len(FINGERS)} fingers x {poses} poses, columns "
          f"{', '.join(columns)}")
    if mode == "points":
        print()
        print("One row per photo. Open each image, click the five joint centres")
        print("along one finger -- a point on the palm behind the knuckle, then the")
        print("knuckle, both interphalangeal joints, and the fingertip -- and paste")
        print("the pixel coordinates in. Any viewer that shows a cursor readout will")
        print("do; the numbers only have to be consistent within one photo.")


def fit_ratio(driver: np.ndarray, driven: np.ndarray) -> tuple[float, float, float]:
    """Least-squares slope through the origin, its 1-sigma, and the residual RMS.

    Through the origin because a coupling has no offset: at rest the finger is
    straight. An intercept large enough to matter means the zero pose was
    recorded wrong, and `check_offset` says so.
    """
    keep = (np.abs(driver) > 1e-9) & np.isfinite(driver) & np.isfinite(driven)
    driver, driven = driver[keep], driven[keep]
    if len(driver) < 2:
        raise SystemExit("need at least two non-zero poses")
    slope = float(driver @ driven / (driver @ driver))
    residual = driven - slope * driver
    rms = float(np.sqrt(np.mean(residual ** 2)))
    sigma = rms / np.sqrt(driver @ driver) if len(driver) > 1 else np.inf
    return slope, float(sigma), rms


def check_offset(driver: np.ndarray, driven: np.ndarray) -> float:
    """Intercept of an unconstrained line, in the driven joint's units."""
    keep = (np.abs(driver) > 1e-9) & np.isfinite(driver) & np.isfinite(driven)
    if keep.sum() < 2:
        return 0.0
    A = np.stack([driver[keep], np.ones(keep.sum())], axis=1)
    return float(np.linalg.lstsq(A, driven[keep], rcond=None)[0][1])


#: Degree tested against the straight line. Cubic, not quadratic: a linkage
#: coupling curves like an odd function about its neutral pose, and a quadratic
#: fitted to one comes back looking almost straight. ALLEX needed a quartic to
#: model the same relation exactly.
CURVATURE_DEGREE = 3


def check_curvature(driver: np.ndarray, driven: np.ndarray) -> tuple[float, float]:
    """Test a cubic against the linear fit. Returns (p-value, departure).

    An F-test rather than a magnitude comparison. Every extra term fits noisy
    data a little better, so the question is whether it fits enough better to
    believe. Meant to be run on all fingers pooled: nine points at a couple of
    degrees of reading error cannot tell curvature from noise on their own.

    `departure` is how far the curve strays from the line as a fraction of the
    driven joint's range -- what the linear fit costs you, if it is real.
    """
    keep = (np.abs(driver) > 1e-9) & np.isfinite(driver) & np.isfinite(driven)
    x, y = driver[keep], driven[keep]
    n, extra = len(x), CURVATURE_DEGREE - 1
    if n < CURVATURE_DEGREE + 3:
        return 1.0, 0.0
    linear = np.polyfit(x, y, 1)
    curve = np.polyfit(x, y, CURVATURE_DEGREE)
    rss_linear = float(np.sum((y - np.polyval(linear, x)) ** 2))
    rss_curve = float(np.sum((y - np.polyval(curve, x)) ** 2))
    departure = float(np.max(np.abs(np.polyval(curve, x) - np.polyval(linear, x)))
                      / max(np.ptp(y), 1e-9))
    if rss_curve <= 0 or rss_linear <= rss_curve:
        return 1.0, departure
    f_statistic = ((rss_linear - rss_curve) / extra) / (rss_curve / (n - CURVATURE_DEGREE - 1))
    return float(stats.f.sf(f_statistic, extra, n - CURVATURE_DEGREE - 1)), departure


def fit_from_angles(data: dict[str, np.ndarray], to_radians: float) -> dict[str, dict]:
    """Fit both stages per finger from measured joint angles."""
    out = {}
    for finger, rows in data.items():
        pitch, flexor, dip = (rows[:, i] * to_radians for i in range(3))
        f_slope, f_sigma, f_rms = fit_ratio(pitch, flexor)
        if np.all(np.isnan(dip)):
            # The distal stage was not measured. Reported as such rather than
            # fitted to nothing; upstream's own CAD already declares it at 1.0.
            d_slope, d_sigma, d_rms = np.nan, np.nan, 0.0
        else:
            d_slope, d_sigma, d_rms = fit_ratio(flexor, dip)
        out[finger] = {
            "flexor": (f_slope, f_sigma),
            "dip": (d_slope, d_sigma),
            "rms_deg": np.degrees(max(f_rms, d_rms)),
            "samples": len(rows),
        }
    return out


def pooled_diagnostics(data: dict[str, np.ndarray], to_radians: float) -> dict:
    """Offset and linearity checks over every finger at once.

    Pooled because the fingers share one mechanism and one build, so this is the
    same relation measured four times -- and because neither test has the power
    to say anything on nine points.
    """
    stacked = np.concatenate([rows for rows in data.values()]) * to_radians
    pitch, flexor, dip = stacked[:, 0], stacked[:, 1], stacked[:, 2]
    stages = [(pitch, flexor)]
    if not np.all(np.isnan(dip)):
        stages.append((flexor, dip))
    curvature = [check_curvature(a, b) for a, b in stages]
    best = min(curvature, key=lambda c: c[0])
    return {
        "offset_deg": max(np.degrees(abs(check_offset(a, b))) for a, b in stages),
        "curvature_p": best[0],
        "curvature_departure": max(c[1] for c in curvature),
    }


def fit_from_positions(hand: HandModel, data: dict[str, np.ndarray]) -> dict[str, dict]:
    """Fit both stages per finger from fingertip positions in the palm frame.

    The two ratios are fitted jointly: the fingertip only sees them through the
    chain, so fitting the proximal stage against a wrong distal one biases both.
    """
    out = {}
    for finger, rows in data.items():
        pitch, measured = rows[:, 0], rows[:, 1:4]
        if len(pitch) < 3:
            raise SystemExit(f"{finger}: {len(pitch)} samples, need at least 3")
        link = tip_frame(finger)

        def predict(ratios):
            flexor, dip = ratios
            return np.stack([
                hand.chain.position(link, PALM_FRAME, {
                    f"R_{finger}_Pitch": p,
                    f"R_{finger}_Flexor": flexor * p,
                    f"R_{finger}_DIP": dip * flexor * p,
                }) for p in pitch
            ])

        result = least_squares(lambda r: (predict(r) - measured).ravel(), x0=[1.0, 1.0],
                               bounds=([0.0, 0.0], [3.0, 3.0]))
        rms = float(np.sqrt(np.mean(np.sum((predict(result.x) - measured) ** 2, axis=1))))
        # Linearised covariance, scaled by the residual: a real error bar rather
        # than the solver's report, which assumes the model is exact.
        J = result.jac
        try:
            covariance = np.linalg.inv(J.T @ J) * (rms ** 2)
            sigma = np.sqrt(np.diag(covariance))
        except np.linalg.LinAlgError:
            sigma = np.full(2, np.inf)
        out[finger] = {
            "flexor": (float(result.x[0]), float(sigma[0])),
            "dip": (float(result.x[1]), float(sigma[1])),
            "rms_mm": rms * 1000,
            "samples": len(pitch),
        }
    return out


def report(fits: dict[str, dict], mode: str, diagnostics: dict | None = None) -> int:
    quality = "RMS" if mode == "angles" else "RMS"
    unit = "deg" if mode == "angles" else "mm"
    print(f"{'finger':8s} {'Flexor<-Pitch':>18s} {'DIP<-Flexor':>18s} "
          f"{quality + ' [' + unit + ']':>10s}  n")
    for finger in FINGERS:
        if finger not in fits:
            continue
        fit = fits[finger]
        f, fs = fit["flexor"]
        d, ds = fit["dip"]
        rms = fit.get("rms_deg", fit.get("rms_mm", float("nan")))
        dip_text = ("     not measured" if not np.isfinite(d)
                    else f"{d:11.4f} +/-{ds:.4f}")
        print(f"{finger:8s} {f:11.4f} +/-{fs:.4f} {dip_text:>18s} "
              f"{rms:10.2f}  {fit['samples']}")

    flexor = np.array([fits[f]["flexor"][0] for f in fits])
    dip = np.array([fits[f]["dip"][0] for f in fits])
    print()

    if diagnostics and diagnostics.get("foreshortening", 0.0) > FORESHORTENING_TOLERANCE:
        print("!!  The camera is not square to the finger. Clicked phalanx lengths")
        print(f"    disagree with the model's by {diagnostics['foreshortening'] * 100:.0f}%, "
              f"against the {FORESHORTENING_TOLERANCE * 100:.0f}% a square view allows.")
        print("    This biases the ratios low and the fit cannot detect it -- a")
        print("    foreshortened finger still gives perfectly consistent angles.")
        print("    Reshoot with the camera across the finger, not along it, and")
        print("    within about 10 degrees of its flexion plane. The numbers below")
        print("    are not usable until then.")
        print()

    if diagnostics:
        curved = diagnostics["curvature_p"] < CURVATURE_ALPHA
        print(f"Linearity: p = {diagnostics['curvature_p']:.2g} against a cubic, "
              f"which strays {diagnostics['curvature_departure'] * 100:.0f}% of the range.")
        print()
        if curved:
            print(f"!  The coupling is not linear (p = {diagnostics['curvature_p']:.1g}): "
                  f"a quadratic strays {diagnostics['curvature_departure'] * 100:.0f}%")
            print("   of the range from the line. URDF can only carry a multiplier, so")
            print("   that is what it gets, but the fingertip will be off near the ends")
            print("   of the travel. ALLEX hits the same wall and keeps the exact")
            print("   polynomial in MJCF only.")
            print()
        elif diagnostics["offset_deg"] > 2.0:
            # Curvature masquerades as an offset, so only trust this when the
            # linearity test came back clean.
            print(f"!  A straight line through these points misses the origin by "
                  f"{diagnostics['offset_deg']:.1f} deg.")
            print("   The zero pose is probably not where the servo thinks zero is.")
            print("   Re-record it before trusting the ratios.")
            print()

    if len(flexor) > 1:
        spread = f"Flexor {np.ptp(flexor):.4f}"
        if np.all(np.isfinite(dip)):
            spread += f", DIP {np.ptp(dip):.4f}"
        print(f"Spread across fingers: {spread}.")
        print("Wider than the per-finger error bars means the rig moved between")
        print("runs, not that the fingers differ.")
        print()

    print("Paste into the `hand` block of config/arm_attachment.yaml, then")
    print("regenerate and re-run analyze_hand.py:")
    print()
    print("  coupled:")
    print(f"    Flexor: {{driver: Pitch, multiplier: {np.mean(flexor):.4f}}}")
    if np.all(np.isfinite(dip)):
        print(f"    DIP: {{driver: Flexor, multiplier: {np.mean(dip):.4f}}}")
    else:
        print("    DIP: {driver: Flexor, multiplier: 1.0}   # NOT measured -- "
              "upstream's CAD value, left as it was")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("--from-points", action="store_true",
                        help="input is clicked joint-centre pixel coordinates; the "
                             "angles are computed from them")
    parser.add_argument("--positions", action="store_true",
                        help="input is finger,pitch,x,y,z fingertip positions [m] "
                             "in the palm frame, rather than joint angles")
    parser.add_argument("--radians", action="store_true",
                        help="angles are in radians (default: degrees)")
    parser.add_argument("--template", action="store_true",
                        help="write an empty measurement file to fill in, and exit")
    parser.add_argument("--poses", type=int, default=9,
                        help="rows per finger in --template (default: 9)")
    parser.add_argument("--urdf", type=Path, default=GENERATED)
    args = parser.parse_args()

    if args.template:
        mode = "points" if args.from_points else "positions" if args.positions else "angles"
        write_template(args.measurements, mode, args.poses)
        return 0

    if not args.measurements.is_file():
        raise SystemExit(
            f"{args.measurements} does not exist. Write it yourself, or start from "
            f"a skeleton:\n"
            f"  python3 {Path(__file__).name} {args.measurements} --from-points --template"
        )

    if args.from_points:
        points = read_csv(args.measurements, POINT_COLUMNS)
        skew = check_foreshortening(HandModel.from_urdf(args.urdf), points)
        data = {f: angles_from_points(v) for f, v in points.items()}
        diagnostics = pooled_diagnostics(data, np.pi / 180.0)
        diagnostics["foreshortening"] = skew
        return report(fit_from_angles(data, np.pi / 180.0), "angles", diagnostics)

    if args.positions:
        hand = HandModel.from_urdf(args.urdf)
        data = read_csv(args.measurements, ["pitch", "x", "y", "z"])
        return report(fit_from_positions(hand, data), "positions")

    data = read_csv(args.measurements, ["pitch", "flexor", "dip"])
    to_radians = 1.0 if args.radians else np.pi / 180.0
    return report(fit_from_angles(data, to_radians), "angles",
                  pooled_diagnostics(data, to_radians))


if __name__ == "__main__":
    raise SystemExit(main())
