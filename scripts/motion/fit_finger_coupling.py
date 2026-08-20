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
them off a fingertip position dilutes the distal stage through a 18 mm lever and
loses most of the precision. Per-parameter 1-sigma from a 9-pose sweep:

    fingertip markers, 0.5 mm      Flexor +/-0.015    DIP +/-0.068
    fingertip markers, 2.0 mm      Flexor +/-0.061    DIP +/-0.273
    angles read to +/-2 deg        Flexor +/-0.021    DIP +/-0.030
    angles read to +/-1 deg        Flexor +/-0.010    DIP +/-0.015

A photograph read to a couple of degrees beats half-millimetre triangulation on
the parameter that is hard to see, and needs no markers and no calibration.

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

2.  Command the knuckle to a handful of angles across its travel -- 0 to 0.95
    rad. Nine poses gives DIP +/-0.030 at 2 deg reading error; three gives
    +/-0.047; seventeen only gets to +/-0.022, so there is little point going
    past nine or so.

3.  Read three angles off each image: the knuckle, then each segment relative to
    the one before it. Angles between *segments*, not to any fixed axis.

4.  Write them down and run this:

        finger,pitch,flexor,dip
        Middle,0,0,0
        Middle,13.6,11.2,7.5
        ...

        python3 scripts/motion/fit_finger_coupling.py angles.csv

    Degrees by default; pass `--radians` if that is what you recorded.

If you would rather use the capture rig, `--positions` takes fingertip
coordinates in the palm frame instead (`finger,pitch,x,y,z`, metres) and fits
through the model's forward kinematics. Expect the wider error bars above.

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
                rows[finger].append([float(row[c]) for c in columns])
            except (TypeError, ValueError):
                raise SystemExit(f"{path}:{line}: non-numeric value")
    if not rows:
        raise SystemExit(f"{path} has no rows")
    return {finger: np.array(v) for finger, v in rows.items()}


def fit_ratio(driver: np.ndarray, driven: np.ndarray) -> tuple[float, float, float]:
    """Least-squares slope through the origin, its 1-sigma, and the residual RMS.

    Through the origin because a coupling has no offset: at rest the finger is
    straight. An intercept large enough to matter means the zero pose was
    recorded wrong, and `check_offset` says so.
    """
    keep = np.abs(driver) > 1e-9
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
    keep = np.abs(driver) > 1e-9
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
    keep = np.abs(driver) > 1e-9
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
    stages = ((pitch, flexor), (flexor, dip))
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
        print(f"{finger:8s} {f:11.4f} +/-{fs:.4f} {d:11.4f} +/-{ds:.4f} "
              f"{rms:10.2f}  {fit['samples']}")

    flexor = [fits[f]["flexor"][0] for f in fits]
    dip = [fits[f]["dip"][0] for f in fits]
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
        print(f"Spread across fingers: Flexor {np.ptp(flexor):.4f}, DIP {np.ptp(dip):.4f}.")
        print("Wider than the per-finger error bars means the rig moved between")
        print("runs, not that the fingers differ.")
        print()

    print("Paste into the `hand` block of config/arm_attachment.yaml, then")
    print("regenerate and re-run analyze_hand.py:")
    print()
    print("  coupled:")
    print(f"    Flexor: {{driver: Pitch, multiplier: {np.mean(flexor):.4f}}}")
    print(f"    DIP: {{driver: Flexor, multiplier: {np.mean(dip):.4f}}}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("--positions", action="store_true",
                        help="input is finger,pitch,x,y,z fingertip positions [m] "
                             "in the palm frame, rather than joint angles")
    parser.add_argument("--radians", action="store_true",
                        help="angles are in radians (default: degrees)")
    parser.add_argument("--urdf", type=Path, default=GENERATED)
    args = parser.parse_args()

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
