#!/usr/bin/env python3
"""Fit the DexHand's finger coupling ratios from measured fingertips.

The ratios in `config/arm_attachment.yaml` decide how far a finger curls for a
given servo command, and they ship as estimates. This fits them to the hardware.

Measure first. With the hand held still and one finger at a time, command the
knuckle across its travel and record where the fingertip ends up in the palm
frame -- a tracked marker on the pad, or a probe, or two calibrated cameras.
Write one CSV per run:

    finger,pitch,x,y,z
    Index,0.00,0.0059,0.0303,0.1818
    Index,0.24,0.0424,0.0271,0.1741
    ...

Then:

    python3 scripts/motion/fit_finger_coupling.py measurements.csv

It fits `Flexor <- Pitch` and `DIP <- Flexor` by least squares against the
model's own forward kinematics and prints the YAML block to paste back. Positions
are in metres, angles in radians, and the palm frame is the hand's `base_link`
-- the same frame `analyze_hand.py` reports in.

ALLEX publishes 0.656 for `DIP <- PIP` on a comparable anthropomorphic finger,
so a fit landing near 1.0 for both stages is worth a second look at the setup
before it is trusted.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source/berkeley_humanoid_lite_motion"))

from berkeley_humanoid_lite_motion.dexhand import (  # noqa: E402
    FINGERS, PALM_FRAME, HandModel, tip_frame,
)

GENERATED = REPO / "ros2_ws/src/berkeley_humanoid_lite_description/urdf/berkeley_humanoid_lite_dexhand.urdf"


def read_measurements(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{finger: (pitch (N,), position (N, 3))} from the CSV described above."""
    rows = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            finger = row["finger"].strip().capitalize()
            if finger not in FINGERS:
                raise SystemExit(f"unknown finger {row['finger']!r}; expected one of {FINGERS}")
            rows[finger].append((
                float(row["pitch"]),
                [float(row["x"]), float(row["y"]), float(row["z"])],
            ))
    if not rows:
        raise SystemExit(f"{path} has no rows")
    return {
        finger: (np.array([r[0] for r in v]), np.array([r[1] for r in v]))
        for finger, v in rows.items()
    }


def fit(hand: HandModel, finger: str, pitch: np.ndarray,
        measured: np.ndarray) -> tuple[float, float, float]:
    """Least-squares (flexor_ratio, dip_ratio) and the RMS residual [m].

    The two ratios are fitted jointly rather than one at a time: the fingertip
    only sees their product through the chain, so fitting the proximal stage
    against a wrong distal one biases both.
    """
    chain = hand.chain
    link = tip_frame(finger)

    def predict(ratios):
        flexor, dip = ratios
        out = []
        for p in pitch:
            out.append(chain.position(link, PALM_FRAME, {
                f"R_{finger}_Pitch": p,
                f"R_{finger}_Flexor": flexor * p,
                f"R_{finger}_DIP": dip * flexor * p,
            }))
        return np.stack(out)

    result = least_squares(
        lambda r: (predict(r) - measured).ravel(), x0=[1.0, 1.0],
        bounds=([0.0, 0.0], [3.0, 3.0]),
    )
    residual = float(np.sqrt(np.mean(np.sum((predict(result.x) - measured) ** 2, axis=1))))
    return float(result.x[0]), float(result.x[1]), residual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("--urdf", type=Path, default=GENERATED)
    args = parser.parse_args()

    hand = HandModel.from_urdf(args.urdf)
    data = read_measurements(args.measurements)

    flexor_ratios, dip_ratios = [], []
    print(f"{'finger':8s} {'Flexor<-Pitch':>14s} {'DIP<-Flexor':>13s} {'RMS':>9s}  samples")
    for finger in FINGERS:
        if finger not in data:
            continue
        pitch, measured = data[finger]
        if len(pitch) < 3:
            print(f"{finger:8s} skipped -- {len(pitch)} samples, need at least 3")
            continue
        flexor, dip, rms = fit(hand, finger, pitch, measured)
        flexor_ratios.append(flexor)
        dip_ratios.append(dip)
        print(f"{finger:8s} {flexor:14.4f} {dip:13.4f} {rms * 1000:7.2f} mm  {len(pitch)}")

    if not flexor_ratios:
        raise SystemExit("nothing fitted")

    print()
    print("Per-finger spread is the honest error bar; a wide one means the setup")
    print("moved between runs, not that the fingers differ. Shared ratios:")
    print()
    print("  coupled:")
    print(f"    Flexor: {{driver: Pitch, multiplier: {np.mean(flexor_ratios):.4f}}}")
    print(f"    DIP: {{driver: Flexor, multiplier: {np.mean(dip_ratios):.4f}}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
