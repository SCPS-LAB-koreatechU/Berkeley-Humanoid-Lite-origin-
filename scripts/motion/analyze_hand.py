#!/usr/bin/env python3
"""Report what the DexHand's eight servos can reach.

Run this before designing anything that depends on the hand -- a grasp
controller, a retargeting objective, a reward term. The numbers come out of the
same URDF MoveIt plans against, so they move when the hardware model does.

    python3 scripts/motion/analyze_hand.py

The thumb study needs the vendored upstream description as well, because the
generated model has already frozen the thumb and discarded its axes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "source/berkeley_humanoid_lite_motion"))

from berkeley_humanoid_lite_motion.dexhand import (  # noqa: E402
    FINGERS, THUMB_JOINTS, HandModel, best_thumb_posture, opposition_gap, tip_frame,
)
from berkeley_humanoid_lite_motion.urdf import Chain  # noqa: E402

GENERATED = REPO / "ros2_ws/src/berkeley_humanoid_lite_description/urdf/berkeley_humanoid_lite_dexhand.urdf"
UPSTREAM = REPO / "ros2_ws/vendor/dexhandv2_description/urdf/dexhandv2_right.urdf"


def palm_plane(hand: HandModel) -> tuple[np.ndarray, np.ndarray]:
    """Centre and outward normal of the plane through the four knuckles."""
    chain = hand.chain
    knuckles = np.stack([
        chain.position(chain.joints[f"R_{f}_Yaw"].child, hand.palm) for f in FINGERS
    ])
    centre = knuckles.mean(axis=0)
    normal = np.linalg.svd(knuckles - centre)[2][2]
    if np.dot(normal, hand.thumb_tip() - centre) < 0:
        normal = -normal
    return centre, normal


#: Below this the fingertip and the thumb can meet, so a pinch closes.
PINCH_GAP_M = 0.005


def _verdict(at_zero: float, best: float) -> str:
    if at_zero <= PINCH_GAP_M:
        return "Configured posture already opposes: pinch grasps are reachable."
    if best <= PINCH_GAP_M:
        return ("Pinch is reachable, but not at the configured posture -- "
                f"{at_zero * 1000:.0f} mm short. Refreeze the thumb at the angles above.")
    return ("No posture opposes: the nearest fingertip stays "
            f"{best * 1000:.0f} mm away. Grasps have to cage against the palm.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urdf", type=Path, default=GENERATED)
    parser.add_argument("--upstream", type=Path, default=UPSTREAM)
    parser.add_argument("--restarts", type=int, default=8,
                        help="restarts for the frozen-thumb posture search")
    args = parser.parse_args()

    hand = HandModel.from_urdf(args.urdf)
    report = hand.analyse()

    print(f"DexHand v2 as generated from {args.urdf.name}")
    print(f"  actuated : {report['n_dof']} -- {', '.join(report['joint_names'])}")
    print(f"  coupled  : {len(report['coupled_joints'])}")
    for line in report["coupled_joints"]:
        print(f"             {line}")
    print(f"  frozen   : {len(report['frozen_joints'])} -- {', '.join(report['frozen_joints'])}")
    print()

    print("Fingertip reach in the palm frame")
    for finger in FINGERS:
        d = report["per_finger"][finger]
        print(f"  {finger:7s} travel {d['max_travel_m'] * 1000:6.1f} mm"
              f"   bbox {np.round(d['bbox_extent_m'] * 1000, 1)} mm")
    span = np.array(report["index_pinky_span_m"]) * 1000
    print(f"  index-to-pinky fingertip span: {span[0]:.1f} .. {span[1]:.1f} mm")
    print()

    centre, normal = palm_plane(hand)
    print("Clearance above the knuckle plane, open -> fully flexed")
    for finger in FINGERS:
        pitch = hand.chain.joints[f"R_{finger}_Pitch"]
        heights = [
            np.dot(hand.chain.position(tip_frame(finger), hand.palm, {f"R_{finger}_Pitch": p}) - centre, normal)
            for p in (pitch.lower, pitch.upper)
        ]
        print(f"  {finger:7s} {heights[0] * 1000:5.1f} -> {heights[1] * 1000:5.1f} mm")
    print("  An object is caged into this sweep. Whether the hand can also pinch")
    print("  depends on the thumb -- see the opposition figures below.")
    print()

    if args.upstream.is_file():
        upstream = Chain.from_urdf(args.upstream, repair_mimics=True)
        at_zero = opposition_gap(hand, upstream, np.zeros(len(THUMB_JOINTS)))
        angles, best = best_thumb_posture(hand, upstream, restarts=args.restarts)
        print("Opposition -- closest any fingertip comes to the frozen thumb tip")
        print(f"  at the configured posture: {at_zero * 1000:6.1f} mm")
        print(f"  best the mechanism allows: {best * 1000:6.1f} mm")
        print("    " + ", ".join(f"{n}={v:+.3f}" for n, v in zip(THUMB_JOINTS, angles))
              + "  (R_Thumb_DIP mimics R_Thumb_Flexor)")
        print()
        print("  No servo reaches the thumb, so its posture is decided when the hand")
        print("  is assembled. Set it in the `hand.thumb` block of")
        print("  config/arm_attachment.yaml and regenerate; the generator bakes the")
        print("  angle into the joint origin.")
        print()
        print(f"  {_verdict(at_zero, best)}")
    else:
        print(f"(skipped the thumb study: {args.upstream} not found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
