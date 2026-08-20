#!/usr/bin/env python3
"""Generate the SRDF for a merged humanoid + DexHand model.

Three things are stitched together:

  * the arm groups, written here, plus a <side>_shoulder joint group per arm
    for hardware bring-up (shoulder motors only, see shoulder_group);
  * the hand group and its grip presets, lifted from the DexHand's own SRDF so
    the poses their team tuned (open, fist, cylinder_grip, ...) carry over
    unchanged -- once per hand, renamed per side;
  * the collision matrix, produced by running generate_collision_matrix.py.

    python3 generate_srdf.py --model v1arm [--samples 20000]

Models:
    dexhand  right arm only, hand welded on, 5-DOF arm
    v1arm    both arms, V1 forearm + 3-DOF wrist + axial rotation, 9-DOF arm
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESCRIPTION = HERE.parent / "berkeley_humanoid_lite_description"
# The DexHand's own SRDF: the source of the 12 grip presets and of the 194
# intra-hand collision pairs its Setup Assistant computed against real mesh
# geometry. Vendored, because it used to be read from one developer's laptop and
# a regeneration anywhere else simply failed.
DEXHAND_SRDF = Path(__file__).resolve().parents[2] / (
    "vendor/dexhandv2_description/config/dexhandv2_right_8servo.srdf"
)

MODELS = {
    "stock": {
        "urdf": "berkeley_humanoid_lite.urdf",
        "srdf": "berkeley_humanoid_lite.srdf",
        "sides": [],
        "note": "Stock end effectors on both arms; 5-DOF arm, no DexHand.",
    },
    "dexhand": {
        "urdf": "berkeley_humanoid_lite_dexhand.urdf",
        "srdf": "berkeley_humanoid_lite_dexhand.srdf",
        "sides": ["right"],
        "note": "DexHand v2 welded to the right forearm; 5-DOF arm.",
    },
    "v1arm": {
        "urdf": "berkeley_humanoid_lite_v1arm.urdf",
        "srdf": "berkeley_humanoid_lite_v1arm.srdf",
        "sides": ["right", "left"],
        "note": "V1 forearm + 3-DOF V1 wrist on both arms; 8-DOF arm.",
    },
}

TIP = {"right": "base_link", "left": "L_base_link"}
STOCK_TIP = {"right": "arm_right_hand_link", "left": "arm_left_hand_link"}

ARM_STATES = {
    "home": {"shoulder_pitch": 0.0, "shoulder_roll": 0.0, "shoulder_yaw": 0.0,
             "elbow_pitch": 0.0, "elbow_roll": 0.0},
    # Elbow bent and the arm clear of the torso: well-conditioned for IK, away
    # from the shoulder singularity at full extension.
    "ready": {"shoulder_pitch": -0.3, "shoulder_roll": 0.3, "shoulder_yaw": 0.0,
              "elbow_pitch": 0.9, "elbow_roll": 0.0},
}
# The elbow pitch limit is one-sided and mirrored ([0, pi/2] left,
# [-pi/2, 0] right), so these flip sign on the right.
MIRRORED_JOINTS = {"shoulder_pitch", "shoulder_roll", "elbow_pitch"}
WRIST_JOINTS = ["wrist_pitch_lower", "wrist_yaw", "wrist_pitch_upper"]


def side_name(name: str, side: str) -> str:
    if side == "right":
        return name
    return "L_" + name[2:] if name.startswith("R_") else "L_" + name


def arm_group(side: str, tip: str) -> str:
    return (f'    <group name="{side}_arm">\n'
            f'        <chain base_link="base" tip_link="{tip}"/>\n'
            f'    </group>')


SHOULDER_JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw"]


def shoulder_group(side: str) -> list[str]:
    """Joint-space group of the three shoulder motors only.

    Exists for hardware bring-up: hardware_joints.yaml enables the shoulder
    motors first (they are the calibrated, bench-tested ones), and the bridge
    refuses any trajectory that names a disabled joint. Planning in <side>_arm
    always names all eight, so this is the group to use until elbow and wrist
    are verified. No IK, no marker -- joint targets only.
    """
    lines = [f'    <group name="{side}_shoulder">']
    for joint in SHOULDER_JOINTS:
        lines.append(f'        <joint name="arm_{side}_{joint}_joint"/>')
    lines.append("    </group>")
    return lines


def shoulder_state(side: str, name: str, values: dict) -> list[str]:
    lines = [f'    <group_state name="{name}" group="{side}_shoulder">']
    for joint in SHOULDER_JOINTS:
        value = values[joint]
        if side == "right" and joint in MIRRORED_JOINTS:
            value = -value
        lines.append(f'        <joint name="arm_{side}_{joint}_joint" value="{value}"/>')
    lines.append("    </group_state>")
    return lines


def arm_state(side: str, name: str, values: dict, wrist: bool) -> list[str]:
    lines = [f'    <group_state name="{name}" group="{side}_arm">']
    for joint, value in values.items():
        if side == "right" and joint in MIRRORED_JOINTS:
            value = -value
        lines.append(
            f'        <joint name="arm_{side}_{joint}_joint" value="{value}"/>'
        )
    if wrist:
        for joint in WRIST_JOINTS:
            lines.append(f'        <joint name="arm_{side}_{joint}_joint" value="0"/>')
    lines.append("    </group_state>")
    return lines


def import_hand(path: Path, sides: list[str]) -> tuple[str, str]:
    """Pull the `hand` group and its grip presets out of the DexHand SRDF."""
    if not path.is_file():
        raise SystemExit(f"DexHand SRDF not found: {path}")
    root = ET.parse(path).getroot()

    source = root.find("group[@name='hand']")
    if source is None:
        raise SystemExit(f"{path}: no group named 'hand'")
    joints = [j.get("name") for j in source.findall("joint")]
    if not joints:
        raise SystemExit(f"{path}: group 'hand' lists no joints")
    presets = root.findall("group_state[@group='hand']")

    groups, states = [], []
    for side in sides:
        groups.append(f'    <group name="{side}_hand">')
        groups += [f'        <joint name="{side_name(n, side)}"/>' for n in joints]
        groups.append("    </group>")
        for state in presets:
            states.append(
                f'    <group_state name="{state.get("name")}" group="{side}_hand">'
            )
            for joint in state.findall("joint"):
                states.append(
                    f'        <joint name="{side_name(joint.get("name"), side)}" '
                    f'value="{joint.get("value")}"/>'
                )
            states.append("    </group_state>")

    print(f"imported hand group ({len(joints)} joints) and {len(presets)} presets "
          f"for {len(sides)} hand(s)", file=sys.stderr)
    return "\n".join(groups), "\n".join(states)


def collision_matrix(urdf: Path, samples: int, mirror: bool,
                     import_srdf: Path | None = DEXHAND_SRDF) -> str:
    command = [
        sys.executable, str(HERE / "generate_collision_matrix.py"),
        "--urdf", str(urdf),
        "--samples", str(samples),
    ]
    if import_srdf is not None:
        command += ["--import-srdf", str(import_srdf)]
    if mirror:
        command.append("--mirror-import")
    print(f"running generate_collision_matrix.py with {samples} samples...",
          file=sys.stderr)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"collision matrix failed:\n{result.stdout}\n{result.stderr}")
    lines = [l for l in result.stdout.splitlines() if "disable_collisions" in l]
    if not lines:
        raise SystemExit(f"no pairs produced:\n{result.stdout}")
    for line in result.stdout.splitlines():
        if line.startswith("<!-- disabled"):
            print(line, file=sys.stderr)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), default="v1arm")
    parser.add_argument("--samples", type=int, default=20000)
    args = parser.parse_args()

    spec = MODELS[args.model]
    sides = spec["sides"]
    wrist = args.model == "v1arm"
    urdf = DESCRIPTION / "urdf" / spec["urdf"]
    if not urdf.is_file():
        raise SystemExit(f"{urdf} missing; run generate_urdf.py first")

    if sides:
        hand_groups, hand_states = import_hand(DEXHAND_SRDF, sides)
        import_srdf = DEXHAND_SRDF
    else:
        # The stock model has no DexHand, so there is no hand matrix to import.
        hand_groups = hand_states = ""
        import_srdf = None
    pairs = collision_matrix(urdf, args.samples, mirror=len(sides) > 1,
                             import_srdf=import_srdf)

    body = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<!-- GENERATED by generate_srdf.py. Do not edit by hand. -->",
        "<!--",
        f"  {spec['note']}",
        "",
        "  Each arm chain ends at its DexHand mount link, which is therefore the",
        "  IK target: drag the marker and the palm goes there. Fingers are a",
        "  separate group per hand so arm planning and grasping stay independent.",
        "",
        "  Right-hand DexHand names are unchanged from upstream so the existing",
        "  dexhand_bringup driver keeps matching them; the left hand is the",
        "  mirrored copy with L_ names.",
        "-->",
        '<robot name="berkeley-humanoid-lite">',
        "",
    ]
    for side in sides:
        body.append(arm_group(side, TIP[side]))
    # The left arm keeps its stock end effector in the right-arm-only model.
    for side in ("right", "left"):
        if side not in sides:
            body.append(arm_group(side, STOCK_TIP[side]))
    body += [
        '    <group name="both_arms">',
        '        <group name="left_arm"/>',
        '        <group name="right_arm"/>',
        "    </group>",
        "",
    ]
    for side in ("left", "right"):
        body += shoulder_group(side)
    body.append("")
    if hand_groups:
        body += [hand_groups, ""]

    for side in ("right", "left"):
        for name, values in ARM_STATES.items():
            body += arm_state(side, name, values, wrist and side in sides)
    for side in ("left", "right"):
        for name, values in ARM_STATES.items():
            body += shoulder_state(side, name, values)
    if hand_states:
        body += ["", hand_states, ""]

    for side in sides:
        body.append(
            f'    <end_effector name="{side}_hand_ee" parent_link="{TIP[side]}"\n'
            f'                  group="{side}_hand" parent_group="{side}_arm"/>'
        )
    body += [
        "",
        "    <!-- Generated by generate_collision_matrix.py. -->",
        pairs,
        "</robot>",
        "",
    ]

    output = HERE / "config" / spec["srdf"]
    output.write_text("\n".join(body))
    print(f"wrote {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
