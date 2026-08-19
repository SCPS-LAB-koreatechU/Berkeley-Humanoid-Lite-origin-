#!/usr/bin/env python3
"""Build the ROS-usable URDFs from the upstream asset URDF and the DexHand.

Four models are written to urdf/:

  berkeley_humanoid_lite.urdf          arms only, stock end effectors
  berkeley_humanoid_lite_dexhand.urdf  DexHand v2 welded to the right forearm
  berkeley_humanoid_lite_v1arm.urdf    both arms: V1 forearm + 3-DOF V1 wrist
                                       + DexHand v2 fingers (left hand mirrored)
  berkeley_humanoid_lite_tuning.urdf   v1arm with the right mount as 6 sliders

The upstream humanoid file is emitted by onshape-to-robot and is not directly
usable: mesh paths are written `package://../meshes/foo.stl`, which the ROS
resource retriever cannot resolve, and the root link `base` is free-floating
where MoveIt needs a fixed root.  Both are fixed here.

The V1 forearm and wrist come from the vendored DexHand V1 description with
their real joint origins, axes and travel limits; the one transform still
unmeasured is where the forearm bolts onto elbow_roll.  See
config/arm_attachment.yaml.

DexHand names are left untouched on the right arm so the existing
dexhand_bringup driver keeps matching them.  The left arm is a mirrored copy
with `R_` swapped for `L_` (and `L_` prefixed where there was no side marker),
since two hands cannot share link names.

    python3 generate_urdf.py
"""

from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

PACKAGE = "berkeley_humanoid_lite_description"

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]

UPSTREAM = (
    WORKSPACE
    / "source/berkeley_humanoid_lite_assets/data/robots/berkeley_humanoid"
    / "berkeley_humanoid_lite/urdf/berkeley_humanoid_lite.urdf"
)
# Vendored, not committed: see fetch_vendor.sh and the licence note there.
VENDOR = HERE.parent.parent / "vendor"
DEXHAND = VENDOR / "dexhandv2_description/urdf/dexhandv2_right.urdf"
# The 8-servo variant lives in the user's dexhand MoveIt config rather than
# upstream; fall back to it when present so the actuated model stays the default.
DEXHAND_8SERVO = Path(
    "/home/scps-ubuntu1/Desktop/dexhand_moveit_ws/src/dexhand_moveit_config"
    "/config/dexhandv2_right_8servo.urdf"
)
if DEXHAND_8SERVO.is_file():
    DEXHAND = DEXHAND_8SERVO
ATTACHMENT_CONFIG = HERE / "config" / "arm_attachment.yaml"
# Vendored DexHand V1 description: the real source of the wrist kinematics.
DEXHAND_V1 = {
    "right": HERE.parent.parent / "vendor/dexhand_v1_description/urdf/dexhand-right.urdf",
    "left": HERE.parent.parent / "vendor/dexhand_v1_description/urdf/dexhand-left.urdf",
}
# Upstream V1 inertials are Fusion defaults at steel density.
V1_SOURCE_DENSITY = 7850.0  # [kg/m3]
# V1 link names per side: forearm, then the two wrist halves.
V1_LINKS = {
    "right": ["forearm_1", "wrist_lower_1", "wrist_upper_1"],
    "left": ["forearm_left_1", "wrist_lower_1", "wrist_upper_1"],
}
OUTPUT_DIR = HERE / "urdf"

# Every leg link, and by extension every joint that drives one.
LEG_PREFIX = "leg_"

STOCK_HAND = {
    "right": ("arm_hand_r", "arm_right_hand_link"),
    "left": ("arm_left_hand_l", "arm_left_hand_link"),
}
ELBOW = {"right": "arm_right_elbow_roll", "left": "arm_left_elbow_roll"}
DEXHAND_ROOT = "base_link"

TUNING_TRANSLATION = 0.25  # [m]
TUNING_ROTATION = 3.14159265  # [rad]


def indent(element: ET.Element, level: int = 0) -> None:
    """Pretty-print in place; ElementTree.indent is 3.9+, so keep it explicit."""
    pad = "\n" + "  " * level
    if len(element):
        if not (element.text or "").strip():
            element.text = pad + "  "
        for child in element:
            indent(child, level + 1)
        if not (child.tail or "").strip():
            child.tail = pad
    if level and not (element.tail or "").strip():
        element.tail = pad


def as_text(values) -> str:
    # 12 significant digits so serialisation contributes nothing measurable.
    # Note the model is still not perfectly left/right symmetric: the upstream
    # humanoid's own arms differ by ~4e-7 m at elbow_roll before anything is
    # attached, and that orientation error grows to ~1.6e-6 m by the fingertips.
    # It is inherited from the source CAD, not introduced by the mirror.
    return " ".join(f"{float(v):.12g}" for v in values)


def side_name(name: str, side: str) -> str:
    """Rename a DexHand link/joint for one side.

    Right keeps upstream names verbatim so the existing hardware driver still
    matches.  Left swaps a leading `R_` for `L_`, or prefixes `L_` when the name
    carries no side marker (`base_link`, `Index_Knuckle_1`, ...).
    """
    if side == "right":
        return name
    return "L_" + name[2:] if name.startswith("R_") else "L_" + name


def mirror_vector(xyz) -> list[float]:
    """Mirror a position across the xz plane."""
    x, y, z = (float(v) for v in xyz)
    return [x, -y, z]


def mirror_axis(xyz) -> list[float]:
    """Mirror a rotation axis across the xz plane.

    A rotation axis is a pseudovector: under a reflection M it maps to
    det(M) * M * a.  With M = diag(1, -1, 1) that is (-ax, ay, -az).  Getting
    this right is what makes a positive joint angle curl a left finger the same
    way it curls the right one, so the imported grip presets carry over.
    """
    x, y, z = (float(v) for v in xyz)
    return [-x, y, -z]


def mirror_rpy(rpy) -> list[float]:
    """Mirror an orientation across the xz plane: roll and yaw flip sign."""
    roll, pitch, yaw = (float(v) for v in rpy)
    return [-roll, pitch, -yaw]


def load_config() -> dict:
    config = yaml.safe_load(ATTACHMENT_CONFIG.read_text())
    mount = config["mount"]
    if mount.get("left") == "mirror":
        right = mount["right"]
        mount["left"] = {
            "xyz": mirror_vector(right["xyz"]),
            "rpy": mirror_rpy(right["rpy"]),
        }
    return config


def load_humanoid(config: dict | None = None) -> ET.Element:
    """Upstream humanoid with mesh paths fixed and a fixed `world` root."""
    text = UPSTREAM.read_text()
    fixed = text.replace(
        'filename="package://../meshes/',
        f'filename="package://{PACKAGE}/meshes/',
    )
    if fixed == text:
        raise SystemExit("no mesh paths rewritten; upstream format changed?")

    root = ET.fromstring(fixed)
    if root.find("link[@name='world']") is not None:
        raise SystemExit("upstream already defines a world link; review by hand")

    world = ET.Element("link", {"name": "world"})
    weld = ET.Element("joint", {"name": "world_to_base", "type": "fixed"})
    ET.SubElement(weld, "parent", {"link": "world"})
    ET.SubElement(weld, "child", {"link": "base"})
    ET.SubElement(weld, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    root.insert(0, weld)
    root.insert(0, world)

    if config is not None and not config.get("include_legs", True):
        strip_legs(root)
    return root


def load_dexhand(side: str) -> ET.Element:
    """DexHand for one side: meshes repointed, and mirrored/renamed if left."""
    if not DEXHAND.is_file():
        raise SystemExit(
            f"DexHand URDF not found: {DEXHAND}\n"
            "The dexhand workspace has moved; update DEXHAND in this script."
        )
    mesh_dir = "meshes_dexhand/right" if side == "right" else "meshes_dexhand_left"
    text = DEXHAND.read_text().replace(
        'filename="package://dexhandv2_description/meshes/right/',
        f'filename="package://{PACKAGE}/{mesh_dir}/',
    )
    if "dexhandv2_description" in text:
        raise SystemExit("some DexHand mesh paths were not rewritten")
    root = ET.fromstring(text)

    if side == "left":
        left_meshes = HERE / "meshes_dexhand_left"
        if not left_meshes.is_dir():
            raise SystemExit(
                f"{left_meshes} missing; run mirror_meshes.py first"
            )

    for element in root:
        if element.get("name"):
            element.set("name", side_name(element.get("name"), side))
        if element.tag == "joint":
            for tag in ("parent", "child"):
                node = element.find(tag)
                node.set("link", side_name(node.get("link"), side))
            if side == "left":
                origin = element.find("origin")
                if origin is not None:
                    origin.set("xyz", as_text(mirror_vector(
                        np.fromstring(origin.get("xyz", "0 0 0"), sep=" "))))
                    origin.set("rpy", as_text(mirror_rpy(
                        np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))))
                axis = element.find("axis")
                if axis is not None:
                    axis.set("xyz", as_text(mirror_axis(
                        np.fromstring(axis.get("xyz"), sep=" "))))
        if element.tag == "link" and side == "left":
            # Visual/collision/inertial origins live in the link frame, which is
            # itself mirrored, so only their own offsets need flipping.
            for child in element:
                origin = child.find("origin") if len(child) else None
                if origin is not None:
                    origin.set("xyz", as_text(mirror_vector(
                        np.fromstring(origin.get("xyz", "0 0 0"), sep=" "))))
                    origin.set("rpy", as_text(mirror_rpy(
                        np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))))
    return root


def strip_legs(robot: ET.Element) -> None:
    """Drop the leg subtrees.

    Removing a link means removing the joint that has it as a child, or the URDF
    stops parsing. Nothing else in this workspace references the legs: they are
    in no planning group and no hardware map.
    """
    legs = {l.get("name") for l in robot.findall("link")
            if l.get("name").startswith(LEG_PREFIX)}
    if not legs:
        return
    removed_joints = 0
    for joint in list(robot.findall("joint")):
        if (joint.find("child").get("link") in legs
                or joint.find("parent").get("link") in legs):
            robot.remove(joint)
            removed_joints += 1
    for link in list(robot.findall("link")):
        if link.get("name") in legs:
            robot.remove(link)
    print(f"  stripped {len(legs)} leg links and {removed_joints} joints")


def strip_stock_hand(robot: ET.Element, side: str) -> None:
    joint_name, link_name = STOCK_HAND[side]
    joint = robot.find(f"joint[@name='{joint_name}']")
    link = robot.find(f"link[@name='{link_name}']")
    if joint is None or link is None:
        raise SystemExit(f"expected {joint_name}/{link_name} in the upstream URDF")
    robot.remove(joint)
    robot.remove(link)


def graft(robot: ET.Element, source: ET.Element) -> None:
    """Copy links, joints and materials from one robot element into another."""
    existing = {
        tag: {e.get("name") for e in robot.findall(tag)}
        for tag in ("link", "joint", "material")
    }
    for element in source:
        name = element.get("name")
        if element.tag in ("link", "joint"):
            if name in existing[element.tag]:
                raise SystemExit(f"{element.tag} name collision: {name}")
            existing[element.tag].add(name)
        elif element.tag == "material":
            if name in existing["material"]:
                continue
            existing["material"].add(name)
        robot.append(copy.deepcopy(element))


def add_joint(robot, name, kind, parent, child, origin, axis=None, limits=None,
              effort="5", velocity="3"):
    joint = ET.SubElement(robot, "joint", {"name": name, "type": kind})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    ET.SubElement(joint, "origin", {
        "xyz": as_text(origin["xyz"]), "rpy": as_text(origin["rpy"])
    })
    if axis is not None:
        ET.SubElement(joint, "axis", {"xyz": as_text(axis)})
    if limits is not None:
        ET.SubElement(joint, "limit", {
            "lower": f"{limits[0]:.9g}", "upper": f"{limits[1]:.9g}",
            "effort": str(effort), "velocity": str(velocity),
        })


def add_frame(robot: ET.Element, name: str) -> None:
    """A massless link. MoveIt warns about visual-without-collision, not this."""
    ET.SubElement(robot, "link", {"name": name})


def scale_inertial(link: ET.Element, factor: float) -> None:
    """Rescale a link's mass and inertia tensor in place."""
    inertial = link.find("inertial")
    if inertial is None:
        return
    mass = inertial.find("mass")
    if mass is not None:
        mass.set("value", f"{float(mass.get('value')) * factor:.9g}")
    inertia = inertial.find("inertia")
    if inertia is not None:
        for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
            if inertia.get(key) is not None:
                inertia.set(key, f"{float(inertia.get(key)) * factor:.9g}")


def load_v1_forearm(side: str, config: dict) -> tuple[list[ET.Element], list[dict]]:
    """Pull the forearm and wrist out of the vendored DexHand V1 description.

    Returns the three renamed links and, for each requested wrist joint, its
    real origin, axis and limits. Nothing here is invented: the numbers are
    whatever upstream ships, read per side because the left variant is
    separately authored rather than a mirror of the right.
    """
    path = DEXHAND_V1[side]
    if not path.is_file():
        raise SystemExit(
            f"V1 description not found: {path}\n"
            "Expected the vendored copy under ros2_ws/vendor/."
        )
    root = ET.parse(path).getroot()

    density = float(config["wrist"].get("structure_density_kgm3", V1_SOURCE_DENSITY))
    factor = density / V1_SOURCE_DENSITY

    links = []
    for index, source_name in enumerate(V1_LINKS[side]):
        source = root.find(f"link[@name='{source_name}']")
        if source is None:
            raise SystemExit(f"{path}: no link named {source_name}")
        link = copy.deepcopy(source)
        link.set("name", v1_link_name(side, index))
        for mesh in link.iter("mesh"):
            mesh.set("filename", mesh.get("filename").replace(
                "package://dexhand_description/meshes/",
                f"package://{PACKAGE}/meshes_v1/",
            ))
            if "dexhand_description" in mesh.get("filename"):
                raise SystemExit(f"unrewritten V1 mesh path: {mesh.get('filename')}")
        scale_inertial(link, factor)
        links.append(link)

    joints = []
    for name in config["wrist"]["joints"]:
        source = root.find(f"joint[@name='{name}']")
        if source is None:
            raise SystemExit(f"{path}: no joint named {name}")
        origin = source.find("origin")
        axis = source.find("axis")
        limit = source.find("limit")
        joints.append({
            "name": name,
            "xyz": np.fromstring(origin.get("xyz", "0 0 0"), sep=" "),
            "rpy": np.fromstring(origin.get("rpy", "0 0 0"), sep=" "),
            "axis": np.fromstring(axis.get("xyz"), sep=" "),
            "limits": [float(limit.get("lower")), float(limit.get("upper"))],
            "effort": limit.get("effort", "5"),
            "velocity": limit.get("velocity", "3"),
        })
    return links, joints


def v1_link_name(side: str, index: int) -> str:
    return f"arm_{side}_" + ["forearm", "wrist_lower", "wrist_upper"][index]


def build_wrist_chain(robot, side, config, parent):
    """Graft the V1 forearm and wrist. Returns the link the hand attaches to."""
    if not config["wrist"].get("enabled", False):
        return parent

    links, joints = load_v1_forearm(side, config)

    # The forearm link itself carries the mount transform, which the caller has
    # already applied, so it is welded to `parent` at identity.
    robot.append(links[0])
    add_joint(robot, f"arm_{side}_forearm_weld", "fixed", parent,
              links[0].get("name"), {"xyz": [0, 0, 0], "rpy": [0, 0, 0]})
    parent = links[0].get("name")

    # Each wrist joint drives the next V1 link; the last one drives the hand,
    # so it needs a frame of its own for the hand interface to hang off.
    for index, spec in enumerate(joints):
        if index + 1 < len(links):
            child_link = links[index + 1]
            robot.append(child_link)
            child = child_link.get("name")
        else:
            child = f"arm_{side}_wrist_tip"
            add_frame(robot, child)
        add_joint(robot, f"arm_{side}_{spec['name']}_joint", "revolute",
                  parent, child,
                  {"xyz": spec["xyz"], "rpy": spec["rpy"]},
                  spec["axis"], spec["limits"],
                  effort=spec["effort"], velocity=spec["velocity"])
        parent = child

    # Any V1 links past the last requested joint are welded on so their geometry
    # is still present rather than silently dropped.
    for link in links[len(joints) + 1:]:
        robot.append(link)
        add_joint(robot, f"weld_{link.get('name')}", "fixed", parent,
                  link.get("name"), {"xyz": [0, 0, 0], "rpy": [0, 0, 0]})
        parent = link.get("name")

    return parent


def attach_arm(robot, side, config, dexhand, tunable=False):
    """Strip the stock hand and build elbow -> mount -> wrist -> hand."""
    strip_stock_hand(robot, side)
    mount = config["mount"][side]
    parent = ELBOW[side]

    if tunable:
        parent = add_tunable_mount(robot, side, mount, parent)
    else:
        child = f"arm_{side}_mount"
        add_frame(robot, child)
        add_joint(robot, f"arm_{side}_forearm_mount", "fixed", parent, child, mount)
        parent = child

    parent = build_wrist_chain(robot, side, config, parent)

    interface = config["hand_interface"]
    if side == "left":
        interface = {"xyz": mirror_vector(interface["xyz"]),
                     "rpy": mirror_rpy(interface["rpy"])}
    graft(robot, dexhand)
    add_joint(robot, f"arm_{side}_hand_interface", "fixed",
              parent, side_name(DEXHAND_ROOT, side), interface)


def add_tunable_mount(robot, side, mount, parent) -> str:
    """Replace the mount weld with 6 joints whose values *are* the URDF origin.

    Order matters. The three slides come first and share an orientation, so they
    add up in the parent frame exactly like an origin `xyz`. The rotations then
    go yaw, pitch, roll, composing to Rz*Ry*Rx -- which is what URDF `rpy`
    means. Read the six slider values off and they drop straight into
    arm_attachment.yaml with no conversion.
    """
    axes = [
        ("dexhand_mount_x", "prismatic", [1, 0, 0], TUNING_TRANSLATION),
        ("dexhand_mount_y", "prismatic", [0, 1, 0], TUNING_TRANSLATION),
        ("dexhand_mount_z", "prismatic", [0, 0, 1], TUNING_TRANSLATION),
        ("dexhand_mount_yaw", "revolute", [0, 0, 1], TUNING_ROTATION),
        ("dexhand_mount_pitch", "revolute", [0, 1, 0], TUNING_ROTATION),
        ("dexhand_mount_roll", "revolute", [1, 0, 0], TUNING_ROTATION),
    ]
    zero = {"xyz": [0, 0, 0], "rpy": [0, 0, 0]}
    for name, kind, axis, span in axes:
        child = f"arm_{side}_{name}_link"
        add_frame(robot, child)
        add_joint(robot, f"arm_{side}_{name}", kind, parent, child, zero,
                  axis, [-span, span])
        parent = child
    return parent


def write(robot: ET.Element, path: Path, note: str) -> None:
    indent(robot)
    body = ET.tostring(robot, encoding="unicode")
    # NB: XML comments may not contain a double hyphen.
    header = (
        '<?xml version="1.0" ?>\n'
        "<!-- GENERATED by generate_urdf.py. Do not edit by hand. -->\n"
        f"<!-- {note} -->\n"
    )
    path.write_text(header + body + "\n")

    joints = robot.findall("joint")
    movable = [j for j in joints if j.get("type") != "fixed"]
    print(f"wrote {path.name}: {len(robot.findall('link'))} links, "
          f"{len(joints)} joints ({len(movable)} movable)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-dexhand", action="store_true",
                        help="only regenerate the arms-only model")
    args = parser.parse_args()

    if not UPSTREAM.is_file():
        raise SystemExit(f"upstream URDF not found: {UPSTREAM}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()
    write(load_humanoid(config), OUTPUT_DIR / "berkeley_humanoid_lite.urdf",
          "Arms only, stock end effectors.")
    if args.skip_dexhand:
        return

    for side in ("right", "left"):
        mount = config["mount"][side]
        print(f"{side:5s} mount: xyz={as_text(mount['xyz'])} "
              f"rpy={as_text(mount['rpy'])}")

    # Right arm only, hand welded straight on: the DexHand V2 arrangement.
    v2_config = copy.deepcopy(config)
    v2_config["wrist"]["enabled"] = False
    v2_config["hand_interface"] = {"xyz": [0, 0, 0], "rpy": [0, 0, 0]}
    robot = load_humanoid(config)
    attach_arm(robot, "right", v2_config, load_dexhand("right"))
    write(robot, OUTPUT_DIR / "berkeley_humanoid_lite_dexhand.urdf",
          "DexHand v2 (8 servo) welded to the right forearm.")

    # Both arms with the V1 forearm and wrist.
    robot = load_humanoid(config)
    for side in ("right", "left"):
        attach_arm(robot, side, config, load_dexhand(side))
    write(robot, OUTPUT_DIR / "berkeley_humanoid_lite_v1arm.urdf",
          "V1 forearm and 3 DOF wrist on both arms, DexHand v2 fingers.")

    # Same, with the right mount exposed as sliders.
    robot = load_humanoid(config)
    attach_arm(robot, "right", config, load_dexhand("right"), tunable=True)
    attach_arm(robot, "left", config, load_dexhand("left"))
    write(robot, OUTPUT_DIR / "berkeley_humanoid_lite_tuning.urdf",
          "Right mount exposed as 6 sliders; not for planning.")


if __name__ == "__main__":
    main()
