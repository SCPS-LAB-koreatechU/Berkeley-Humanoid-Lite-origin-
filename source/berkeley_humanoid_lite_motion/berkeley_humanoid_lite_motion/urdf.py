"""Minimal URDF kinematics.

Only what retargeting needs: the joint tree, fixed and revolute transforms, and
forward kinematics to a named frame. Deliberately not a URDF library -- it
ignores inertials, visuals, collisions and materials, and it does not resolve
xacro. `pinocchio` is the right tool once you need dynamics; this exists so the
retargeting package can be imported without a compiled dependency, and so the
hand chain can be read straight out of the same URDF MoveIt plans against.

Continuous joints are treated as revolute with infinite limits. Prismatic and
planar joints are not supported -- the humanoid has none.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw (X then Y then Z) to a rotation matrix."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' formula. `axis` need not be normalised; a zero axis is identity."""
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    k = axis / norm
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


@dataclass
class Joint:
    """One URDF joint, reduced to what forward kinematics needs."""

    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray            # 4x4, parent link -> joint frame at zero
    axis: np.ndarray              # unit rotation axis in the joint frame
    lower: float
    upper: float
    mimic: tuple[str, float, float] | None = None   # (driver, multiplier, offset)

    @property
    def movable(self) -> bool:
        """The joint has a degree of freedom, whether or not a servo drives it."""
        return self.type in ("revolute", "continuous")

    @property
    def actuated(self) -> bool:
        """A joint the solver may command directly.

        Excludes both ends of the DexHand's reduction: `fixed` joints (the
        thumb, welded at build time) have no freedom at all, and `mimic` joints
        (the flexors and DIPs) have freedom but no servo -- a linkage ties them
        to a knuckle. Reading this off the URDF is what keeps the retargeting
        aiming at the hand that exists.
        """
        return self.movable and self.mimic is None

    def transform(self, angle: float = 0.0) -> np.ndarray:
        if not self.movable:
            return self.origin
        return self.origin @ _transform(axis_angle_to_matrix(self.axis, angle), np.zeros(3))

    def clamp(self, angle: float) -> float:
        return float(np.clip(angle, self.lower, self.upper))


@dataclass
class Chain:
    """A URDF's joint tree, indexed for repeated forward-kinematics queries."""

    joints: dict[str, Joint]
    parent_of: dict[str, str] = field(init=False)   # child link -> joint name

    def __post_init__(self) -> None:
        self.parent_of = {j.child: j.name for j in self.joints.values()}

    @classmethod
    def from_urdf(cls, path: str | Path, repair_mimics: bool = False) -> "Chain":
        """Parse a URDF.

        `repair_mimics` recovers a `<mimic>` whose driver name is missing the
        side prefix the joints carry, which is how the upstream DexHand
        description ships -- every one of its five says `Index_Flexor` where the
        joint is `R_Index_Flexor`. Off by default: a broken reference silently
        turns a coupled joint into a free one, and that should be an error
        unless the caller knows it is reading the unfixed file.
        """
        root = ET.parse(Path(path)).getroot()
        joints: dict[str, Joint] = {}
        for element in root.iter("joint"):
            name = element.get("name")
            jtype = element.get("type")
            parent = element.find("parent")
            child = element.find("child")
            # `<joint>` also appears inside `<transmission>` and `<group>` blocks
            # without parent/child; those are not kinematic joints.
            if name is None or jtype is None or parent is None or child is None:
                continue
            origin = element.find("origin")
            xyz = _floats(origin, "xyz", (0.0, 0.0, 0.0)) if origin is not None else np.zeros(3)
            rpy = _floats(origin, "rpy", (0.0, 0.0, 0.0)) if origin is not None else np.zeros(3)
            axis_el = element.find("axis")
            axis = _floats(axis_el, "xyz", (1.0, 0.0, 0.0)) if axis_el is not None else np.array([1.0, 0.0, 0.0])
            norm = np.linalg.norm(axis)
            if norm > 1e-12:
                axis = axis / norm
            limit = element.find("limit")
            lower = float(limit.get("lower", -np.pi)) if limit is not None else -np.pi
            upper = float(limit.get("upper", np.pi)) if limit is not None else np.pi
            if jtype == "continuous":
                lower, upper = -np.inf, np.inf
            mimic_el = element.find("mimic")
            mimic = None
            if mimic_el is not None:
                mimic = (mimic_el.get("joint"),
                         float(mimic_el.get("multiplier", 1.0)),
                         float(mimic_el.get("offset", 0.0)))
            joints[name] = Joint(
                name=name,
                type=jtype,
                parent=parent.get("link"),
                child=child.get("link"),
                origin=_transform(rpy_to_matrix(rpy), xyz),
                axis=axis,
                lower=lower,
                upper=upper,
                mimic=mimic,
            )
        chain = cls(joints=joints)
        if repair_mimics:
            chain.repair_mimics()
        chain.check_mimics()
        return chain

    def check_mimics(self) -> None:
        """Fail on a `<mimic>` that names a joint the model does not have.

        Worth being strict about: the upstream DexHand description ships five
        such tags -- they say `Index_Flexor` where the joint is `R_Index_Flexor`
        -- and a parser that shrugs at them turns a coupled finger into a free
        one without saying so.
        """
        broken = {
            name: joint.mimic[0] for name, joint in self.joints.items()
            if joint.mimic is not None and joint.mimic[0] not in self.joints
        }
        if broken:
            raise KeyError(f"<mimic> names joints that do not exist: {broken}")

    def repair_mimics(self) -> None:
        """Re-point `<mimic>` drivers that are missing their side prefix.

        Only rewrites when exactly one joint's name ends in the broken one after
        a prefix; an ambiguous or absent match is still an error from
        `check_mimics`. Nothing is guessed beyond the prefix.
        """
        for name, joint in self.joints.items():
            if joint.mimic is None or joint.mimic[0] in self.joints:
                continue
            driver, multiplier, offset = joint.mimic
            candidates = [n for n in self.joints if n.endswith(driver) and n != driver]
            if len(candidates) == 1:
                self.joints[name] = replace(joint, mimic=(candidates[0], multiplier, offset))

    def resolve(self, angles: dict[str, float]) -> dict[str, float]:
        """Fill in every mimic joint's angle from the joints that drive them.

        Iterated rather than applied once, because a coupling can be chained:
        the DexHand's DIP follows its flexor, which itself follows the knuckle.
        """
        resolved = dict(angles)
        for _ in range(len(self.joints)):
            changed = False
            for name, joint in self.joints.items():
                if joint.mimic is None or name in resolved:
                    continue
                driver, multiplier, offset = joint.mimic
                if driver in resolved:
                    resolved[name] = multiplier * resolved[driver] + offset
                    changed = True
            if not changed:
                break
        return resolved

    def path_to(self, link: str, root: str) -> list[str]:
        """Joint names from `root` down to `link`, nearest the root first.

        Raises if `link` is not a descendant of `root`, which is the useful
        failure: a typo'd frame name is otherwise a silently identity transform.
        """
        names: list[str] = []
        current = link
        while current != root:
            joint_name = self.parent_of.get(current)
            if joint_name is None:
                raise KeyError(f"link {link!r} is not a descendant of {root!r}")
            names.append(joint_name)
            current = self.joints[joint_name].parent
        names.reverse()
        return names

    def forward(self, link: str, root: str, angles: dict[str, float] | None = None) -> np.ndarray:
        """4x4 pose of `link` in `root`. Joints absent from `angles` sit at zero.

        Mimic joints are resolved from their drivers, so passing only the
        actuated angles gives the pose the hardware would actually take.
        """
        angles = self.resolve(angles or {})
        T = np.eye(4)
        for joint_name in self.path_to(link, root):
            joint = self.joints[joint_name]
            T = T @ joint.transform(angles.get(joint_name, 0.0))
        return T

    def position(self, link: str, root: str, angles: dict[str, float] | None = None) -> np.ndarray:
        return self.forward(link, root, angles)[:3, 3]


def _floats(element, attribute: str, default: tuple[float, ...]) -> np.ndarray:
    raw = element.get(attribute) if element is not None else None
    if raw is None:
        return np.array(default, dtype=float)
    return np.array([float(v) for v in raw.split()], dtype=float)
