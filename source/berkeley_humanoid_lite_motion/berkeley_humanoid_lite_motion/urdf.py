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
from dataclasses import dataclass, field
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

    @property
    def actuated(self) -> bool:
        """A joint the solver may move. `fixed` joints are structure, not freedom.

        The DexHand's 8-servo variant freezes the thumb, both interphalangeal
        rows and the flexors this way, so reading `actuated` off the URDF is
        what keeps the retargeting honest about the hand it is aiming at.
        """
        return self.type in ("revolute", "continuous")

    def transform(self, angle: float = 0.0) -> np.ndarray:
        if not self.actuated:
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
    def from_urdf(cls, path: str | Path) -> "Chain":
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
            joints[name] = Joint(
                name=name,
                type=jtype,
                parent=parent.get("link"),
                child=child.get("link"),
                origin=_transform(rpy_to_matrix(rpy), xyz),
                axis=axis,
                lower=lower,
                upper=upper,
            )
        return cls(joints=joints)

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
        """4x4 pose of `link` in `root`. Joints absent from `angles` sit at zero."""
        angles = angles or {}
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
