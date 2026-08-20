"""The DexHand v2 8-servo hand, as this robot actually has it.

Everything here is read out of the generated URDF rather than restated, because
the actuated set is a property of the hardware variant and this repository's
`generate_urdf.py` bakes it in: the 8-servo build freezes the whole thumb, both
interphalangeal rows and the flexors to `type="fixed"`. A retargeting objective
written against the upstream 21-joint description would be aiming at joints no
servo can reach.

What is left per finger is two hinges -- a yaw at the palm (spread) and a pitch
at the knuckle (flexion) -- driving a rigid 75 mm finger. `analyse()` reports
what that leaves reachable, in metres, so the pipeline's expectations can be set
from measurements instead of from the joint count.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .urdf import Chain

FINGERS = ("Index", "Middle", "Ring", "Pinky")

#: Servo order. Matches `right_hand_controller.joints` in
#: `moveit_controllers_dexhand.yaml`, so a solved vector can be published as a
#: trajectory point without reordering.
ACTUATED_JOINTS = tuple(
    f"R_{finger}_{axis}" for axis in ("Pitch", "Yaw") for finger in FINGERS
)

#: The frozen thumb still has a pose, and it is the only opposing surface the
#: hand has. Grasp reasoning needs its tip as a fixed point in the palm frame.
THUMB_TIP_FRAME = "Thumb_Tip_1"

#: The V2 hand's own palm link. Only correct when the digits are mounted on a V2
#: palm; with the V1 palm they sit on four separate bulk pieces and there is no
#: such link. `HandModel` derives the frame instead -- see `palm_frame`.
V2_PALM_FRAME = "base_link"
PALM_FRAME = V2_PALM_FRAME


def tip_frame(finger: str) -> str:
    return f"R_{finger}_tip_frame"


def palm_frame(chain: Chain, joint_names=ACTUATED_JOINTS) -> str:
    """The link the fingers are all measured from, derived from the model.

    The lowest common ancestor of the links the finger yaw joints hang off. On a
    V2 palm that is its one printed part; on a V1 palm the digits sit on four
    separate bulk pieces bolted in a row, and the answer is the first of them --
    the piece the wrist drives. Deriving it rather than naming `base_link` is
    what lets the same code measure either build.
    """
    mounts = [chain.joints[name].parent for name in joint_names if name.endswith("_Yaw")]
    if not mounts:
        raise KeyError("no yaw joints; cannot tell where the fingers are mounted")

    def ancestry(link: str) -> list[str]:
        out, seen = [link], {link}
        while (joint := chain.parent_of.get(link)) is not None:
            link = chain.joints[joint].parent
            if link in seen:
                break
            out.append(link)
            seen.add(link)
        return out

    shared = set(ancestry(mounts[0]))
    for mount in mounts[1:]:
        shared &= set(ancestry(mount))
    if not shared:
        raise KeyError(f"finger mounts {mounts} share no ancestor")
    # The lowest of the common ancestors is the deepest in the first mount's
    # own chain, which runs child-to-root.
    for link in ancestry(mounts[0]):
        if link in shared:
            return link
    raise KeyError("unreachable")


@dataclass
class HandModel:
    """Forward kinematics and limits for one DexHand, in its palm frame."""

    chain: Chain
    joint_names: tuple[str, ...] = ACTUATED_JOINTS
    palm: str | None = None

    def __post_init__(self) -> None:
        if self.palm is None:
            self.palm = palm_frame(self.chain, self.joint_names)

    @classmethod
    def from_urdf(cls, path: str | Path) -> "HandModel":
        return cls(chain=Chain.from_urdf(path))

    @property
    def n_dof(self) -> int:
        return len(self.joint_names)

    @property
    def limits(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.array([self.chain.joints[n].lower for n in self.joint_names])
        upper = np.array([self.chain.joints[n].upper for n in self.joint_names])
        return lower, upper

    @property
    def rest_pose(self) -> np.ndarray:
        """Joint zero, clamped into limits -- the URDF's open hand."""
        lower, upper = self.limits
        return np.clip(np.zeros(self.n_dof), lower, upper)

    def clamp(self, q: np.ndarray) -> np.ndarray:
        lower, upper = self.limits
        return np.clip(np.asarray(q, dtype=float), lower, upper)

    def _angles(self, q: np.ndarray) -> dict[str, float]:
        return dict(zip(self.joint_names, np.asarray(q, dtype=float)))

    def fingertips(self, q: np.ndarray) -> np.ndarray:
        """(4, 3) fingertip positions in the palm frame, in FINGERS order."""
        angles = self._angles(q)
        return np.stack([
            self.chain.position(tip_frame(f), self.palm, angles) for f in FINGERS
        ])

    def thumb_tip(self) -> np.ndarray:
        """The frozen thumb tip in the palm frame. Constant by construction."""
        return self.chain.position(THUMB_TIP_FRAME, self.palm)

    @property
    def palm_is_v1(self) -> bool:
        """Whether the digits sit on the V1 bulk chain rather than a V2 palm."""
        return self.palm != V2_PALM_FRAME

    def fingertip_cloud(self, pitch_steps: int = 30, yaw_steps: int = 9) -> np.ndarray:
        """(N, 3) every fingertip position the eight servos can realise.

        A grid rather than a random sample: the actuated space is two dimensions
        per finger, small enough to enumerate, and enumerating makes the minima
        this feeds reproducible.
        """
        points = []
        for finger in FINGERS:
            pitch = self.chain.joints[f"R_{finger}_Pitch"]
            yaw = self.chain.joints[f"R_{finger}_Yaw"]
            for p in np.linspace(pitch.lower, pitch.upper, pitch_steps):
                for y in np.linspace(yaw.lower, yaw.upper, yaw_steps):
                    points.append(self.chain.position(
                        tip_frame(finger), self.palm,
                        {f"R_{finger}_Pitch": p, f"R_{finger}_Yaw": y},
                    ))
        return np.stack(points)

    def analyse(self, samples: int = 20000, seed: int = 0) -> dict:
        """Measure what the actuated space can actually reach.

        Uniform sampling in joint space, which is not uniform in task space --
        the numbers are extents and ranges, not volumes, and that is all they
        are used for.
        """
        rng = np.random.default_rng(seed)
        lower, upper = self.limits
        q = rng.uniform(lower, upper, size=(samples, self.n_dof))
        tips = np.stack([self.fingertips(row) for row in q])   # (N, 4, 3)
        thumb = self.thumb_tip()

        # Opposition aperture: how close any fingertip can be brought to the one
        # opposing surface the hand has, and how far it can open from it.
        to_thumb = np.linalg.norm(tips - thumb, axis=-1)        # (N, 4)

        per_finger = {}
        for i, finger in enumerate(FINGERS):
            extent = tips[:, i, :].max(axis=0) - tips[:, i, :].min(axis=0)
            travel = np.linalg.norm(tips[:, i, :] - tips[:, i, :].mean(axis=0), axis=-1).max() * 2
            per_finger[finger] = {
                "bbox_extent_m": extent,
                "max_travel_m": float(travel),
                "thumb_gap_min_m": float(to_thumb[:, i].min()),
                "thumb_gap_max_m": float(to_thumb[:, i].max()),
            }

        # Spread: how far apart the outer fingertips can be set.
        span = np.linalg.norm(tips[:, 0, :] - tips[:, 3, :], axis=-1)
        return {
            "n_dof": self.n_dof,
            "joint_names": list(self.joint_names),
            "coupled_joints": sorted(
                f"{n} <- {j.mimic[0]} x{j.mimic[1]:g}"
                for n, j in self.chain.joints.items()
                if j.mimic is not None and n.startswith("R_")
            ),
            "frozen_joints": sorted(
                n for n, j in self.chain.joints.items()
                if not j.movable and n.startswith("R_") and "tip_fixed" not in n
            ),
            "per_finger": per_finger,
            "index_pinky_span_m": (float(span.min()), float(span.max())),
            "thumb_tip_palm_m": thumb,
        }


# --- The frozen thumb ------------------------------------------------------
#
# `generate_urdf.py` freezes the thumb at joint zero, which is not a neutral
# choice: zero is the fully extended, splayed posture. Since no servo drives the
# thumb, its posture is a *design parameter* -- whatever angles the mechanism is
# assembled or shimmed at is what the URDF should freeze, and the choice decides
# whether the hand can oppose anything at all. These helpers read the thumb's
# axes and ranges from the upstream description, where the joints are still
# revolute, and measure what a candidate choice buys.
#
# Finger geometry always comes from the *generated* model: the 8-servo build has
# strictly tighter travel than upstream (pitch 0.95 vs 1.309 rad, yaw 0.30 vs
# 0.349), and using upstream's limits would overstate the reachable set. The two
# descriptions agree on frames to within floating point, so mixing them is safe.

THUMB_JOINTS = ("R_Thumb_Yaw", "R_Thumb_Roll", "R_Thumb_Pitch", "R_Thumb_Flexor")

#: Upstream has no `*_tip_frame` links -- `generate_urdf.py` appends them at this
#: offset from each tip link. Needed only to check the two descriptions against
#: each other.
TIP_FRAME_OFFSET = np.array([-0.0011, -0.00309, 0.0179])

#: Upstream misspells the middle finger's tip link, so the mapping is explicit.
UPSTREAM_TIP_LINKS = {
    "Index": "Index_Tip_1", "Middle": "Midle_Tip_1",
    "Ring": "Ring_Tip_1", "Pinky": "Pinky_Tip_1",
}

#: `R_Thumb_DIP` carries `<mimic joint="Thumb_Flexor" multiplier="1" offset="0"/>`
#: upstream, so it is not free -- it follows the flexor exactly.
THUMB_MIMIC = {"R_Thumb_DIP": ("R_Thumb_Flexor", 1.0, 0.0)}


def thumb_tip_at(upstream: Chain, thumb_angles) -> np.ndarray:
    """Thumb tip in the palm frame for a candidate frozen posture.

    `upstream` must be the vendored DexHand URDF; the generated one has the
    thumb already fixed and has therefore discarded its axes.
    """
    angles = dict(zip(THUMB_JOINTS, np.asarray(thumb_angles, dtype=float)))
    for driven, (driver, multiplier, offset) in THUMB_MIMIC.items():
        angles[driven] = multiplier * angles[driver] + offset
    return upstream.position(THUMB_TIP_FRAME, PALM_FRAME, angles)


def opposition_gap(hand: HandModel, upstream: Chain, thumb_angles,
                   cloud: np.ndarray | None = None) -> float:
    """Smallest fingertip-to-thumb-tip distance [m] for a frozen thumb posture.

    The hand's best case pinch aperture. Much above ~30 mm there is no pinch at
    all, only caging an object against the palm.
    """
    cloud = hand.fingertip_cloud() if cloud is None else cloud
    return float(np.linalg.norm(cloud - thumb_tip_at(upstream, thumb_angles), axis=1).min())


def best_thumb_posture(hand: HandModel, upstream: Chain,
                       restarts: int = 8, seed: int = 0) -> tuple[np.ndarray, float]:
    """Frozen thumb angles minimising `opposition_gap`, and that gap.

    Powell from several starts: the objective is a minimum over a point cloud,
    so it is piecewise smooth and a gradient method stalls on the seams.
    """
    from scipy.optimize import minimize

    cloud = hand.fingertip_cloud()
    lower = np.array([upstream.joints[n].lower for n in THUMB_JOINTS])
    upper = np.array([upstream.joints[n].upper for n in THUMB_JOINTS])
    rng = np.random.default_rng(seed)
    best_x, best_f = np.zeros(len(THUMB_JOINTS)), np.inf
    for _ in range(restarts):
        result = minimize(
            lambda q: opposition_gap(hand, upstream, q, cloud),
            rng.uniform(lower, upper),
            method="Powell",
            bounds=list(zip(lower, upper)),
        )
        if result.fun < best_f:
            best_x, best_f = result.x, float(result.fun)
    return best_x, best_f
