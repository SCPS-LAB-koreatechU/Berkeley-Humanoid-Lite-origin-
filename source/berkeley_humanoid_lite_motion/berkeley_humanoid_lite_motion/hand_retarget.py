"""Human hand keypoints to the DexHand's eight servos.

The mapping cannot be joint-to-joint. A human hand has more than twenty degrees
of freedom and an opposable thumb; this one has eight hinges, four rigid fingers
and a thumb bolted in place. Matching joint angles would mean matching angles
that have no counterpart.

So the objective is fingertip *position* in the palm frame, scaled by hand size,
solved for the eight angles that come closest. That is well posed -- the
reachable fingertip set is a two-dimensional surface per finger and the solver
returns the nearest point on it -- and the residual it reports is the honest
measure of what the hardware dropped. Expect tens of millimetres on a curled
hand; that is the hand, not the solver.

Two quantities the robot cannot represent are carried alongside as scalars
rather than silently discarded, so a downstream policy can still condition on
them: `aperture` (thumb-to-index pinch distance) and `curl` (how flexed the
human fingers are). See `MotionSequence.extras`.

Keypoints follow the 21-point hand layout that whole-body 2D pose detectors
emit (wrist, then thumb, index, middle, ring, pinky from base to tip), which is
what a 32-view triangulation of the studio's mp4s produces. `from_mano_order`
converts MANO's ordering if the input came from a mesh fit instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .dexhand import FINGERS, HandModel
from .motion import MotionSequence

#: Index into a 21-keypoint hand, in detector order.
WRIST = 0
THUMB_TIP = 4
FINGER_MCP = {"Index": 5, "Middle": 9, "Ring": 13, "Pinky": 17}
FINGER_PIP = {"Index": 6, "Middle": 10, "Ring": 14, "Pinky": 18}
FINGER_TIP = {"Index": 8, "Middle": 12, "Ring": 16, "Pinky": 20}

HAND21_NAMES = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)

#: MANO orders fingers index, middle, pinky, ring, thumb with the five tips last.
_MANO_TO_DETECTOR = [
    0, 13, 14, 15, 20, 1, 2, 3, 16, 4, 5, 6, 17, 10, 11, 12, 19, 7, 8, 9, 18,
]


def from_mano_order(keypoints: np.ndarray) -> np.ndarray:
    """Reorder (..., 21, 3) MANO joints into detector order."""
    return np.asarray(keypoints)[..., _MANO_TO_DETECTOR, :]


def palm_basis(wrist: np.ndarray, index_mcp: np.ndarray, pinky_mcp: np.ndarray) -> np.ndarray:
    """3x3 rotation whose columns are the palm axes, right-handed.

    Built from the three landmarks a multi-view rig sees most reliably -- the
    wrist and the two outer knuckles are rarely both occluded, whereas any
    fingertip routinely is. Column 0 spans the knuckles, column 2 points along
    the fingers, column 1 completes the frame out of the palm.

    The same construction is applied to the robot (`HandFrame.from_model`), so
    the two frames are comparable by construction rather than by a hand-authored
    offset that would drift the moment the mount changes.
    """
    across = index_mcp - pinky_mcp
    along = 0.5 * (index_mcp + pinky_mcp) - wrist
    x = across / max(np.linalg.norm(across), 1e-9)
    z = along - np.dot(along, x) * x
    z = z / max(np.linalg.norm(z), 1e-9)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


@dataclass
class HandFrame:
    """A palm frame plus the hand's width, which is what sets retargeting scale."""

    origin: np.ndarray
    rotation: np.ndarray
    width: float

    @classmethod
    def from_keypoints(cls, keypoints: np.ndarray) -> "HandFrame":
        wrist = keypoints[WRIST]
        index_mcp = keypoints[FINGER_MCP["Index"]]
        pinky_mcp = keypoints[FINGER_MCP["Pinky"]]
        return cls(
            origin=wrist,
            rotation=palm_basis(wrist, index_mcp, pinky_mcp),
            width=float(np.linalg.norm(index_mcp - pinky_mcp)),
        )

    @classmethod
    def from_model(cls, hand: HandModel) -> "HandFrame":
        """The same construction applied to the robot's own knuckles.

        The robot's wrist landmark is its palm link origin, which sits proximal
        of the knuckles where a wrist would be.
        """
        chain = hand.chain
        mcp = {
            f: chain.position(chain.joints[f"R_{f}_Yaw"].child, hand.palm) for f in FINGERS
        }
        wrist = np.zeros(3)
        return cls(
            origin=wrist,
            rotation=palm_basis(wrist, mcp["Index"], mcp["Pinky"]),
            width=float(np.linalg.norm(mcp["Index"] - mcp["Pinky"])),
        )

    def to_local(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points) - self.origin) @ self.rotation

    def to_world(self, local: np.ndarray) -> np.ndarray:
        return np.asarray(local) @ self.rotation.T + self.origin


#: Distance from a knuckle to its fingertip [m]. Used to convert the joint-space
#: regularisers into metres so every residual the solver sees is the same unit.
#: Without it a weight of 1 on a radian competes with a weight of 1 on a
#: millimetre, and the regularisers win by a factor of a thousand.
FINGER_LEVER = 0.075

#: How far inside its bounds a start point is pushed before solving [rad].
#: `least_squares(method="trf")` estimates its Jacobian by forward differences
#: and clips the step at an active bound, so a variable that starts exactly on
#: one reads as having zero gradient and the solve terminates on its first
#: evaluation. The open hand sits on the pitch lower bound, so this is the
#: default start, not a corner case. The solution may still land on a bound --
#: only the start has to be interior.
BOUND_MARGIN = 1e-3


@dataclass
class RetargetWeights:
    """Objective weights, all applied to residuals in metres.

    `smoothness` couples consecutive frames through the warm start rather than
    through a global solve: each frame is solved independently but pulled toward
    its predecessor. That keeps the problem eight-dimensional and streaming, at
    the cost of a lag that is negligible next to the tracking residual.
    """

    fingertip: float = 1.0
    #: Pulls toward the previous frame. Raise it if triangulation noise makes
    #: the servos chatter; lower it if fast motions smear.
    smoothness: float = 0.1
    #: Pulls toward the open hand, which resolves the redundancy that appears
    #: when a target sits off the reachable surface and many angles tie.
    rest: float = 0.01


@dataclass
class HandRetargeter:
    """Solves 21 human keypoints per frame into the eight servo angles."""

    hand: HandModel
    weights: RetargetWeights = None
    robot_frame: HandFrame = None

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = RetargetWeights()
        if self.robot_frame is None:
            self.robot_frame = HandFrame.from_model(self.hand)

    def targets(self, keypoints: np.ndarray) -> np.ndarray:
        """(4, 3) fingertip targets in the robot palm frame, size-normalised.

        Scaling by the knuckle-span ratio is what makes the retarget invariant
        to who was captured. Without it a large hand simply reads as a more
        open one.
        """
        human = HandFrame.from_keypoints(keypoints)
        scale = self.robot_frame.width / max(human.width, 1e-9)
        local = human.to_local(np.stack([keypoints[FINGER_TIP[f]] for f in FINGERS]))
        return self.robot_frame.to_world(local * scale)

    def solve_targets(self, targets: np.ndarray, q_init=None,
                      confidence=None) -> tuple[np.ndarray, np.ndarray]:
        """Nearest reachable pose to (4, 3) palm-frame fingertip targets.

        Returns the eight angles and the per-finger tip error [m]. The error is
        the point of the exercise as much as the angles are: it is how far the
        hardware fell short on that frame.
        """
        lower, upper = self.hand.limits
        rest = self.hand.rest_pose
        q_prev = rest if q_init is None else self.hand.clamp(q_init)
        start = np.clip(q_prev, lower + BOUND_MARGIN, upper - BOUND_MARGIN)
        w = self.weights
        conf = np.ones(len(FINGERS)) if confidence is None else np.asarray(confidence, float)
        tip_w = w.fingertip * np.sqrt(np.clip(conf, 0.0, 1.0))[:, None]

        def residual(q):
            return np.concatenate([
                ((self.hand.fingertips(q) - targets) * tip_w).ravel(),
                np.sqrt(w.smoothness) * FINGER_LEVER * (q - q_prev),
                np.sqrt(w.rest) * FINGER_LEVER * (q - rest),
            ])

        result = least_squares(residual, start, bounds=(lower, upper), method="trf")
        q = self.hand.clamp(result.x)
        error = np.linalg.norm(self.hand.fingertips(q) - targets, axis=1)
        return q, error

    def solve_sequence(self, keypoints: np.ndarray, fps: float,
                       confidence=None) -> MotionSequence:
        """Retarget a (T, 21, 3) keypoint trajectory.

        Each frame warm-starts from the previous solution, which both smooths
        the result and roughly halves the iteration count.
        """
        keypoints = np.asarray(keypoints, dtype=float)
        if keypoints.ndim != 3 or keypoints.shape[1:] != (21, 3):
            raise ValueError(f"expected (T, 21, 3) keypoints, got {keypoints.shape}")
        if confidence is not None:
            confidence = np.asarray(confidence, dtype=float)

        q_traj, err_traj = [], []
        q = None
        for t, frame in enumerate(keypoints):
            conf = None
            if confidence is not None:
                conf = [confidence[t, FINGER_TIP[f]] for f in FINGERS]
            q, error = self.solve_targets(self.targets(frame), q_init=q, confidence=conf)
            q_traj.append(q)
            err_traj.append(error)

        return MotionSequence(
            fps=fps,
            joint_names=tuple(self.hand.joint_names),
            joint_pos=np.stack(q_traj),
            keypoint_names=HAND21_NAMES,
            keypoint_pos=keypoints,
            keypoint_conf=confidence,
            extras={
                "fingertip_error": np.stack(err_traj).astype(np.float32),
                "aperture": aperture(keypoints).astype(np.float32),
                "curl": curl(keypoints).astype(np.float32),
            },
        )


def aperture(keypoints: np.ndarray) -> np.ndarray:
    """Thumb-to-index pinch distance over knuckle span, per frame.

    Dimensionless so it survives the size normalisation, and retained because
    the robot has no joint that encodes it: with the thumb frozen the closest a
    fingertip ever comes to it is 43 mm, at the best posture the mechanism
    allows. A policy that needs to know the human was pinching has to read it
    from here.
    """
    keypoints = np.atleast_3d(np.asarray(keypoints, dtype=float))
    span = np.linalg.norm(
        keypoints[:, FINGER_MCP["Index"]] - keypoints[:, FINGER_MCP["Pinky"]], axis=-1
    )
    gap = np.linalg.norm(keypoints[:, THUMB_TIP] - keypoints[:, FINGER_TIP["Index"]], axis=-1)
    return gap / np.maximum(span, 1e-9)


def curl(keypoints: np.ndarray) -> np.ndarray:
    """(T, 4) per-finger flexion in [0, 1], 0 extended and 1 fully curled.

    Measured as tip-to-knuckle distance against the finger's own length, so it
    is invariant to hand size and to the palm frame. The robot's rigid fingers
    can only express the first few tens of a percent of this; keeping the full
    signal lets an augmentation or policy stage decide what to do with the rest.
    """
    keypoints = np.atleast_3d(np.asarray(keypoints, dtype=float))
    out = []
    for finger in FINGERS:
        mcp = keypoints[:, FINGER_MCP[finger]]
        pip = keypoints[:, FINGER_PIP[finger]]
        tip = keypoints[:, FINGER_TIP[finger]]
        extended = np.linalg.norm(pip - mcp, axis=-1) * 3.0   # MCP..PIP is ~1/3 of the finger
        out.append(1.0 - np.linalg.norm(tip - mcp, axis=-1) / np.maximum(extended, 1e-9))
    return np.clip(np.stack(out, axis=1), 0.0, 1.0)
