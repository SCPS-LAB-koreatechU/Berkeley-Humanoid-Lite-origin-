"""The interchange format every stage of the pipeline reads and writes.

Capture, retargeting, augmentation and the RL reference buffer all need to hand
each other the same thing: a time series of robot joint angles, optionally with
the root pose, the human keypoints it came from, and per-frame scalars that the
robot cannot represent as a joint (grasp aperture being the one that matters
here). Keeping that in one container with one on-disk form is what stops each
stage from inventing its own npz layout.

Stored as npz because it is dependency-free, memory-mappable and diffable by
shape. Frames are uniformly spaced at `fps`; a variable-rate capture must be
resampled on the way in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class MotionSequence:
    """One retargeted (or captured) motion.

    Attributes:
        fps: Frame rate. Frames are uniformly spaced.
        joint_names: Names in the same order as `joint_pos`'s columns. Keeping
            names with the data is what lets the RL side reorder into its own
            actuator ordering instead of trusting two configs to agree.
        joint_pos: (T, J) joint angles [rad].
        root_pos: (T, 3) root position [m], or None for a fixed-base clip.
        root_quat: (T, 4) root orientation as wxyz, or None. wxyz rather than
            xyzw to match Isaac Lab, which is where these end up.
        keypoint_names: Names for `keypoint_pos`'s second axis, or None.
        keypoint_pos: (T, K, 3) source keypoints [m], kept alongside the joints
            so a retarget can be re-run or scored without re-ingesting video.
        keypoint_conf: (T, K) per-keypoint confidence in [0, 1], or None.
        extras: Named (T, ...) arrays for anything the robot has no joint for.
    """

    fps: float
    joint_names: tuple[str, ...]
    joint_pos: np.ndarray
    root_pos: np.ndarray | None = None
    root_quat: np.ndarray | None = None
    keypoint_names: tuple[str, ...] | None = None
    keypoint_pos: np.ndarray | None = None
    keypoint_conf: np.ndarray | None = None
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.joint_pos = np.asarray(self.joint_pos, dtype=np.float32)
        if self.joint_pos.ndim != 2:
            raise ValueError(f"joint_pos must be (T, J), got {self.joint_pos.shape}")
        if self.joint_pos.shape[1] != len(self.joint_names):
            raise ValueError(
                f"{self.joint_pos.shape[1]} joint columns for {len(self.joint_names)} names"
            )
        for name in ("root_pos", "root_quat", "keypoint_pos", "keypoint_conf"):
            value = getattr(self, name)
            if value is None:
                continue
            value = np.asarray(value, dtype=np.float32)
            if len(value) != len(self.joint_pos):
                raise ValueError(f"{name} has {len(value)} frames, joint_pos has {len(self)}")
            setattr(self, name, value)
        for key, value in self.extras.items():
            if len(value) != len(self.joint_pos):
                raise ValueError(f"extras[{key!r}] has {len(value)} frames, joint_pos has {len(self)}")

    def __len__(self) -> int:
        return len(self.joint_pos)

    @property
    def duration(self) -> float:
        return len(self) / self.fps

    @property
    def times(self) -> np.ndarray:
        return np.arange(len(self), dtype=np.float64) / self.fps

    def joint_index(self, name: str) -> int:
        return self.joint_names.index(name)

    def select(self, names) -> np.ndarray:
        """(T, len(names)) columns in the requested order.

        The RL environment and the MoveIt controller order joints differently
        from each other and from the capture; ask by name rather than slicing.
        """
        return self.joint_pos[:, [self.joint_index(n) for n in names]]

    def slice(self, start: int, stop: int) -> "MotionSequence":
        def take(a):
            return None if a is None else a[start:stop]

        return MotionSequence(
            fps=self.fps,
            joint_names=self.joint_names,
            joint_pos=self.joint_pos[start:stop],
            root_pos=take(self.root_pos),
            root_quat=take(self.root_quat),
            keypoint_names=self.keypoint_names,
            keypoint_pos=take(self.keypoint_pos),
            keypoint_conf=take(self.keypoint_conf),
            extras={k: v[start:stop] for k, v in self.extras.items()},
        )

    def resample(self, fps: float) -> "MotionSequence":
        """Linear resampling onto a new frame rate.

        Linear and not slerp: `root_quat` is renormalised afterwards, which is
        accurate to well under the retargeting residual at any sane frame rate,
        and keeps one code path for every array.
        """
        if fps <= 0:
            raise ValueError("fps must be positive")
        n = max(int(round(self.duration * fps)), 1)
        src = self.times
        dst = np.arange(n, dtype=np.float64) / fps

        def interp(a):
            if a is None:
                return None
            flat = a.reshape(len(a), -1)
            out = np.stack([np.interp(dst, src, flat[:, i]) for i in range(flat.shape[1])], axis=1)
            return out.reshape((n,) + a.shape[1:]).astype(np.float32)

        quat = interp(self.root_quat)
        if quat is not None:
            quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        return MotionSequence(
            fps=fps,
            joint_names=self.joint_names,
            joint_pos=interp(self.joint_pos),
            root_pos=interp(self.root_pos),
            root_quat=quat,
            keypoint_names=self.keypoint_names,
            keypoint_pos=interp(self.keypoint_pos),
            keypoint_conf=interp(self.keypoint_conf),
            extras={k: interp(v) for k, v in self.extras.items()},
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "fps": np.asarray(self.fps),
            "joint_names": np.asarray(self.joint_names),
            "joint_pos": self.joint_pos,
        }
        for name in ("root_pos", "root_quat", "keypoint_pos", "keypoint_conf"):
            value = getattr(self, name)
            if value is not None:
                arrays[name] = value
        if self.keypoint_names is not None:
            arrays["keypoint_names"] = np.asarray(self.keypoint_names)
        for key, value in self.extras.items():
            arrays[f"extra__{key}"] = value
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MotionSequence":
        with np.load(Path(path), allow_pickle=False) as data:
            keys = set(data.files)

            def get(key):
                return data[key] if key in keys else None

            names = get("keypoint_names")
            return cls(
                fps=float(data["fps"]),
                joint_names=tuple(str(n) for n in data["joint_names"]),
                joint_pos=data["joint_pos"],
                root_pos=get("root_pos"),
                root_quat=get("root_quat"),
                keypoint_names=None if names is None else tuple(str(n) for n in names),
                keypoint_pos=get("keypoint_pos"),
                keypoint_conf=get("keypoint_conf"),
                extras={k[len("extra__"):]: data[k] for k in sorted(keys) if k.startswith("extra__")},
            )
