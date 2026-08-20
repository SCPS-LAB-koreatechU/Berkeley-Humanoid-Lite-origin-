"""Which generated URDF is this robot, in one place.

Four models are generated, and only one of them is the machine on the bench:

    v1arm    V1 forearm and 3-DOF wrist on both arms, DexHand V2 hands
    dexhand  a V2 hand welded straight to the right elbow, no wrist, no left hand
    stock    the original end effectors, no hand at all
    tuning   v1arm with the right mount exposed as six sliders, for fitting it

`v1arm` is the build: the V1 forearm and wrist carry a V2 hand, which is what
"DexHand V1 with the fingers swapped for V2" means as a kinematic chain. The
other three are narrower views of it and are not what anything should default
to -- `dexhand` in particular has no wrist, so a plan made against it cannot be
executed on the robot.

The scripts and tests import from here rather than each naming a path, because
they drifted: MoveIt already defaulted to v1arm while the measurement tools
still pointed at the wristless model, and the RViz launch did not offer v1arm
at all.
"""

from __future__ import annotations

from pathlib import Path

#: Repository root, from this file's location inside source/.
REPO_ROOT = Path(__file__).resolve().parents[3]

URDF_DIR = REPO_ROOT / "ros2_ws/src/berkeley_humanoid_lite_description/urdf"

MODELS = {
    "v1arm": "berkeley_humanoid_lite_v1arm.urdf",
    "dexhand": "berkeley_humanoid_lite_dexhand.urdf",
    "stock": "berkeley_humanoid_lite.urdf",
    "tuning": "berkeley_humanoid_lite_tuning.urdf",
}

#: The one that matches the hardware. Everything defaults to this.
DEFAULT_MODEL = "v1arm"

#: The upstream DexHand V2 description, where the thumb is still revolute and
#: the mimic tags still name joints that do not exist.
UPSTREAM_DEXHAND = REPO_ROOT / "ros2_ws/vendor/dexhandv2_description/urdf/dexhandv2_right.urdf"

ATTACHMENT_CONFIG = (
    REPO_ROOT / "ros2_ws/src/berkeley_humanoid_lite_description/config/arm_attachment.yaml"
)


def urdf(model: str = DEFAULT_MODEL) -> Path:
    """Path to a generated model's URDF."""
    if model not in MODELS:
        raise KeyError(f"unknown model {model!r}; choose one of {sorted(MODELS)}")
    return URDF_DIR / MODELS[model]
