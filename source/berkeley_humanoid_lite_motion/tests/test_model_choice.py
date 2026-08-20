"""The one model everything is supposed to agree on.

Four URDFs are generated and only `v1arm` is the machine on the bench. The
tools disagreed about that for a while -- MoveIt planned against v1arm while the
measurement scripts analysed the wristless model and the RViz launch did not
offer v1arm at all -- so the choice is pinned here.
"""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from berkeley_humanoid_lite_motion.dexhand import FINGERS, HandModel
from berkeley_humanoid_lite_motion.models import DEFAULT_MODEL, MODELS, urdf

from conftest import REPO


def joints_of(model):
    root = ET.parse(urdf(model)).getroot()
    return [j for j in root.iter("joint") if j.find("parent") is not None]


def test_the_default_model_is_the_one_with_a_wrist():
    """A plan made against a wristless model cannot be executed on a robot that
    has one, so this is the property that decides the default."""
    assert DEFAULT_MODEL == "v1arm"
    wrist = [j for j in joints_of(DEFAULT_MODEL)
             if "wrist" in j.get("name") and j.get("type") != "fixed"]
    assert len(wrist) == 6                       # three per arm


def test_the_default_model_carries_both_hands():
    prefixes = {(j.get("name") or "")[:2] for j in joints_of(DEFAULT_MODEL)}
    assert {"R_", "L_"} <= prefixes


def test_every_named_model_exists():
    for model in MODELS:
        assert urdf(model).is_file(), f"{model} is named but not generated"
    with pytest.raises(KeyError):
        urdf("nonesuch")


def test_the_hand_is_the_same_hand_in_every_model_that_has_one():
    """Measurements are made on one model and used with another, so the finger
    geometry has to be identical -- otherwise a coupling ratio fitted against
    `dexhand` would quietly not apply to the robot."""
    reference = HandModel.from_urdf(urdf("v1arm"))
    other = HandModel.from_urdf(urdf("dexhand"))
    assert reference.joint_names == other.joint_names
    rng = np.random.default_rng(0)
    lower, upper = reference.limits
    for _ in range(5):
        q = rng.uniform(lower, upper)
        assert np.allclose(reference.fingertips(q), other.fingertips(q), atol=1e-12)


def test_both_hands_couple_the_same_way():
    """The left hand is mirrored, not re-derived, so its coupling must match."""
    chain = HandModel.from_urdf(urdf("v1arm")).chain
    for finger in FINGERS:
        for row, driver in (("Flexor", "Pitch"), ("DIP", "Flexor")):
            right = chain.joints[f"R_{finger}_{row}"].mimic
            left = chain.joints[f"L_{finger}_{row}"].mimic
            assert right is not None and left is not None
            assert right[0] == f"R_{finger}_{driver}"
            assert left[0] == f"L_{finger}_{driver}"
            assert right[1:] == left[1:]         # same multiplier and offset


def test_the_scripts_and_the_rviz_launch_point_at_the_default():
    """Three places used to name a model independently and drifted apart."""
    launch = (REPO / "ros2_ws/src/berkeley_humanoid_lite_description/launch"
              / "display.launch.py").read_text()
    assert f'default_value="{DEFAULT_MODEL}"' in launch
    assert f'"{DEFAULT_MODEL}"' in launch.split("MODELS = {")[1].split("}")[0]

    moveit = (REPO / "ros2_ws/src/berkeley_humanoid_lite_moveit_config/launch"
              / "demo.launch.py").read_text()
    assert f'default_value="{DEFAULT_MODEL}"' in moveit

    for script in ("analyze_hand.py", "fit_finger_coupling.py", "track_finger_joints.py"):
        text = (REPO / "scripts/motion" / script).read_text()
        assert "berkeley_humanoid_lite_dexhand.urdf" not in text, (
            f"{script} names a model directly; import it from models.py instead")
