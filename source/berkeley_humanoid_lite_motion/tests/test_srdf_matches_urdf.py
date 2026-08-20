"""Every link and joint an SRDF names must exist in the URDF it describes.

MoveIt refuses a group whose chain ends at a link the model does not have, and
says so only at launch. The arm chains used to end at the DexHand V2 hand's own
root, which the V1-palm build does not have -- generated cleanly, loaded never.
"""

import xml.etree.ElementTree as ET

import pytest

from berkeley_humanoid_lite_motion.models import urdf

from conftest import REPO

SRDF_DIR = REPO / "ros2_ws/src/berkeley_humanoid_lite_moveit_config/config"
SRDF = {
    "v1arm": "berkeley_humanoid_lite_v1arm.srdf",
    "dexhand": "berkeley_humanoid_lite_dexhand.srdf",
    "stock": "berkeley_humanoid_lite.srdf",
}


def model_names(model):
    root = ET.parse(urdf(model)).getroot()
    links = {link.get("name") for link in root.iter("link")}
    joints = {j.get("name") for j in root.iter("joint") if j.find("parent") is not None}
    return links, joints


@pytest.mark.parametrize("model", sorted(SRDF))
def test_srdf_only_names_links_the_urdf_has(model):
    path = SRDF_DIR / SRDF[model]
    if not path.is_file():
        pytest.skip(f"{path.name} not generated")
    links, _ = model_names(model)
    root = ET.parse(path).getroot()

    referenced = []
    for chain in root.iter("chain"):
        referenced += [(chain.get("base_link"), "chain base"),
                       (chain.get("tip_link"), "chain tip")]
    for effector in root.iter("end_effector"):
        referenced.append((effector.get("parent_link"), "end effector"))
    for pair in root.iter("disable_collisions"):
        referenced += [(pair.get("link1"), "collision pair"),
                       (pair.get("link2"), "collision pair")]

    missing = sorted({(name, why) for name, why in referenced if name not in links})
    assert not missing, f"{path.name} names links {model} does not have: {missing}"


@pytest.mark.parametrize("model", sorted(SRDF))
def test_srdf_only_names_joints_the_urdf_has(model):
    path = SRDF_DIR / SRDF[model]
    if not path.is_file():
        pytest.skip(f"{path.name} not generated")
    _, joints = model_names(model)
    root = ET.parse(path).getroot()

    named = {j.get("name") for group in root.iter("group") for j in group.findall("joint")}
    named |= {j.get("name") for state in root.iter("group_state")
              for j in state.findall("joint")}
    missing = sorted(named - joints)
    assert not missing, f"{path.name} names joints {model} does not have: {missing}"


def test_the_arm_chain_ends_at_the_palm():
    """Not at a fixed frame further up, or the hand would not move with a pose
    goal, and not at a link from the other palm variant."""
    links, _ = model_names("v1arm")
    root = ET.parse(SRDF_DIR / SRDF["v1arm"]).getroot()
    tips = {g.get("name"): g.find("chain").get("tip_link")
            for g in root.findall("group") if g.find("chain") is not None}
    assert tips["right_arm"] == "arm_right_index_bulk"
    assert tips["left_arm"] == "arm_left_index_bulk"
    assert set(tips.values()) <= links
