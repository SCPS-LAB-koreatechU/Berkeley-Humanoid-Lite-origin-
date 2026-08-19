from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
GENERATED_URDF = REPO / "ros2_ws/src/berkeley_humanoid_lite_description/urdf/berkeley_humanoid_lite_dexhand.urdf"
UPSTREAM_URDF = REPO / "ros2_ws/vendor/dexhandv2_description/urdf/dexhandv2_right.urdf"


@pytest.fixture(scope="session")
def hand():
    from berkeley_humanoid_lite_motion.dexhand import HandModel
    return HandModel.from_urdf(GENERATED_URDF)


@pytest.fixture(scope="session")
def upstream():
    from berkeley_humanoid_lite_motion.urdf import Chain
    return Chain.from_urdf(UPSTREAM_URDF)
