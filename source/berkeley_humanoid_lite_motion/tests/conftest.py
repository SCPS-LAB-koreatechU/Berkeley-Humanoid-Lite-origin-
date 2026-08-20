import pytest

from berkeley_humanoid_lite_motion.models import (  # noqa: F401  (re-exported)
    ATTACHMENT_CONFIG, REPO_ROOT, UPSTREAM_DEXHAND, urdf,
)

REPO = REPO_ROOT
GENERATED_URDF = urdf()
UPSTREAM_URDF = UPSTREAM_DEXHAND


@pytest.fixture(scope="session")
def hand():
    from berkeley_humanoid_lite_motion.dexhand import HandModel
    return HandModel.from_urdf(GENERATED_URDF)


@pytest.fixture(scope="session")
def upstream():
    from berkeley_humanoid_lite_motion.urdf import Chain
    return Chain.from_urdf(UPSTREAM_URDF, repair_mimics=True)
