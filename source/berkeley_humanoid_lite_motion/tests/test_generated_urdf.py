"""The generated URDFs must be reproducible from the repository alone.

They were not: `generate_urdf.py` picked up an 8-servo description from a path
on one developer's laptop when it happened to exist, and fell back to the full
16-joint hand when it did not. Two machines, two different robots, no error
either way. The variant lives in `arm_attachment.yaml` now, and this checks that
regenerating reproduces what is committed.

Skipped when the assets submodule is not checked out, since the upstream
humanoid URDF is the one input that genuinely is not in this repository.
"""

import subprocess
import sys

import pytest

from conftest import REPO

DESCRIPTION = REPO / "ros2_ws/src/berkeley_humanoid_lite_description"
UPSTREAM_HUMANOID = (
    REPO / "source/berkeley_humanoid_lite_assets/data/robots/berkeley_humanoid"
    / "berkeley_humanoid_lite/urdf/berkeley_humanoid_lite.urdf"
)
MODELS = [
    "berkeley_humanoid_lite.urdf",
    "berkeley_humanoid_lite_dexhand.urdf",
    "berkeley_humanoid_lite_v1arm.urdf",
    "berkeley_humanoid_lite_tuning.urdf",
]

needs_assets = pytest.mark.skipif(
    not UPSTREAM_HUMANOID.is_file(),
    reason="assets submodule not checked out (git submodule update --init)",
)


@needs_assets
def test_regenerating_reproduces_the_committed_urdfs(tmp_path):
    before = {name: (DESCRIPTION / "urdf" / name).read_text() for name in MODELS}
    result = subprocess.run([sys.executable, "generate_urdf.py"],
                            cwd=DESCRIPTION, capture_output=True, text=True)
    after = {name: (DESCRIPTION / "urdf" / name).read_text() for name in MODELS}
    for name, text in before.items():          # restore before asserting
        (DESCRIPTION / "urdf" / name).write_text(text)

    assert result.returncode == 0, result.stderr
    changed = [name for name in MODELS if before[name] != after[name]]
    assert not changed, (
        f"{changed} differ from the committed copy -- regenerate and commit, "
        "or the model on disk is not the model this repository describes"
    )


@needs_assets
def test_the_generator_does_not_read_outside_the_repository():
    """No absolute path into somebody's home directory. That is what made the
    output depend on which machine ran it."""
    for script in ("berkeley_humanoid_lite_description/generate_urdf.py",
                   "berkeley_humanoid_lite_moveit_config/generate_srdf.py"):
        text = (REPO / "ros2_ws/src" / script).read_text()
        offenders = [line.strip() for line in text.splitlines()
                     if '"/home/' in line or "'/home/" in line]
        assert not offenders, f"{script} reads an absolute path: {offenders}"
