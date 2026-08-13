"""`__version__` (`src/action_recognition/__init__.py`) and `pyproject.toml`'s
`[project].version` are two hand-maintained copies of the same string with
nothing else checking they match — a release that bumps one and forgets the
other ships silently inconsistent. Regex rather than a TOML parser: the repo
supports Python 3.9+ (`pyproject.toml`'s `requires-python`) and `tomllib` is
3.11+ only; a single anchored `version = "..."` line under `[project]` is
simple enough not to need a real parser or a new dependency.
"""
import re
from pathlib import Path

import action_recognition

REPO_ROOT = Path(__file__).resolve().parent.parent


def _pyproject_version() -> str:
    text = (REPO_ROOT / "pyproject.toml").read_text()
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    assert match, "no top-level `version = \"...\"` line found in pyproject.toml"
    return match.group(1)


def test_package_version_matches_pyproject():
    assert action_recognition.__version__ == _pyproject_version()
