# Contributing

## Setup

```bash
pip install -e ".[webapp,dev]"
```

Python 3.9+. PyTorch/torchvision wheels are large; `pretrained: true` (the
default for model configs) needs network access once to download
ImageNet/Kinetics weights.

## Running tests

```bash
pytest                                  # full suite (library + webapp)
pytest tests/test_models.py             # one file
pytest tests/test_models.py::test_name  # one test
pytest tests/webapp                     # webapp-only tests
```

Run these from the repo root — `[tool.pytest.ini_options]` in
`pyproject.toml` sets `testpaths = ["tests"]` and `pythonpath = ["."]` so the
repo root (where `manage.py`, `webapp/`, and `core/` all live) is importable
and `DJANGO_SETTINGS_MODULE = "webapp.settings"` resolves correctly; this has
to be a pytest ini option rather than fixture setup because pytest-django
needs it earlier than any `conftest.py` runs.

CI (`.github/workflows/tests.yml`) runs the same `pytest` command on every
push/PR to `master` — a green local run is the bar, and CI is that same
bar enforced automatically.

**Every change to `src/` or `core/` should land with matching test
coverage**, same layout as the existing suite: library tests as
`tests/test_<module>.py`, webapp/Django tests under
`tests/webapp/test_<module>.py`. There's no configured lint/format command
— match existing style rather than introducing a new tool.

## Adding a model / detector / tracker

Three independent registries follow the same pattern — write the class,
decorate it, then import the module in that package's `__init__.py`
(registration is an import-time side effect, nothing is auto-discovered):

```python
# src/action_recognition/models/my_model.py
from .registry import register_model
import torch.nn as nn

@register_model("my_model")
class MyModel(nn.Module):
    def __init__(self, num_classes: int, **kwargs): ...
    def forward(self, x):  # (B, T, C, H, W) -> (B, num_classes)
        ...
```

Then add `from . import my_model  # noqa: F401,E402` to
`src/action_recognition/models/__init__.py` next to the existing imports,
and reference `name: my_model` from a config. The same shape applies to
`action_recognition.tracking.detectors` (`detect(frame) -> list[Detection]`)
and `action_recognition.tracking.trackers` (`update`/`finished_tracks`) —
see `configs/tracking/default.yaml` for how they're selected.

## Commit / PR conventions

- Keep unrelated changes in separate commits — a reviewer (or a future
  `git bisect`) should be able to look at one commit and understand one
  thing.
- Don't mention "Claude" (or other AI-attribution) anywhere in code, code
  comments, or git history (commit messages, branch names, PR titles/
  descriptions) — see `CLAUDE.md`'s Attribution section.
- Add a `CHANGELOG.md` entry for anything user-visible.

## Where things are

See `CLAUDE.md` for the full architecture writeup (config loading, the
registry pattern, checkpoint format, webapp service/view split, background-
process liveness handling). It's written for both human contributors and
AI coding assistants working in this repo — read it before making a
non-trivial change.
