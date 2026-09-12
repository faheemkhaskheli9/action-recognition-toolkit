"""Verify the local environment is ready to run this project.

Checks, in order: Python version, the core library and its required
dependencies import cleanly, the `action_recognition` package itself is
importable, the repo's data/configs directories are in place, the webapp's
Django settings load (skipped if the `webapp` extra isn't installed), and
whether a GPU is available for training/inference.

Each check is independent so one failure (e.g. a missing dependency) doesn't
stop the rest from reporting — a user fixing their environment wants the
full list of what's wrong, not one error at a time. Required checks make
`main()` exit non-zero; optional checks (GPU, webapp) only ever report
`ok=True` or a warning, never a hard failure, since their absence doesn't
prevent the CLI tools from working.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Import name -> package name in requirements, for dependencies whose module
# name differs from the pip package (the common case for these libraries).
REQUIRED_MODULES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "cv2": "opencv-python-headless",
    "numpy": "numpy",
    "pandas": "pandas",
    "yaml": "PyYAML",
    "tqdm": "tqdm",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
}

REQUIRED_DIRS = ["data", "configs", "src/action_recognition"]


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str
    required: bool = True


def check_python_version(minimum: tuple[int, int] = (3, 9)) -> CheckResult:
    actual = sys.version_info[:2]
    ok = actual >= minimum
    version_str = f"{actual[0]}.{actual[1]}"
    minimum_str = f"{minimum[0]}.{minimum[1]}"
    return CheckResult(
        "Python version",
        ok,
        f"{version_str} (>= {minimum_str} required)" if ok else f"{version_str} found, but >= {minimum_str} required",
    )


def check_module_importable(module_name: str, package_name: str) -> CheckResult:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        return CheckResult(
            f"import {module_name}",
            False,
            f"not importable ({exc}); install with `pip install {package_name}`",
        )
    version = getattr(module, "__version__", "unknown version")
    return CheckResult(f"import {module_name}", True, f"{package_name} {version}")


def check_package_installed() -> CheckResult:
    try:
        import action_recognition
    except ImportError as exc:
        return CheckResult(
            "action_recognition package",
            False,
            f"not importable ({exc}); run `pip install -e .` from the repo root",
        )
    return CheckResult("action_recognition package", True, f"version {action_recognition.__version__}")


def check_directory_exists(relative_path: str) -> CheckResult:
    path = REPO_ROOT / relative_path
    ok = path.is_dir()
    return CheckResult(
        f"directory {relative_path}",
        ok,
        str(path) if ok else f"{path} is missing",
    )


def check_gpu_available() -> CheckResult:
    try:
        import torch
    except ImportError:
        return CheckResult("GPU availability", True, "skipped (torch not importable)", required=False)
    if torch.cuda.is_available():
        return CheckResult("GPU availability", True, f"CUDA available ({torch.cuda.get_device_name(0)})", required=False)
    if torch.backends.mps.is_available():
        return CheckResult("GPU availability", True, "MPS (Apple Silicon) available", required=False)
    return CheckResult("GPU availability", True, "none found; training/inference will run on CPU", required=False)


def check_webapp_django_settings() -> CheckResult:
    try:
        import django  # noqa: F401
    except ImportError:
        return CheckResult(
            "webapp Django settings",
            True,
            "skipped (django not installed; run `pip install -e \".[webapp]\"` to use the web app)",
            required=False,
        )

    import os

    # manage.py, the `core` app, and the `webapp` settings package all live
    # directly under the repo root, so that's what needs to be importable.
    added_to_path = str(REPO_ROOT) not in sys.path
    if added_to_path:
        sys.path.insert(0, str(REPO_ROOT))
    previous_settings_module = os.environ.get("DJANGO_SETTINGS_MODULE")
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webapp.settings")
    try:
        django.setup()
        from django.conf import settings

        # Access a setting to force Django to actually finish resolving the
        # settings module (django.setup() alone doesn't validate REPO_ROOT-
        # derived paths like DATA_DIR).
        _ = settings.DATA_DIR
    except Exception as exc:  # noqa: BLE001 - report any settings-load failure, whatever it is
        return CheckResult("webapp Django settings", False, f"failed to load ({exc})")
    finally:
        if added_to_path:
            sys.path.remove(str(REPO_ROOT))
        if previous_settings_module is None:
            os.environ.pop("DJANGO_SETTINGS_MODULE", None)
        else:
            os.environ["DJANGO_SETTINGS_MODULE"] = previous_settings_module
    return CheckResult("webapp Django settings", True, "webapp.settings loaded", required=False)


def run_checks() -> list[CheckResult]:
    checks = [check_python_version()]
    checks += [check_module_importable(module, package) for module, package in REQUIRED_MODULES.items()]
    checks.append(check_package_installed())
    checks += [check_directory_exists(path) for path in REQUIRED_DIRS]
    checks.append(check_webapp_django_settings())
    checks.append(check_gpu_available())
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    checks = run_checks()

    name_width = max(len(check.name) for check in checks)
    for check in checks:
        status = "OK" if check.ok else "FAIL"
        print(f"[{status:>4}] {check.name.ljust(name_width)}  {check.message}")

    required_failures = [check for check in checks if check.required and not check.ok]
    if required_failures:
        print(f"\n{len(required_failures)} required check(s) failed. Fix the items above before running this project.")
        sys.exit(1)
    print("\nAll required checks passed.")


if __name__ == "__main__":
    main()
