import pytest

from action_recognition.scripts import check_setup
from action_recognition.scripts.check_setup import CheckResult, main, run_checks


def test_run_checks_passes_in_this_dev_environment():
    """Guards against the check script itself drifting from the actual
    requirements (e.g. a dependency added to pyproject.toml but not to
    REQUIRED_MODULES) — every *required* check must pass in the environment
    this test suite runs in, since that environment is exactly what
    `pip install -e ".[webapp,dev]"` (README's documented install step) produces.
    """
    checks = run_checks()
    required_failures = [c for c in checks if c.required and not c.ok]
    assert not required_failures, required_failures


def test_run_checks_covers_every_required_dependency():
    checks = run_checks()
    names = {c.name for c in checks}
    for module_name in check_setup.REQUIRED_MODULES:
        assert f"import {module_name}" in names


def test_check_python_version_fails_below_minimum():
    result = check_setup.check_python_version(minimum=(99, 0))
    assert result.ok is False
    assert "99.0" in result.message


def test_check_python_version_passes_at_or_above_minimum():
    result = check_setup.check_python_version(minimum=(3, 9))
    assert result.ok is True


def test_check_module_importable_reports_failure_for_missing_module():
    result = check_setup.check_module_importable("no_such_module_xyz", "no-such-package")
    assert result.ok is False
    assert "pip install no-such-package" in result.message


def test_check_module_importable_reports_success_for_real_module():
    result = check_setup.check_module_importable("json", "json")
    assert result.ok is True


def test_check_directory_exists_fails_for_missing_path():
    result = check_setup.check_directory_exists("no/such/directory")
    assert result.ok is False


def test_check_directory_exists_passes_for_real_path():
    result = check_setup.check_directory_exists("src/action_recognition")
    assert result.ok is True


def test_check_webapp_django_settings_is_never_a_required_failure():
    # Whether or not the `webapp` extra is installed, this check must not be
    # able to fail the overall `main()` run on its own (see docstring: it's
    # optional). It's allowed to fail (required=False) if Django settings
    # are actually broken, but never marked required=True.
    result = check_setup.check_webapp_django_settings()
    assert result.required is False


def test_main_exits_zero_when_all_required_checks_pass(monkeypatch, capsys):
    passing = [CheckResult("dummy required", True, "fine"), CheckResult("dummy optional", False, "meh", required=False)]
    monkeypatch.setattr(check_setup, "run_checks", lambda: passing)
    monkeypatch.setattr("sys.argv", ["ar-check-setup"])

    main()  # must not raise/exit

    out = capsys.readouterr().out
    assert "All required checks passed." in out


def test_main_exits_nonzero_when_a_required_check_fails(monkeypatch, capsys):
    failing = [CheckResult("dummy required", False, "broken")]
    monkeypatch.setattr(check_setup, "run_checks", lambda: failing)
    monkeypatch.setattr("sys.argv", ["ar-check-setup"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    assert "1 required check(s) failed" in capsys.readouterr().out
