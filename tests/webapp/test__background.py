"""Covers core.services._background directly — the cross-platform detached-
process launcher shared by training.py and extraction.py. Referenced by
test_training_service.py's comment ("launch_detached ... owns all the
shell-wrapping/platform-specific plumbing") but wasn't actually present
until now; the module's platform-specific mechanics (POSIX bash wrapper vs.
Windows cmd wrapper, exit-marker append, pid liveness probe) previously had
no test of their own, only mocked-out call-site assertions in
test_training_service.py/test_views.py.
"""
import sys
import time

from core.services import _background


def _wait_for_marker(log_file, prefix, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = log_file.read_text(errors="replace")
        if prefix in text:
            return text
        time.sleep(0.05)
    raise AssertionError(f"exit marker {prefix!r} never appeared in {log_file}: {log_file.read_text()!r}")


def test_launch_detached_creates_log_file_immediately(tmp_path):
    log_file = tmp_path / "run.log"
    argv = [sys.executable, "-c", "print('hello')"]

    pid = _background.launch_detached(argv, log_file=log_file, exit_marker_prefix="EXIT:", cwd=tmp_path)

    assert isinstance(pid, int) and pid > 0
    # True even if the OS hasn't scheduled the detached process yet -- the
    # file is created synchronously by launch_detached itself, not the
    # child (see the comment in _background.py).
    assert log_file.exists()


def test_launch_detached_writes_output_and_success_exit_marker(tmp_path):
    log_file = tmp_path / "run.log"
    argv = [sys.executable, "-c", "print('hello from child')"]

    _background.launch_detached(argv, log_file=log_file, exit_marker_prefix="EXIT:", cwd=tmp_path)

    text = _wait_for_marker(log_file, "EXIT:")
    assert "hello from child" in text
    assert "EXIT:0" in text


def test_launch_detached_records_nonzero_exit_marker_on_failure(tmp_path):
    log_file = tmp_path / "run.log"
    argv = [sys.executable, "-c", "import sys; sys.exit(3)"]

    _background.launch_detached(argv, log_file=log_file, exit_marker_prefix="EXIT:", cwd=tmp_path)

    text = _wait_for_marker(log_file, "EXIT:")
    assert "EXIT:3" in text


def test_is_pid_alive_true_for_running_process_false_after_it_exits(tmp_path):
    log_file = tmp_path / "run.log"
    argv = [sys.executable, "-c", "import time; time.sleep(1)"]

    pid = _background.launch_detached(argv, log_file=log_file, exit_marker_prefix="EXIT:", cwd=tmp_path)

    assert _background.is_pid_alive(pid) is True
    _wait_for_marker(log_file, "EXIT:", timeout=15.0)
    # Give the OS a moment to actually reap/finalize the process after the
    # marker line is written (the marker is written by the wrapper shell
    # right as the child exits, not necessarily atomically with the OS
    # tearing the process down).
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and _background.is_pid_alive(pid):
        time.sleep(0.05)
    assert _background.is_pid_alive(pid) is False


def test_is_pid_alive_false_for_none():
    assert _background.is_pid_alive(None) is False


def test_launch_detached_append_keeps_prior_log_content(tmp_path):
    log_file = tmp_path / "run.log"
    log_file.write_text("previous attempt's output\n")
    argv = [sys.executable, "-c", "print('hello from resumed run')"]

    _background.launch_detached(argv, log_file=log_file, exit_marker_prefix="EXIT:", cwd=tmp_path, append=True)

    text = _wait_for_marker(log_file, "EXIT:")
    assert "previous attempt's output" in text
    assert "hello from resumed run" in text
