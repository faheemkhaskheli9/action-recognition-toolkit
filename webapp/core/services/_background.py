"""Cross-platform helpers shared by services.training and services.extraction
for launching a detached background OS process and later checking on it.

Both callers need the same two things, done the same way on POSIX and
Windows: (1) start `argv` detached from this (Django) process so it keeps
running after the request/dev-server ends, redirecting its stdout/stderr to
`log_file` and appending an `{exit_marker_prefix}<code>` line once it exits —
even if this process is long gone by then; and (2) later check whether a pid
is still alive, to catch runs that died without ever reaching that marker
line (killed, host restart, crash before the shell wrapper could run).

POSIX: a `bash -c` wrapper does the redirection/marker-append, and
`start_new_session=True` detaches it from our session so it isn't killed by
signals sent to our process group. `os.kill(pid, 0)` is the standard liveness
probe.

Windows has no `bash` on PATH by default, and `os.kill(pid, 0)` never raises
there regardless of whether the pid is alive, so it can't be used as a
liveness probe. Instead: `cmd /v:on /c` does the equivalent wrapping (`!errorlevel!`
needs delayed expansion — `/v:on` — because cmd expands `%errorlevel%` once,
at parse time, before the chained command has even run, which would capture
a stale value); `CREATE_NEW_PROCESS_GROUP` is the detachment equivalent of
`start_new_session` (note: `DETACHED_PROCESS` looks like the more obvious
choice but breaks the inner command's `>` redirection — verified empirically,
not documented); and liveness is checked via `OpenProcess`/`GetExitCodeProcess`.
"""
from __future__ import annotations

import ctypes
import os
import shlex
import subprocess
import threading
from pathlib import Path


def launch_detached(
    argv: list[str], *, log_file: Path, exit_marker_prefix: str, cwd: Path, append: bool = False
) -> int:
    """Start argv as a detached background process.

    Its combined stdout/stderr go to log_file (created/truncated fresh unless
    `append=True`, e.g. resuming a run into its existing log), and once it
    exits a line `{exit_marker_prefix}{returncode}` is appended to that same
    file. Returns the new process's pid.
    """
    # Create it up front (then close immediately -- the child re-opens it by
    # path) so callers can rely on the file existing as soon as this returns,
    # not just once the detached process gets around to writing to it.
    open(log_file, "a" if append else "w").close()
    redirect = ">>" if append else ">"

    if os.name == "nt":
        inner = subprocess.list2cmdline(argv)
        quoted_log = f'"{log_file}"'
        command = f"{inner} {redirect}{quoted_log} 2>&1 & echo {exit_marker_prefix}!errorlevel!>>{quoted_log}"
        # Passed as a raw string, not a list: list2cmdline-style escaping of
        # the outer ["cmd", "/v:on", "/c", command] would double-escape the
        # quotes already inside `command`, and cmd's own /c parser doesn't
        # follow that convention — verified empirically.
        full_command = f'cmd /v:on /c "{command}"'
        proc = subprocess.Popen(
            full_command,
            cwd=cwd,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    else:
        quoted_log = shlex.quote(str(log_file))
        command = f"{shlex.join(argv)} {redirect} {quoted_log} 2>&1; echo {exit_marker_prefix}$? >> {quoted_log}"
        proc = subprocess.Popen(
            ["bash", "-c", command],
            cwd=cwd,
            start_new_session=True,
        )
        # We never otherwise wait()/poll() this Popen -- callers only learn
        # about the run again via is_pid_alive(pid) or the exit-marker line,
        # potentially from a different process entirely. Left unreaped, the
        # wrapper becomes a zombie the moment it exits, and a zombie still
        # answers os.kill(pid, 0) successfully (it's unreaped, not gone), so
        # is_pid_alive would report it alive forever. Reap it in the
        # background as soon as it exits so the pid actually frees up; this
        # doesn't affect the child's detachment (start_new_session already
        # means it survives this process exiting, at which point init
        # reparents and reaps it instead).
        threading.Thread(target=proc.wait, daemon=True).start()
    return proc.pid


def is_pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    if os.name == "nt":
        return _is_pid_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _is_pid_alive_windows(pid: int) -> bool:
    # os.kill(pid, 0) doesn't work as a liveness probe on Windows -- it never
    # raises, alive or not. Ask the OS directly instead.
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)
