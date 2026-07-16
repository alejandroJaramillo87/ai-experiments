"""Training run lock (Phase 0 D9).

Mechanically enforces the one-GPU-consumer policy: a training job holds
/mnt/ai-data/training/.train.lock (the path scripts/eval/serve_eval.py already
checks) for its whole run; a second job or a serving swap refuses to start
while it exists.

The lock is an O_CREAT|O_EXCL-created JSON file recording pid, run name, and
creation time, released on exit (including exceptions) by the context manager.
Staleness (a lock whose pid is no longer alive, e.g. after a power loss) is
reported, never auto-removed: automatic removal races two simultaneous
starters, so the operator deletes the file by hand as the error instructs.

TRAIN_LOCK_PATH overrides the path (tests point it at a tmp dir; the container
default is the shared training root).
"""

from __future__ import annotations

import contextlib
import json
import os
import time

DEFAULT_LOCK_PATH = "/mnt/ai-data/training/.train.lock"


class TrainLockError(RuntimeError):
    """The lock is already held (or stale); message says how to proceed."""


def lock_path() -> str:
    return os.environ.get("TRAIN_LOCK_PATH", DEFAULT_LOCK_PATH)


def read_lock(path: str | None = None) -> dict | None:
    """The current lock's contents, or None when no lock is held."""
    path = path or lock_path()
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError):
        return {"pid": None, "run": "unknown", "created": "unknown"}


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    return True


@contextlib.contextmanager
def run_lock(run_name: str, path: str | None = None):
    """Hold the training lock for the duration of the with-block.

    Raises TrainLockError when another job holds the lock; the message
    distinguishes a live holder from a stale lock and names the file so the
    operator can remove a stale one manually.
    """
    path = path or lock_path()
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        raise TrainLockError(
            f"Lock directory does not exist: {parent}\n"
            f"  Solution: verify the training root is mounted "
            f"(or set TRAIN_LOCK_PATH)"
        )
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        holder = read_lock(path) or {}
        pid = holder.get("pid")
        if _pid_alive(pid):
            raise TrainLockError(
                f"Training lock held by pid {pid} (run "
                f"{holder.get('run', 'unknown')!r}): {path}\n"
                f"  Solution: wait for the running job to finish "
                f"(one GPU consumer at a time)"
            ) from None
        raise TrainLockError(
            f"Stale training lock (pid {pid} is not alive): {path}\n"
            f"  Solution: confirm no job is running, then remove it: "
            f"rm {path}"
        ) from None
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(
                {
                    "pid": os.getpid(),
                    "run": run_name,
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
                f,
                indent=2,
            )
        yield path
    finally:
        # Remove only our own lock; a crashed-and-replaced holder must not
        # delete a successor's file.
        current_holder = read_lock(path)
        if current_holder and current_holder.get("pid") == os.getpid():
            with contextlib.suppress(FileNotFoundError):
                os.remove(path)
