"""
File-based locking for the bootstrap process.

Prevents parallel builds when Streamlit reruns the script concurrently
(multiple browser tabs, hot reload, etc.).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout


@contextmanager
def acquire_lock(lock_path: str | Path, timeout: float = 120.0):
    """
    Acquire a file lock at *lock_path*.

    Args:
        lock_path: Path to the lock file (created if absent).
        timeout: Seconds to wait before raising Timeout.

    Raises:
        filelock.Timeout if lock cannot be acquired within *timeout* seconds.
    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_path), timeout=timeout)
    with lock:
        yield
