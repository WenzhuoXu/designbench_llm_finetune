"""Writes that survive a full file system.

Orchard's shared /project/community reached 0 bytes free on 2026-11-10: the trainer died inside
save_ckpt ("Disk quota exceeded (os error 122)") and the queued follow-on job died creating its
first log file. A full shared pool usually does not stay full, so the writers of a run wait and
repeat the write instead of failing the job, and they fsync what they wrote: on NFS a write can
succeed into the page cache and lose its data later, with the error reported only by fsync/close.

No torch here (the rollout runner imports this in the simulator env).
"""
from __future__ import annotations

import errno
import os
import threading
import time
from typing import Callable

# safetensors (Rust), sqlite and others wrap the I/O error and keep only its text
_MESSAGES = ("Disk quota exceeded", "No space left on device", "(os error 122)", "(os error 28)",
             "database or disk is full")


def disk_full_text(s: str) -> bool:
    """An error text (a message, the stderr of a child) that reports a full file system."""
    return any(m in s for m in _MESSAGES)


def disk_full(e: BaseException) -> bool:
    """A write that failed because the file system, or the quota on it, is full."""
    if isinstance(e, OSError) and e.errno is not None:
        return e.errno in (errno.ENOSPC, errno.EDQUOT)
    return disk_full_text(str(e))


def disk_full_cause(e: BaseException) -> bool:
    """disk_full for an exception or for one it was raised from (a simulator that wraps the OSError)."""
    seen = set()
    while e is not None and id(e) not in seen:
        if disk_full(e):
            return True
        seen.add(id(e))
        e = e.__cause__ or e.__context__
    return False


def has_room(dirs, nbytes: int = 1 << 20) -> bool:
    """False when nbytes cannot be written (and fsync'd) in one of dirs because the file system is
    full. For failures that do not say why: numpy's short write ("50000 requested and 2032
    written"), a simulator's own sub-process, SIGBUS from a memory map (measured on a full tmpfs).
    Any other problem with a directory (it is not there) is not a full disk."""
    for d in dirs:
        p = os.path.join(d, ".room_%d_%d" % (os.getpid(), threading.get_ident()))
        try:
            try:
                with open(p, "wb") as f:
                    f.write(b"\0" * nbytes)
                    f.flush()
                    os.fsync(f.fileno())
            finally:
                try:
                    os.remove(p)
                except OSError:
                    pass
        except Exception as e:                               # noqa: BLE001
            if disk_full(e):
                return False
    return True


# exit status of a child process that stopped on a full file system (EX_TEMPFAIL): its parent
# runs it again, where any other status is a crash
DISK_FULL_RC = 75
POLL_S = 60.0


def retry_when_full(fn: Callable, what: str, wait_min: float, log: Callable[[str], None] = print,
                    poll_s: float = None):
    """fn(), repeated while it fails on a full file system: every poll_s seconds for up to wait_min
    minutes in total. After that, or on any other error, the error is raised. fn must be safe to
    repeat (it rewrites whatever an earlier attempt left)."""
    poll_s = POLL_S if poll_s is None else poll_s
    t0 = time.time()
    n = 0
    while True:
        try:
            out = fn()
        except Exception as e:                               # noqa: BLE001
            waited = time.time() - t0
            if not disk_full(e) or waited >= wait_min * 60:
                raise
            if n % 10 == 0:
                log("file system full while writing %s (%s: %s); trying again every %.0f s, %.0f of %.0f min used"
                    % (what, type(e).__name__, str(e)[:200], poll_s, waited / 60, wait_min))
            n += 1
            time.sleep(poll_s)
            continue
        if n:
            log("%s written after waiting %.1f min for disk space" % (what, (time.time() - t0) / 60))
        return out


def fsync_file(path: str) -> None:
    """Flush a file some library wrote; raises the write error the library's close() dropped."""
    fd = os.open(path, os.O_RDWR)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_tree(root: str) -> None:
    for d, _, files in os.walk(root):
        for f in files:
            fsync_file(os.path.join(d, f))


def write_text(path: str, text: str) -> None:
    """Replace path with text, atomically: the file is either the old one or the complete new one."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class WriteErrors:
    """A binary file handed to a library writer (torch.save), remembering the OSError its write()
    raised: the library may re-raise it as something that no longer says why the write failed."""

    def __init__(self, f):
        self.f = f
        self.error = None

    def write(self, b):
        try:
            return self.f.write(b)
        except OSError as e:
            self.error = e
            raise

    def flush(self):
        try:
            self.f.flush()
        except OSError as e:
            self.error = e
            raise

    def __getattr__(self, name):                             # seek, tell, ...
        return getattr(self.f, name)
