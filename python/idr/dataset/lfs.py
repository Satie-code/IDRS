"""Git LFS pointer detection and parsing.

A Git LFS pointer is a small text stub that stands in for a large file when
the LFS objects were never fetched (downloading a GitHub repo as a ZIP
produces exactly this). Parsing one as if it were CSV yields nonsense, so
every file this package touches is screened here first.

Pointer format (see https://github.com/git-lfs/git-lfs/blob/main/docs/spec.md)::

    version https://git-lfs.github.com/spec/v1
    oid sha256:4d7a2...
    size 19798721
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: Pointers are tiny; anything larger cannot be one, so we never read more.
POINTER_PROBE_BYTES = 1024

_VERSION_PREFIX = b"version https://git-lfs.github.com/spec/v1"
# Tolerate CRLF: a pointer checked out on Windows can carry \r before the
# line break, and rejecting those would misclassify real pointers as data.
_OID_RE = re.compile(r"^oid sha256:([0-9a-f]{64})\r?$", re.MULTILINE)
_SIZE_RE = re.compile(r"^size (\d+)\r?$", re.MULTILINE)


@dataclass(frozen=True)
class LfsPointer:
    """A parsed Git LFS pointer file.

    ``oid`` is the SHA-256 of the *real* content, which makes it both an
    identifier for fetching the object and a checksum for verifying it.
    """

    oid: str
    size: int


def read_pointer(path: Path) -> LfsPointer | None:
    """Return the parsed pointer if ``path`` is an LFS pointer, else ``None``.

    Never raises for ordinary unreadable/binary content: a file that cannot
    be decoded simply is not a pointer.
    """
    try:
        head = path.read_bytes()[:POINTER_PROBE_BYTES]
    except OSError:
        return None
    if not head.startswith(_VERSION_PREFIX):
        return None
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError:
        return None

    oid_match = _OID_RE.search(text)
    size_match = _SIZE_RE.search(text)
    if oid_match is None or size_match is None:
        return None
    return LfsPointer(oid=oid_match.group(1), size=int(size_match.group(1)))


def is_pointer(path: Path) -> bool:
    """True if ``path`` is a Git LFS pointer rather than real content."""
    return read_pointer(path) is not None
