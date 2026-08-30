"""File inventory, dataset-family classification, and session discovery.

Everything here is derived from evidence actually present on disk — the
directory layout and filenames of the IO-VNBD tree — and each derived
attribute records the :class:`~idr.dataset.evidence.Evidence` level that
supports it. Where the layout does not justify a conclusion (a driver name,
a country), the field is left ``None`` rather than guessed.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from idr.dataset.evidence import Evidence
from idr.dataset.lfs import read_pointer

#: Files at or below this size get a full SHA-256; larger ones are skipped
#: rather than hashed from a sample, so a recorded hash is always the whole file.
HASH_SIZE_LIMIT_BYTES = 64 * 1024 * 1024

_STREAM_PREFIX_RE = re.compile(r"^(?P<stream>[SV])-(?P<session>.+)$", re.IGNORECASE)
_DRIVER_DIR_RE = re.compile(r"^(?P<label>.+?)\s*\(Driver\s+(?P<driver>[A-Z])\)$", re.IGNORECASE)


@dataclass(frozen=True)
class FileRecord:
    """One file discovered under the dataset root."""

    relative_path: str
    filename: str
    extension: str
    size_bytes: int
    is_file: bool
    is_directory: bool
    is_lfs_pointer: bool
    sha256: str | None
    sha256_scope: str
    lfs_oid: str | None
    lfs_size: int | None
    dataset_family: str
    stream: str | None
    session_id: str | None
    session_id_evidence: str


@dataclass
class SessionRecord:
    """One logical recording session, possibly with paired S/V streams."""

    session_id: str
    dataset_family: str
    driver: str | None
    driver_evidence: str
    category: str | None
    country: str | None
    country_evidence: str
    smartphone_files: list[str] = field(default_factory=list)
    vehicle_files: list[str] = field(default_factory=list)

    @property
    def is_paired(self) -> bool:
        """Both an S- and a V- file exist for this session id."""
        return bool(self.smartphone_files) and bool(self.vehicle_files)

    @property
    def pairing_evidence(self) -> str:
        # Filename convention only. Whether the streams are actually time
        # aligned is a separate question answered in synchronization.py.
        return Evidence.INFERRED if self.is_paired else Evidence.UNVERIFIED


def classify_family(relative_path: str) -> str:
    """Classify a file into a dataset family from its location in the tree.

    Uses only directory names that actually exist in the IO-VNBD layout.
    """
    lowered = relative_path.lower()
    if lowered.endswith((".md", ".pdf", ".py", ".gitattributes")):
        return "documentation"
    if lowered.endswith(".jpg"):
        return "imagery"

    synchronized = "synchronised" in lowered and "unsynchronised" not in lowered
    unsynchronized = "unsynchronised" in lowered
    categorized = "uncategorised" not in lowered and "categorised" in lowered

    if synchronized:
        base = "synchronized"
    elif unsynchronized:
        base = "unsynchronized"
    else:
        return "other"
    return f"{base}_{'categorized' if categorized else 'uncategorized'}"


def _stream_and_session(filename: str) -> tuple[str | None, str | None]:
    """Split ``S-S1.csv`` into stream ``S`` and session id ``S1``.

    The dataset mixes capitalization (``V-vta11.csv`` vs ``S-Vta11.csv``), so
    session ids are normalized to upper case for pairing.
    """
    stem = Path(filename).stem
    match = _STREAM_PREFIX_RE.match(stem)
    if match is None:
        return None, None
    stream = "smartphone" if match.group("stream").upper() == "S" else "vehicle"
    return stream, match.group("session").upper()


def _driver_and_category(relative_path: str) -> tuple[str | None, str | None]:
    """Extract driver letter and category label from a ``X (Driver B)`` directory."""
    for part in Path(relative_path).parts:
        match = _DRIVER_DIR_RE.match(part)
        if match is not None:
            return match.group("driver").upper(), match.group("label").strip()
    return None, None


def _sha256_of(path: Path) -> tuple[str | None, str]:
    size = path.stat().st_size
    if size > HASH_SIZE_LIMIT_BYTES:
        return None, f"omitted (>{HASH_SIZE_LIMIT_BYTES} bytes)"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest(), "full_file"


def build_file_inventory(root: Path, *, compute_hashes: bool = True) -> list[FileRecord]:
    """Walk ``root`` and record every file, flagging LFS pointers explicitly."""
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    records: list[FileRecord] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            records.append(
                FileRecord(
                    relative_path=relative,
                    filename=path.name,
                    extension="",
                    size_bytes=0,
                    is_file=False,
                    is_directory=True,
                    is_lfs_pointer=False,
                    sha256=None,
                    sha256_scope="not_applicable",
                    lfs_oid=None,
                    lfs_size=None,
                    dataset_family=classify_family(relative),
                    stream=None,
                    session_id=None,
                    session_id_evidence=Evidence.UNVERIFIED,
                )
            )
            continue

        pointer = read_pointer(path)
        stream, session_id = _stream_and_session(path.name)
        sha256, scope = (None, "not_computed")
        if compute_hashes:
            sha256, scope = _sha256_of(path)

        records.append(
            FileRecord(
                relative_path=relative,
                filename=path.name,
                extension=path.suffix.lower(),
                size_bytes=path.stat().st_size,
                is_file=True,
                is_directory=False,
                is_lfs_pointer=pointer is not None,
                sha256=sha256,
                sha256_scope=scope,
                lfs_oid=pointer.oid if pointer else None,
                lfs_size=pointer.size if pointer else None,
                dataset_family=classify_family(relative),
                stream=stream,
                session_id=session_id,
                session_id_evidence=(Evidence.INFERRED if session_id else Evidence.UNVERIFIED),
            )
        )
    return records


def discover_sessions(records: list[FileRecord]) -> list[SessionRecord]:
    """Group data files into sessions keyed by ``(family, session_id)``.

    Sessions are keyed by family as well as id because the same id can appear
    in both the synchronized and unsynchronized trees, and those are different
    recordings of the same route — merging them would fabricate pairings.
    """
    sessions: dict[tuple[str, str], SessionRecord] = {}

    for record in records:
        if not record.is_file or record.session_id is None or record.extension != ".csv":
            continue
        driver, category = _driver_and_category(record.relative_path)
        key = (record.dataset_family, record.session_id)
        session = sessions.get(key)
        if session is None:
            session = SessionRecord(
                session_id=record.session_id,
                dataset_family=record.dataset_family,
                driver=driver,
                driver_evidence=(Evidence.VERIFIED_FROM_FILE if driver else Evidence.UNVERIFIED),
                category=category,
                # IO-VNBD spans the UK, Nigeria and France, but nothing in the
                # directory layout says which session was recorded where.
                country=None,
                country_evidence=Evidence.UNVERIFIED,
            )
            sessions[key] = session

        if record.stream == "smartphone":
            session.smartphone_files.append(record.relative_path)
        elif record.stream == "vehicle":
            session.vehicle_files.append(record.relative_path)

    return [sessions[key] for key in sorted(sessions)]


def file_records_to_dicts(records: list[FileRecord]) -> list[dict[str, object]]:
    """Serialization helper for manifest/CSV writing."""
    return [asdict(record) for record in records]


def find_duplicate_groups(records: list[FileRecord]) -> dict[str, list[str]]:
    """Group data files by content hash, returning only genuine duplicates.

    IO-VNBD ships the same recordings in several places at once (a session can
    appear in both the categorized and uncategorized trees), so a file-level
    train/test split would put byte-identical data on both sides. Detecting
    this is what makes Rule 9 (no information leakage) enforceable in Phase 2.

    Returns ``{sha256: [relative_path, ...]}`` sorted for determinism.
    """
    by_hash: dict[str, list[str]] = {}
    for record in records:
        if not record.is_file or record.extension != ".csv" or not record.sha256:
            continue
        by_hash.setdefault(record.sha256, []).append(record.relative_path)
    return {digest: sorted(paths) for digest, paths in sorted(by_hash.items()) if len(paths) > 1}


def canonical_paths(records: list[FileRecord]) -> set[str]:
    """One representative path per distinct content hash.

    Use this to total durations or row counts without double-counting the
    duplicated copies described in :func:`find_duplicate_groups`.
    """
    seen: dict[str, str] = {}
    for record in records:
        if not record.is_file or record.extension != ".csv":
            continue
        key = record.sha256 or record.relative_path
        if key not in seen or record.relative_path < seen[key]:
            seen[key] = record.relative_path
    return set(seen.values())
