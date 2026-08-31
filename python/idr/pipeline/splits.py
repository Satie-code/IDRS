"""Leakage-safe train/validation/test split generation.

Three rules, each of which Phase 1 or Phase 2 showed to be necessary:

1. **Group by payload hash, not filename.** Phase 1 found 235 byte-identical
   duplicate files; Phase 2 found 15 further groups whose bytes differ only in
   header labels while the data rows are identical. Splitting on path — or even
   on the whole-file hash — would put the same recording on both sides.

2. **A segment never straddles a split.** Whole logical trips are assigned as
   units, so overlapping windows drawn from one trip cannot land in different
   partitions.

3. **Deterministic.** Assignment is a pure function of the group key and the
   configured seed, so the same configuration always reproduces the same split.
   No wall-clock or iteration-order dependence.

Group keys are further merged across sessions that share a payload: if two
sessions contain the same recording, they form one indivisible group.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

DEFAULT_RATIOS: dict[str, float] = {"train": 0.7, "validation": 0.15, "test": 0.15}
DEFAULT_SEED = "sih26168-phase2"

#: Canonical prefixes of measured sensor channels, used by the content audit.
_SENSOR_PREFIXES = (
    "accel_",
    "gyro_",
    "mag_",
    "gravity_",
    "orientation_",
    "gnss_",
    "latitude",
    "longitude",
    "altitude_",
    "speed_",
    "heading_",
    "yaw_rate",
    "vehicle_",
)

#: Rounding applied before fingerprinting, so float formatting cannot mask a
#: duplicate while remaining far finer than any sensor's real resolution.
_FINGERPRINT_DECIMALS = 6


@dataclass(frozen=True)
class SplitUnit:
    """One indivisible unit of data offered to the splitter."""

    segment_id: str
    payload_hash: str
    session_id: str
    source_file: str
    dataset_family: str
    stream: str
    row_count: int
    duration_s: float


@dataclass
class SplitAssignment:
    """Where one unit ended up, and why."""

    segment_id: str
    group_key: str
    split: str
    payload_hash: str
    session_id: str
    source_file: str
    dataset_family: str
    stream: str
    row_count: int
    duration_s: float


@dataclass
class SplitSummary:
    """Aggregate view of a generated split."""

    seed: str
    ratios: dict[str, float]
    group_count: int
    segment_count: int
    counts: dict[str, int] = field(default_factory=dict)
    rows: dict[str, int] = field(default_factory=dict)
    duration_s: dict[str, float] = field(default_factory=dict)
    grouping_rule: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class ContentAudit:
    """Result of checking the split for duplicates the payload hash cannot see."""

    segments_fingerprinted: int
    distinct_content_groups: int
    duplicate_content_groups: int
    violations: list[str] = field(default_factory=list)


def _group_key(payload_hashes: set[str]) -> str:
    """A stable key for a set of payloads that must stay together."""
    joined = "|".join(sorted(payload_hashes))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def build_groups(units: list[SplitUnit]) -> dict[str, list[SplitUnit]]:
    """Merge units into indivisible groups.

    Units sharing a payload hash must stay together. Units sharing a session id
    are also merged, because the same drive appearing under two session labels
    would otherwise be separable. The result is the transitive closure over
    both relations, so a chain of shared payloads/sessions collapses into one
    group.
    """
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for unit in units:
        payload_node = f"payload:{unit.payload_hash}"
        session_node = f"session:{unit.dataset_family}:{unit.session_id}"
        find(payload_node)
        find(session_node)
        union(payload_node, session_node)

    clusters: dict[str, list[SplitUnit]] = defaultdict(list)
    for unit in units:
        root = find(f"payload:{unit.payload_hash}")
        clusters[root].append(unit)

    # Re-key each cluster by the content it contains, so the key is a property
    # of the data rather than of iteration order.
    grouped: dict[str, list[SplitUnit]] = {}
    for members in clusters.values():
        key = _group_key({member.payload_hash for member in members})
        grouped.setdefault(key, []).extend(members)
    return grouped


def _score(group_key: str, seed: str) -> float:
    """Deterministic value in [0, 1) derived from the group key and seed."""
    digest = hashlib.sha256(f"{seed}:{group_key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _pack_groups(
    groups: dict[str, list[SplitUnit]],
    names: list[str],
    normalized: dict[str, float],
    seed: str,
) -> dict[str, str]:
    """Assign whole groups to splits, balancing by row count.

    Pure hash assignment balances *group counts*, which is useless here: group
    sizes span several orders of magnitude, so hashing 115 groups into 70/15/15
    left the test split with well under 1% of the rows. Instead groups are
    walked largest-first and each is placed in whichever split is furthest below
    its target share of rows.

    Determinism is preserved: the walk order is a stable sort on
    ``(-row_count, hash(seed, group_key))``, and ties in the deficit comparison
    break on the split name. Nothing depends on dict iteration order or wall
    time. The seed still shuffles which same-sized groups land where.
    """
    sizes = {key: sum(unit.row_count for unit in members) for key, members in groups.items()}
    total_rows = sum(sizes.values())
    if total_rows == 0:
        # Degenerate input: fall back to balancing group counts instead.
        sizes = dict.fromkeys(groups, 1)
        total_rows = len(groups)

    targets = {name: normalized[name] * total_rows for name in names}
    assigned: dict[str, float] = dict.fromkeys(names, 0.0)
    result: dict[str, str] = {}

    order = sorted(groups, key=lambda key: (-sizes[key], _score(key, seed), key))
    for group_key in order:
        # Largest remaining deficit wins; a split with a zero target never does.
        best = min(
            names,
            key=lambda name: (-(targets[name] - assigned[name]), name),
        )
        result[group_key] = best
        assigned[best] += sizes[group_key]
    return result


def assign_splits(
    units: list[SplitUnit],
    *,
    ratios: dict[str, float] | None = None,
    seed: str = DEFAULT_SEED,
) -> tuple[list[SplitAssignment], SplitSummary]:
    """Assign every unit to exactly one split, grouped and deterministically."""
    ratios = dict(ratios or DEFAULT_RATIOS)
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")
    if any(value < 0 for value in ratios.values()):
        raise ValueError("split ratios must be non-negative")
    normalized = {name: value / total for name, value in ratios.items()}

    groups = build_groups(units)

    names = sorted(normalized)
    placement = _pack_groups(groups, names, normalized, seed)

    assignments: list[SplitAssignment] = []
    for group_key, members in sorted(groups.items()):
        split = placement[group_key]
        for unit in members:
            assignments.append(
                SplitAssignment(
                    segment_id=unit.segment_id,
                    group_key=group_key,
                    split=split,
                    payload_hash=unit.payload_hash,
                    session_id=unit.session_id,
                    source_file=unit.source_file,
                    dataset_family=unit.dataset_family,
                    stream=unit.stream,
                    row_count=unit.row_count,
                    duration_s=unit.duration_s,
                )
            )

    summary = SplitSummary(
        seed=seed,
        ratios=normalized,
        group_count=len(groups),
        segment_count=len(assignments),
        grouping_rule=(
            "transitive closure over shared payload_hash and shared "
            "(dataset_family, session_id); whole groups packed largest-first into "
            "the split with the largest row-count deficit, ordered by "
            "(-rows, sha256(seed:group_key), key) for determinism"
        ),
    )
    for assignment in assignments:
        summary.counts[assignment.split] = summary.counts.get(assignment.split, 0) + 1
        summary.rows[assignment.split] = (
            summary.rows.get(assignment.split, 0) + assignment.row_count
        )
        summary.duration_s[assignment.split] = round(
            summary.duration_s.get(assignment.split, 0.0) + assignment.duration_s, 3
        )
    return assignments, summary


def verify_no_leakage(assignments: list[SplitAssignment]) -> list[str]:
    """Return a list of leakage violations; empty means the split is sound.

    Checks that no payload hash, and no (family, session), appears in more than
    one split, and that no segment was assigned twice.
    """
    violations: list[str] = []

    by_payload: dict[str, set[str]] = defaultdict(set)
    by_session: dict[tuple[str, str], set[str]] = defaultdict(set)
    seen_segments: dict[str, str] = {}

    for assignment in assignments:
        by_payload[assignment.payload_hash].add(assignment.split)
        by_session[(assignment.dataset_family, assignment.session_id)].add(assignment.split)
        if assignment.segment_id in seen_segments:
            if seen_segments[assignment.segment_id] != assignment.split:
                violations.append(f"segment {assignment.segment_id} assigned to multiple splits")
        else:
            seen_segments[assignment.segment_id] = assignment.split

    for payload, splits in by_payload.items():
        if len(splits) > 1:
            violations.append(f"payload {payload[:12]} spans splits {sorted(splits)}")
    for (family, session), splits in by_session.items():
        if len(splits) > 1:
            violations.append(f"session {family}/{session} spans splits {sorted(splits)}")
    return violations


def content_fingerprint(frame: pd.DataFrame) -> str:
    """A hash of a segment's canonical *sensor* values, ignoring formatting.

    The payload hash catches files whose data rows are byte-identical. It does
    not catch files that hold the same measurements written differently —
    ``52.4`` against ``52.400``, say — because those differ as bytes while being
    the same recording. This fingerprint is taken after parsing, from the
    numeric sensor columns only, so such a pair collides as it should.

    Identity, label, and quality columns are deliberately excluded: two copies
    of one recording can legitimately carry different labels (one may have been
    synchronized to a vehicle stream and the other not), and that difference
    must not hide the fact that the sensor data is the same.
    """
    columns = sorted(
        name
        for name in frame.columns
        if frame[name].dtype.kind == "f" and name.startswith(_SENSOR_PREFIXES)
    )
    digest = hashlib.sha256()
    digest.update(f"{len(frame)}|{'|'.join(columns)}".encode())
    for name in columns:
        values = np.round(frame[name].to_numpy(dtype="float64"), _FINGERPRINT_DECIMALS)
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def audit_content_leakage(
    assignments: list[SplitAssignment], fingerprints: dict[str, str]
) -> ContentAudit:
    """Check the split at *content* level, beyond byte-identical payloads.

    ``fingerprints`` maps segment id to :func:`content_fingerprint`. Segments
    with no fingerprint are ignored rather than assumed distinct.
    """
    split_of = {assignment.segment_id: assignment.split for assignment in assignments}
    by_content: dict[str, list[str]] = defaultdict(list)
    for segment_id, fingerprint in fingerprints.items():
        if segment_id in split_of:
            by_content[fingerprint].append(segment_id)

    duplicate_groups = {key: members for key, members in by_content.items() if len(members) > 1}
    violations = [
        f"content group {key[:12]} spans splits {sorted({split_of[member] for member in members})}"
        for key, members in sorted(duplicate_groups.items())
        if len({split_of[member] for member in members}) > 1
    ]
    return ContentAudit(
        segments_fingerprinted=sum(len(members) for members in by_content.values()),
        distinct_content_groups=len(by_content),
        duplicate_content_groups=len(duplicate_groups),
        violations=violations,
    )


def splits_to_json(assignments: list[SplitAssignment], summary: SplitSummary) -> str:
    payload = {
        "summary": asdict(summary),
        "splits": {
            name: sorted(a.segment_id for a in assignments if a.split == name)
            for name in sorted(summary.counts)
        },
    }
    return json.dumps(payload, indent=2)
