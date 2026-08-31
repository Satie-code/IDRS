from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from idr.pipeline.reader import read_canonical
from idr.pipeline.reference import LabelSource, build_velocity_labels
from idr.pipeline.sequences import WindowConfig, generate_windows
from idr.pipeline.splits import (
    SplitAssignment,
    SplitUnit,
    assign_splits,
    audit_content_leakage,
    build_groups,
    content_fingerprint,
    verify_no_leakage,
)
from idr.pipeline.sync import SyncStatus, analyze_pair, estimate_offset
from idr.pipeline.timeline import diagnose_and_segment


def _prepare(path: Path, stream: str):
    read = read_canonical(path, path.name, stream, dataset_family="f", session_id="T1")
    frame, _, _ = diagnose_and_segment(
        read.frame,
        source_file=path.name,
        session_id="T1",
        stream=stream,
        dataset_family="f",
        payload_hash=read.payload_hash,
    )
    return frame


# --- synchronization -------------------------------------------------------


def _pair(phone: pd.DataFrame, vehicle: pd.DataFrame):
    return analyze_pair(
        "pair",
        "T1",
        "f",
        phone={
            "file": "S.csv",
            "segment": "s0",
            "time": phone["time_of_day_s"].to_numpy(dtype="float64"),
            "speed": phone["gnss_speed_mps"].to_numpy(dtype="float64"),
        },
        vehicle={
            "file": "V.csv",
            "segment": "v0",
            "time": vehicle["time_of_day_s"].to_numpy(dtype="float64"),
            "speed": vehicle["ref_speed_mps"].to_numpy(dtype="float64"),
        },
    )


def test_recovers_a_known_offset() -> None:
    """A synthetic pair with a planted 12.0 s offset must be recovered."""
    rng = np.random.default_rng(0)
    vehicle_time = np.arange(0.0, 600.0, 0.1)
    # A varying speed profile gives the correlator something to lock onto.
    speed = 10.0 + 8.0 * np.sin(vehicle_time / 30.0) + rng.normal(0, 0.05, vehicle_time.size)
    offset = 12.0
    phone_time = vehicle_time + offset

    estimated, correlation, margin = estimate_offset(
        phone_time, speed, vehicle_time, speed, coarse_offset=0.0, half_width_s=30.0
    )
    assert estimated == pytest.approx(offset, abs=0.15)
    assert correlation is not None and correlation > 0.9
    assert margin is not None and margin > 0.05


def test_unresolved_when_overlap_is_too_short(smartphone_csv: Path, vehicle_csv: Path) -> None:
    """60 rows is 6 s of data — far below the 60 s minimum overlap."""
    result = _pair(_prepare(smartphone_csv, "smartphone"), _prepare(vehicle_csv, "vehicle"))
    assert result.status == SyncStatus.UNRESOLVED
    assert result.offset_confidence == "none"


def test_unresolved_pair_is_not_forced_to_an_offset(
    smartphone_csv: Path, vehicle_csv: Path
) -> None:
    result = _pair(_prepare(smartphone_csv, "smartphone"), _prepare(vehicle_csv, "vehicle"))
    assert "UNRESOLVED" in result.status
    assert "below" in result.note or "overlap" in result.note


def test_missing_clock_yields_unresolved(smartphone_csv: Path, vehicle_csv: Path) -> None:
    phone = _prepare(smartphone_csv, "smartphone")
    vehicle = _prepare(vehicle_csv, "vehicle").copy()
    vehicle["time_of_day_s"] = np.nan
    result = _pair(phone, vehicle)
    assert result.status == SyncStatus.UNRESOLVED
    assert "usable time-of-day clock" in result.note


def test_drift_is_detected_and_reported() -> None:
    vehicle_time = np.arange(0.0, 600.0, 0.1)
    speed = 10.0 + 5.0 * np.sin(vehicle_time / 20.0)
    # Phone clock runs 1% fast: a growing offset across the session.
    phone_time = vehicle_time * 1.01
    result = analyze_pair(
        "p",
        "s",
        "f",
        phone={"file": "S", "segment": "a", "time": phone_time, "speed": speed},
        vehicle={"file": "V", "segment": "b", "time": vehicle_time, "speed": speed},
    )
    assert result.drift_detected is True
    assert result.drift_estimate_s is not None and abs(result.drift_estimate_s) > 1.0


def test_sync_is_deterministic() -> None:
    vehicle_time = np.arange(0.0, 300.0, 0.1)
    speed = 10.0 + 5.0 * np.sin(vehicle_time / 15.0)
    phone: dict[str, object] = {
        "file": "S",
        "segment": "a",
        "time": vehicle_time + 3.0,
        "speed": speed,
    }
    vehicle: dict[str, object] = {
        "file": "V",
        "segment": "b",
        "time": vehicle_time,
        "speed": speed,
    }
    first = analyze_pair("p", "s", "f", phone, vehicle)
    second = analyze_pair("p", "s", "f", phone, vehicle)
    assert first.estimated_offset_s == second.estimated_offset_s
    assert first.status == second.status


# --- reference labels ------------------------------------------------------


def test_tier1_used_when_sync_confirmed(smartphone_csv: Path) -> None:
    phone = _prepare(smartphone_csv, "smartphone")
    times = phone["time_of_day_s"].to_numpy(dtype="float64")
    frame, coverage = build_velocity_labels(
        phone,
        segment_id="s",
        session_id="T1",
        source_file="S.csv",
        vehicle_time_s=times,
        vehicle_speed_mps=np.full(times.size, 7.5),
        sync_offset_s=0.0,
        sync_confirmed=True,
    )
    assert coverage.label_source == LabelSource.TIER_1_VEHICLE_SYNCED
    assert coverage.label_quality == "high"
    assert frame["ref_velocity_mps"].dropna().iloc[0] == pytest.approx(7.5)


def test_tier2_used_when_sync_is_clock_only(smartphone_csv: Path) -> None:
    phone = _prepare(smartphone_csv, "smartphone")
    times = phone["time_of_day_s"].to_numpy(dtype="float64")
    _, coverage = build_velocity_labels(
        phone,
        segment_id="s",
        session_id="T1",
        source_file="S.csv",
        vehicle_time_s=times,
        vehicle_speed_mps=np.full(times.size, 7.5),
        sync_offset_s=0.0,
        sync_confirmed=False,
    )
    assert coverage.label_source == LabelSource.TIER_2_VEHICLE_CLOCK
    assert coverage.label_quality == "medium"


def test_tier3_fallback_is_low_confidence(smartphone_csv: Path) -> None:
    from idr.pipeline.freshness import annotate_freshness

    phone, _ = annotate_freshness(
        _prepare(smartphone_csv, "smartphone"), source_file="S.csv", stream="smartphone"
    )
    frame, coverage = build_velocity_labels(
        phone, segment_id="s", session_id="T1", source_file="S.csv"
    )
    assert coverage.label_source == LabelSource.TIER_3_SMARTPHONE_GNSS
    assert coverage.label_quality == "low"
    from idr.pipeline.canonical import QualityFlag

    flags = frame["quality_flags"].to_numpy()
    assert bool(np.all(flags & QualityFlag.REFERENCE_LOW_CONFIDENCE.value))


def test_labels_not_extrapolated_beyond_reference_coverage(smartphone_csv: Path) -> None:
    """Rows far from any reference sample must stay unlabelled."""
    phone = _prepare(smartphone_csv, "smartphone")
    times = phone["time_of_day_s"].to_numpy(dtype="float64")
    # Reference covers only the first two samples.
    frame, coverage = build_velocity_labels(
        phone,
        segment_id="s",
        session_id="T1",
        source_file="S.csv",
        vehicle_time_s=times[:2],
        vehicle_speed_mps=np.array([7.0, 7.0]),
        sync_offset_s=0.0,
        sync_confirmed=True,
    )
    assert coverage.coverage_fraction < 1.0
    assert frame["ref_velocity_mps"].isna().any()


# --- splits ----------------------------------------------------------------


def _unit(segment: str, payload: str, session: str, family: str = "f") -> SplitUnit:
    return SplitUnit(
        segment_id=segment,
        payload_hash=payload,
        session_id=session,
        source_file=f"{segment}.csv",
        dataset_family=family,
        stream="smartphone",
        row_count=100,
        duration_s=10.0,
    )


def test_identical_payloads_never_cross_splits() -> None:
    units = [_unit(f"seg{i}", "PAYLOAD_A", f"S{i}") for i in range(6)]
    units += [_unit(f"other{i}", f"PAYLOAD_{i}", f"O{i}") for i in range(20)]
    assignments, _ = assign_splits(units)
    assert verify_no_leakage(assignments) == []
    duplicated = {a.split for a in assignments if a.payload_hash == "PAYLOAD_A"}
    assert len(duplicated) == 1


def test_same_session_stays_together() -> None:
    units = [
        _unit("a", "P1", "SHARED"),
        _unit("b", "P2", "SHARED"),
        _unit("c", "P3", "OTHER"),
    ]
    assignments, _ = assign_splits(units)
    shared = {a.split for a in assignments if a.session_id == "SHARED"}
    assert len(shared) == 1


def test_segments_from_one_file_share_a_split() -> None:
    """Multiple segments of one recording must not straddle a boundary."""
    units = [_unit(f"seg{i}", "SAME_PAYLOAD", "T1") for i in range(5)]
    units += [_unit(f"x{i}", f"P{i}", f"S{i}") for i in range(30)]
    assignments, _ = assign_splits(units)
    same = {a.split for a in assignments if a.payload_hash == "SAME_PAYLOAD"}
    assert len(same) == 1


def test_split_assignment_is_deterministic() -> None:
    units = [_unit(f"s{i}", f"P{i}", f"S{i}") for i in range(40)]
    first, _ = assign_splits(units, seed="fixed")
    second, _ = assign_splits(list(reversed(units)), seed="fixed")
    assert {a.segment_id: a.split for a in first} == {a.segment_id: a.split for a in second}


def test_different_seeds_can_produce_different_splits() -> None:
    units = [_unit(f"s{i}", f"P{i}", f"S{i}") for i in range(40)]
    a, _ = assign_splits(units, seed="alpha")
    b, _ = assign_splits(units, seed="beta")
    assert {x.segment_id: x.split for x in a} != {x.segment_id: x.split for x in b}


def test_every_unit_is_assigned_exactly_once() -> None:
    units = [_unit(f"s{i}", f"P{i}", f"S{i}") for i in range(25)]
    assignments, summary = assign_splits(units)
    assert len(assignments) == len(units)
    assert sum(summary.counts.values()) == len(units)


def test_transitive_grouping_merges_chains() -> None:
    """A shares a payload with B; B shares a session with C — all one group."""
    units = [
        _unit("a", "P_SHARED", "SESS_A"),
        _unit("b", "P_SHARED", "SESS_B"),
        _unit("c", "P_OTHER", "SESS_B"),
    ]
    groups = build_groups(units)
    assert len(groups) == 1


def test_leakage_verifier_catches_a_bad_split() -> None:
    from idr.pipeline.splits import SplitAssignment

    bad = [
        SplitAssignment("s1", "g1", "train", "P", "S", "f1", "f", "smartphone", 1, 1.0),
        SplitAssignment("s2", "g2", "test", "P", "S", "f2", "f", "smartphone", 1, 1.0),
    ]
    violations = verify_no_leakage(bad)
    assert violations
    assert any("payload" in v for v in violations)


def test_invalid_ratios_are_rejected() -> None:
    with pytest.raises(ValueError):
        assign_splits([_unit("a", "P", "S")], ratios={"train": 0.0, "test": 0.0})


# --- sequences -------------------------------------------------------------


def _window_frame(n: int = 200, rate: float = 10.0) -> pd.DataFrame:
    times = np.arange(n) / rate
    return pd.DataFrame(
        {
            "elapsed_s": times,
            "ref_velocity_mps": np.linspace(0.0, 20.0, n),
            "quality_flags": np.zeros(n, dtype="int32"),
            "label_source": "TIER_1_VEHICLE_SYNCED",
            "label_quality": "high",
        }
    )


def test_windows_are_causal_no_future_samples() -> None:
    """The core guarantee: nothing after the target may enter the window."""
    frame = _window_frame()
    config = WindowConfig(duration_seconds=2.0, stride_seconds=0.5)
    windows = list(generate_windows(frame, config, segment_id="s", session_id="T1"))
    assert windows
    times = frame["elapsed_s"].to_numpy()
    for window in windows:
        assert times[window.input_indices].max() <= window.target_time_s + 1e-9
        assert np.all(times[window.input_indices] <= window.target_time_s + 1e-9)


def test_window_spans_the_configured_duration() -> None:
    frame = _window_frame()
    config = WindowConfig(duration_seconds=2.0, stride_seconds=0.5)
    window = next(iter(generate_windows(frame, config, segment_id="s", session_id="T1")))
    times = frame["elapsed_s"].to_numpy()[window.input_indices]
    assert times.max() - times.min() == pytest.approx(2.0, abs=0.15)


def test_window_geometry_is_configurable() -> None:
    frame = _window_frame()
    short = list(generate_windows(frame, WindowConfig(1.0, 0.5), segment_id="s", session_id="T"))
    long = list(generate_windows(frame, WindowConfig(5.0, 0.5), segment_id="s", session_id="T"))
    assert short[0].sample_count < long[0].sample_count
    assert len(short) > len(long)


def test_target_is_the_last_sample_in_the_window() -> None:
    frame = _window_frame()
    for window in generate_windows(frame, WindowConfig(2.0, 1.0), segment_id="s", session_id="T"):
        assert window.target_index == window.input_indices[-1]


def test_windows_require_a_finite_label_when_configured() -> None:
    frame = _window_frame()
    frame.loc[frame.index[100:], "ref_velocity_mps"] = np.nan
    windows = list(generate_windows(frame, WindowConfig(1.0, 0.5), segment_id="s", session_id="T"))
    assert windows
    assert all(np.isfinite(w.target_value) for w in windows)


def test_low_coverage_windows_are_rejected() -> None:
    """A stretch with a hole should not yield a dense-looking window."""
    times = np.concatenate([np.arange(0, 2.0, 0.1), np.arange(10.0, 12.0, 0.1)])
    frame = pd.DataFrame(
        {
            "elapsed_s": times,
            "ref_velocity_mps": np.ones(times.size),
            "quality_flags": np.zeros(times.size, dtype="int32"),
        }
    )
    windows = list(
        generate_windows(
            frame, WindowConfig(5.0, 0.5, min_coverage=0.9), segment_id="s", session_id="T"
        )
    )
    for window in windows:
        assert window.coverage >= 0.9


def test_invalid_window_config_is_rejected() -> None:
    with pytest.raises(ValueError):
        WindowConfig(duration_seconds=0.0)
    with pytest.raises(ValueError):
        WindowConfig(stride_seconds=-1.0)
    with pytest.raises(ValueError):
        WindowConfig(min_coverage=2.0)


def test_empty_frame_yields_no_windows() -> None:
    empty = pd.DataFrame({"elapsed_s": [], "ref_velocity_mps": [], "quality_flags": []})
    assert list(generate_windows(empty, WindowConfig(), segment_id="s", session_id="T")) == []


def test_windows_carry_segment_and_split_identity() -> None:
    frame = _window_frame()
    windows = list(
        generate_windows(
            frame, WindowConfig(1.0, 1.0), segment_id="SEG", session_id="SESS", split="train"
        )
    )
    assert windows
    assert all(w.segment_id == "SEG" and w.session_id == "SESS" for w in windows)
    assert all(w.split == "train" for w in windows)


def test_split_is_balanced_by_rows_not_group_count() -> None:
    """Wildly uneven group sizes must still yield roughly the target row shares.

    Hash-bucketing groups balances group *counts*, which left the test split
    with under 1% of rows on the real dataset.
    """
    units = [
        SplitUnit(
            segment_id=f"big{i}",
            payload_hash=f"BIG{i}",
            session_id=f"BIG{i}",
            source_file=f"big{i}.csv",
            dataset_family="f",
            stream="smartphone",
            row_count=100_000,
            duration_s=1000.0,
        )
        for i in range(10)
    ]
    units += [
        SplitUnit(
            segment_id=f"small{i}",
            payload_hash=f"SMALL{i}",
            session_id=f"SMALL{i}",
            source_file=f"small{i}.csv",
            dataset_family="f",
            stream="smartphone",
            row_count=500,
            duration_s=5.0,
        )
        for i in range(100)
    ]
    _, summary = assign_splits(units, ratios={"train": 0.7, "validation": 0.15, "test": 0.15})
    total = sum(summary.rows.values())
    for name, target in (("train", 0.7), ("validation", 0.15), ("test", 0.15)):
        share = summary.rows[name] / total
        assert abs(share - target) < 0.10, f"{name} share {share:.3f} far from {target}"


def test_balanced_packing_is_still_deterministic() -> None:
    units = [
        SplitUnit(
            segment_id=f"s{i}",
            payload_hash=f"P{i}",
            session_id=f"S{i}",
            source_file=f"{i}.csv",
            dataset_family="f",
            stream="smartphone",
            row_count=(i * 137) % 5000 + 10,
            duration_s=1.0,
        )
        for i in range(60)
    ]
    first, _ = assign_splits(units, seed="fixed")
    second, _ = assign_splits(list(reversed(units)), seed="fixed")
    assert {a.segment_id: a.split for a in first} == {a.segment_id: a.split for a in second}


def test_balanced_packing_still_prevents_leakage() -> None:
    units = [
        SplitUnit(
            segment_id=f"seg{i}",
            payload_hash="SHARED" if i < 5 else f"P{i}",
            session_id=f"S{i}",
            source_file=f"{i}.csv",
            dataset_family="f",
            stream="smartphone",
            row_count=1000 * (i + 1),
            duration_s=10.0,
        )
        for i in range(30)
    ]
    assignments, _ = assign_splits(units)
    assert verify_no_leakage(assignments) == []
    assert len({a.split for a in assignments if a.payload_hash == "SHARED"}) == 1


# --- content-level leakage audit -------------------------------------------


def _content_frame(values: list[float], *, noise: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "accel_x_mps2": [v + noise for v in values],
            "gyro_z_rps": [v * 0.1 for v in values],
            "latitude_deg": [52.4 + v * 1e-5 for v in values],
            # Excluded from the fingerprint: identity, label and quality fields.
            "segment_id": ["seg"] * len(values),
            "ref_velocity_mps": [float("nan")] * len(values),
            "quality_flags": [0] * len(values),
        }
    )


def test_content_fingerprint_ignores_numeric_formatting() -> None:
    """Same measurements, different float text, must fingerprint the same."""
    values = [1.0, 2.5, 3.25]
    a = _content_frame(values)
    b = _content_frame(values)
    b["accel_x_mps2"] = b["accel_x_mps2"].map(lambda v: float(f"{v:.10f}"))
    assert content_fingerprint(a) == content_fingerprint(b)


def test_content_fingerprint_ignores_labels_and_quality() -> None:
    """Two copies of one recording may carry different labels; that must not hide them."""
    a = _content_frame([1.0, 2.0, 3.0])
    b = _content_frame([1.0, 2.0, 3.0])
    b["ref_velocity_mps"] = [4.0, 5.0, 6.0]
    b["quality_flags"] = [7, 7, 7]
    b["segment_id"] = ["other"] * 3
    assert content_fingerprint(a) == content_fingerprint(b)


def test_content_fingerprint_separates_different_measurements() -> None:
    assert content_fingerprint(_content_frame([1.0, 2.0, 3.0])) != content_fingerprint(
        _content_frame([1.0, 2.0, 3.0], noise=0.01)
    )


def _assignment(segment_id: str, split: str) -> SplitAssignment:
    return SplitAssignment(
        segment_id=segment_id,
        group_key="g",
        split=split,
        payload_hash="p",
        session_id="S1",
        source_file=f"{segment_id}.csv",
        dataset_family="f",
        stream="smartphone",
        row_count=10,
        duration_s=1.0,
    )


def test_content_audit_reports_duplicates_that_stay_within_one_split() -> None:
    assignments = [_assignment("a", "train"), _assignment("b", "train"), _assignment("c", "test")]
    audit = audit_content_leakage(assignments, {"a": "h1", "b": "h1", "c": "h2"})
    assert audit.segments_fingerprinted == 3
    assert audit.distinct_content_groups == 2
    assert audit.duplicate_content_groups == 1
    assert audit.violations == []


def test_content_audit_catches_a_duplicate_spanning_splits() -> None:
    """The failure the payload hash alone cannot see."""
    assignments = [_assignment("a", "train"), _assignment("b", "test")]
    audit = audit_content_leakage(assignments, {"a": "h1", "b": "h1"})
    assert len(audit.violations) == 1
    assert "spans splits" in audit.violations[0]
    assert "['test', 'train']" in audit.violations[0]


def test_content_audit_ignores_segments_that_were_not_assigned() -> None:
    """An unassigned fingerprint is skipped, not assumed to be its own group."""
    audit = audit_content_leakage([_assignment("a", "train")], {"a": "h1", "ghost": "h2"})
    assert audit.segments_fingerprinted == 1
    assert audit.distinct_content_groups == 1
