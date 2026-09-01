"""Timing policy, duplicate timestamps, gaps, and refusing to invent data.

Phase 2 retains 147,946 duplicate timestamps and Phase 3 measured segment rates
from 2 Hz to 1000 Hz, so irregular and invalid steps are the *normal* case in
this dataset rather than an edge case. These tests pin what happens at each one,
because "the mechanization coped" is not a specification.

The recurring assertion is that nothing is silently repaired. A bad sample gets
a flag and a held state; it does not get replaced with something plausible.
"""

from __future__ import annotations

import numpy as np
import pytest

from idr.frames import quaternion as quat
from idr.frames.conventions import Frame
from idr.frames.vectors import VectorSeries
from idr.navigation import synthetic as syn
from idr.navigation.integration import (
    InvalidStepAction,
    InvalidTimeStepError,
    StepVerdict,
    TimeStepPolicy,
    integrate_series,
    median_interval,
    trapezoidal_step,
)
from idr.navigation.mechanization import (
    MAX_PLAUSIBLE_SPECIFIC_FORCE_MPS2,
    AlignmentPolicy,
    AttitudeMode,
    MechanizationConfig,
    MechanizationError,
    StrapdownMechanization,
    TrajectoryBreakError,
)
from idr.navigation.state import NavigationFlag
from idr.navigation.trajectory import mechanize_series


def run(drive, **config_kwargs):
    return mechanize_series(
        initial=drive.initial_state(),
        specific_force=drive.specific_force_phone,
        angular_rate=drive.angular_rate_phone,
        orientations=drive.orientation,
        config=MechanizationConfig(
            alignment_policy=AlignmentPolicy.PHONE_FRAME_DIAGNOSTIC, **config_kwargs
        ),
    )


# --- Test 7: irregular sampling -------------------------------------------


def test_irregular_sampling_matches_the_closed_form():
    """Non-uniform Δt must not degrade an exactly-integrable profile."""
    times = syn.irregular_times(duration_s=20.0, rate_hz=10.0, jitter_s=0.03)
    drive = syn.analytic_trajectory(times=times, acceleration_nav_mps2=[0.8, 0.3, 0.0])
    trajectory = run(drive)
    assert np.abs(trajectory.velocity_mps - drive.velocity_nav_mps).max() < 1e-12
    assert np.abs(trajectory.position_m - drive.position_nav_m).max() < 1e-9


def test_a_fixed_dt_would_fail_the_irregular_case():
    """The irregular test is only meaningful if a naive 0.1 s step would fail it.

    Without this, a mechanization that hardcoded ``dt = 0.1`` could pass the
    test above by coincidence on nearly-uniform timestamps.
    """
    times = syn.irregular_times(duration_s=20.0, rate_hz=10.0, jitter_s=0.03)
    drive = syn.analytic_trajectory(times=times, acceleration_nav_mps2=[0.8, 0.3, 0.0])
    naive = integrate_series(
        drive.acceleration_nav_mps2, np.arange(len(drive)) * 0.1, drive.velocity_nav_mps[0]
    )
    assert np.abs(naive - drive.velocity_nav_mps).max() > 1e-3


def test_extreme_rates_are_both_handled():
    """2 Hz and 1000 Hz are both in this dataset; both must integrate exactly."""
    for rate_hz in (2.0, 1000.0):
        drive = syn.constant_acceleration(acceleration_mps2=1.5, duration_s=4.0, rate_hz=rate_hz)
        trajectory = run(drive)
        assert np.abs(trajectory.velocity_mps - drive.velocity_nav_mps).max() < 1e-11
        assert np.abs(trajectory.position_m - drive.position_nav_m).max() < 1e-9


def test_irregular_steps_are_flagged_against_the_segment_median():
    """A step is irregular relative to its own segment, not in absolute terms.

    The stretched step here is 0.21 s: twenty-one times the 0.01 s median and so
    clearly irregular, yet still under the 0.5 s integration limit — so it is
    flagged *and* integrated, which is the distinction the flag exists to make.
    """
    drive = syn.constant_acceleration(duration_s=5.0, rate_hz=100.0)
    stamps = drive.analysis_time_s.copy()
    stamps[250:] += 0.2
    stretched = syn.retimed(drive, stamps, "one stretched interval")
    trajectory = run(stretched, median_dt_s=median_interval(stamps))
    assert trajectory.counters.irregular == 1
    assert trajectory.counters.skipped == 0
    assert int(np.count_nonzero(trajectory.flags & int(NavigationFlag.IRREGULAR_DT))) == 1


def test_median_interval_ignores_duplicates():
    """Duplicate timestamps must not drag the median toward zero."""
    times = np.array([0.0, 0.1, 0.1, 0.2, 0.3, 0.3, 0.4])
    assert median_interval(times) == pytest.approx(0.1)
    assert median_interval(np.array([1.0])) is None


# --- Test 8: duplicate timestamps ------------------------------------------


def test_duplicate_timestamps_are_skipped_deterministically():
    """Δt = 0 is not integrated across, is flagged, and is counted."""
    drive = syn.with_duplicate_timestamps(
        syn.constant_acceleration(acceleration_mps2=1.0, duration_s=20.0), count=5
    )
    trajectory = run(drive)
    assert trajectory.counters.skipped_non_positive == 5
    duplicates = int(np.count_nonzero(trajectory.flags & int(NavigationFlag.DUPLICATE_TIMESTAMP)))
    assert duplicates == 5
    # The state is held across a duplicate rather than advanced by zero time.
    indices = np.flatnonzero(trajectory.flags & int(NavigationFlag.DUPLICATE_TIMESTAMP))
    for index in indices:
        assert trajectory.position_m[index] == pytest.approx(trajectory.position_m[index - 1])
        assert trajectory.velocity_mps[index] == pytest.approx(trajectory.velocity_mps[index - 1])


def test_duplicate_handling_is_reproducible():
    """The same input twice gives byte-identical output."""
    drive = syn.with_duplicate_timestamps(syn.constant_acceleration(duration_s=10.0))
    first, second = run(drive), run(drive)
    assert first.position_m == pytest.approx(second.position_m, nan_ok=True)
    assert first.flags.tolist() == second.flags.tolist()


def test_duplicates_do_not_corrupt_the_trapezoid():
    """After a held step the accumulator restarts rather than averaging across it."""
    drive = syn.with_duplicate_timestamps(
        syn.constant_acceleration(acceleration_mps2=1.0, duration_s=20.0), count=3
    )
    trajectory = run(drive)
    # Constant acceleration means even a restarted trapezoid is exact, so the
    # only error that could appear here is from bridging the zero interval.
    finite = np.isfinite(trajectory.velocity_mps).all(axis=1)
    assert np.abs(trajectory.velocity_mps[finite] - drive.velocity_nav_mps[finite]).max() < 1e-11


# --- Test 9: large gaps ----------------------------------------------------


def test_large_gap_is_not_integrated_across_by_default():
    """SKIP holds the state and flags the sample."""
    drive = syn.with_time_gap(
        syn.constant_acceleration(duration_s=20.0), at_fraction=0.5, gap_s=5.0
    )
    trajectory = run(drive)
    assert trajectory.counters.skipped_too_large == 1
    flagged = np.flatnonzero(trajectory.flags & int(NavigationFlag.LARGE_TIME_GAP))
    assert flagged.size == 1
    index = int(flagged[0])
    assert trajectory.position_m[index] == pytest.approx(trajectory.position_m[index - 1])


def test_large_gap_can_end_the_trajectory():
    """SEGMENT truncates rather than producing one continuous-looking solution."""
    drive = syn.with_time_gap(
        syn.constant_acceleration(duration_s=20.0), at_fraction=0.5, gap_s=5.0
    )
    trajectory = run(drive, time_step=TimeStepPolicy(action=InvalidStepAction.SEGMENT))
    assert len(trajectory) < len(drive)
    assert any("ended at sample" in note for note in trajectory.notes)


def test_large_gap_can_raise():
    """FAIL gives the caller an exception instead of a number."""
    drive = syn.with_time_gap(
        syn.constant_acceleration(duration_s=20.0), at_fraction=0.5, gap_s=5.0
    )
    with pytest.raises(InvalidTimeStepError) as raised:
        run(drive, time_step=TimeStepPolicy(action=InvalidStepAction.FAIL))
    assert raised.value.verdict is StepVerdict.TOO_LARGE


def test_gap_policy_threshold_is_configurable():
    """A caller who genuinely wants a 6 s step can have one, explicitly."""
    drive = syn.with_time_gap(
        syn.constant_acceleration(duration_s=20.0), at_fraction=0.5, gap_s=5.0
    )
    trajectory = run(drive, time_step=TimeStepPolicy(max_step_s=10.0))
    assert trajectory.counters.skipped_too_large == 0


# --- the policy itself -----------------------------------------------------


def test_time_step_policy_classifies():
    policy = TimeStepPolicy(max_step_s=0.5)
    assert policy.classify(0.1) is StepVerdict.VALID
    assert policy.classify(0.0) is StepVerdict.NON_POSITIVE
    assert policy.classify(-0.1) is StepVerdict.NON_POSITIVE
    assert policy.classify(0.6) is StepVerdict.TOO_LARGE
    assert policy.classify(float("nan")) is StepVerdict.NOT_FINITE
    assert policy.classify(float("inf")) is StepVerdict.NOT_FINITE


def test_time_step_policy_rejects_nonsense_configuration():
    with pytest.raises(ValueError, match="max_step_s"):
        TimeStepPolicy(max_step_s=0.0)
    with pytest.raises(ValueError, match="irregular_factor"):
        TimeStepPolicy(irregular_factor=1.0)


def test_irregularity_is_relative_and_safe():
    policy = TimeStepPolicy(irregular_factor=5.0)
    assert policy.is_irregular(0.6, 0.1)
    assert policy.is_irregular(0.01, 0.1)
    assert not policy.is_irregular(0.2, 0.1)
    assert not policy.is_irregular(0.6, None)
    assert not policy.is_irregular(float("nan"), 0.1)


def test_trapezoidal_step_refuses_an_invalid_dt():
    with pytest.raises(ValueError, match="positive finite dt"):
        trapezoidal_step(np.zeros(3), np.zeros(3), np.zeros(3), 0.0)
    with pytest.raises(ValueError, match="positive finite dt"):
        trapezoidal_step(np.zeros(3), np.zeros(3), np.zeros(3), float("nan"))


def test_integrate_series_matches_the_loop():
    """The vectorized helper and the stepwise loop must agree."""
    times = np.array([0.0, 0.1, 0.25, 0.4, 0.7])
    rates = np.array(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
    )
    vectorized = integrate_series(rates, times)
    manual = np.zeros_like(rates)
    for index in range(1, len(times)):
        manual[index] = trapezoidal_step(
            manual[index - 1], rates[index - 1], rates[index], times[index] - times[index - 1]
        )
    assert vectorized == pytest.approx(manual)


def test_integrate_series_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="against"):
        integrate_series(np.zeros((3, 3)), np.zeros(4))


# --- sensor validity -------------------------------------------------------


def test_nan_specific_force_is_flagged_not_sanitized():
    """A NaN sample holds the state; it is never replaced with a plausible value."""
    drive = syn.corrupt_samples(
        syn.constant_acceleration(duration_s=10.0), [20, 21], value=float("nan")
    )
    trajectory = run(drive)
    assert trajectory.counters.invalid_sensor == 2
    flagged = np.flatnonzero(trajectory.flags & int(NavigationFlag.INVALID_SENSOR))
    assert flagged.tolist() == [20, 21]
    assert trajectory.position_m[20] == pytest.approx(trajectory.position_m[19])


def test_infinite_specific_force_is_flagged():
    drive = syn.corrupt_samples(
        syn.constant_acceleration(duration_s=10.0), [30], value=float("inf")
    )
    trajectory = run(drive)
    assert trajectory.counters.invalid_sensor == 1
    assert bool(trajectory.flags[30] & int(NavigationFlag.INVALID_SENSOR))


def test_implausible_specific_force_is_rejected():
    """A 20 g reading is a fault, not a manoeuvre, and must not be integrated."""
    drive = syn.corrupt_samples(
        syn.constant_acceleration(duration_s=10.0),
        [40],
        value=MAX_PLAUSIBLE_SPECIFIC_FORCE_MPS2 * 2.0,
    )
    trajectory = run(drive)
    assert bool(trajectory.flags[40] & int(NavigationFlag.INVALID_SENSOR))
    assert trajectory.velocity_mps[40] == pytest.approx(trajectory.velocity_mps[39])


def test_a_plausible_sample_is_not_rejected():
    """The plausibility bound must not clip a hard but real braking event."""
    drive = syn.analytic_trajectory(
        acceleration_nav_mps2=[-8.0, 0.0, 0.0],
        initial_velocity_nav_mps=[30.0, 0.0, 0.0],
        duration_s=3.0,
    )
    trajectory = run(drive)
    assert trajectory.counters.invalid_sensor == 0


# --- alignment policy ------------------------------------------------------


def test_missing_alignment_refuses_by_default():
    """No identity fallback, ever. The refusal names the reason."""
    drive = syn.constant_acceleration(duration_s=5.0)
    engine = StrapdownMechanization(MechanizationConfig())
    with pytest.raises(MechanizationError, match="substituting identity"):
        engine.initialize(drive.initial_state(), alignment=None)


def test_missing_alignment_can_run_as_a_diagnostic():
    """The diagnostic mode runs, flags every sample, and says it is not navigation grade."""
    drive = syn.constant_acceleration(duration_s=10.0)
    trajectory = run(drive)
    assert not trajectory.is_navigation_grade
    assert (trajectory.flags[1:] & int(NavigationFlag.ALIGNMENT_UNAVAILABLE)).all()
    assert any("phone-frame diagnostic" in note for note in trajectory.notes)


def test_step_before_initialize_raises():
    engine = StrapdownMechanization(MechanizationConfig())
    with pytest.raises(MechanizationError, match="initialize"):
        engine.step(timestamp=0.0, specific_force=np.zeros(3))
    with pytest.raises(MechanizationError, match="initialize"):
        _ = engine.state


def test_phase3_mode_requires_an_orientation_every_step():
    """Silence is not an acceptable input to the filtered attitude mode."""
    drive = syn.constant_acceleration(duration_s=5.0)
    with pytest.raises(MechanizationError, match="needs an orientation"):
        mechanize_series(
            initial=drive.initial_state(),
            specific_force=drive.specific_force_phone,
            angular_rate=drive.angular_rate_phone,
            orientations=None,
            config=MechanizationConfig(
                attitude_mode=AttitudeMode.PHASE3_FILTER,
                alignment_policy=AlignmentPolicy.PHONE_FRAME_DIAGNOSTIC,
            ),
        )


def test_gyro_mode_ignores_a_supplied_orientation():
    """The two attitude sources must not blend. §10 is enforced, not documented.

    A supplied orientation deliberately different from the propagated one must
    leave the result equal to pure propagation — if it moved at all, the two
    measurements would be counted twice.
    """
    drive = syn.pure_rotation(rate_rps=0.2, duration_s=5.0, rate_hz=20.0)
    misleading = np.tile(quat.from_euler(1.0, 0.5, -0.8), (len(drive), 1))
    config = MechanizationConfig(
        attitude_mode=AttitudeMode.GYRO_PROPAGATION,
        alignment_policy=AlignmentPolicy.PHONE_FRAME_DIAGNOSTIC,
    )
    with_orientation = mechanize_series(
        initial=drive.initial_state(),
        specific_force=drive.specific_force_phone,
        angular_rate=drive.angular_rate_phone,
        orientations=misleading,
        config=config,
    )
    without = mechanize_series(
        initial=drive.initial_state(),
        specific_force=drive.specific_force_phone,
        angular_rate=drive.angular_rate_phone,
        orientations=None,
        config=config,
    )
    assert with_orientation.orientation == pytest.approx(without.orientation)


def test_mechanize_rejects_a_non_phone_frame_series():
    """A bias is only defined in the sensor frame, so the input frame is checked."""
    drive = syn.constant_acceleration(duration_s=5.0)
    rotated = VectorSeries(
        samples=drive.specific_force_phone.samples,
        analysis_time_s=drive.analysis_time_s,
        frame=Frame.VEHICLE,
        source="wrong frame",
    )
    with pytest.raises(ValueError, match="phone-frame specific force"):
        mechanize_series(
            initial=drive.initial_state(),
            specific_force=rotated,
            config=MechanizationConfig(alignment_policy=AlignmentPolicy.PHONE_FRAME_DIAGNOSTIC),
        )


def test_mechanize_rejects_an_empty_series():
    drive = syn.constant_acceleration(duration_s=5.0)
    empty = VectorSeries(
        samples=np.zeros((0, 3)),
        analysis_time_s=np.zeros(0),
        frame=Frame.PHONE,
        source="empty",
    )
    with pytest.raises(ValueError, match="empty series"):
        mechanize_series(initial=drive.initial_state(), specific_force=empty)


def test_trajectory_break_carries_its_context():
    broke = TrajectoryBreakError(5.0, StepVerdict.TOO_LARGE, 42)
    assert broke.index == 42
    assert broke.verdict is StepVerdict.TOO_LARGE
    assert "42" in str(broke)


def test_a_mismatched_gyroscope_is_dropped_with_a_note():
    """Dropping it silently would freeze the attitude with nothing to explain why."""
    drive = syn.constant_acceleration(duration_s=10.0)
    wrong_length = VectorSeries(
        samples=np.zeros((5, 3)),
        analysis_time_s=np.arange(5.0),
        frame=Frame.PHONE,
        source="mismatched gyro",
    )
    trajectory = mechanize_series(
        initial=drive.initial_state(),
        specific_force=drive.specific_force_phone,
        angular_rate=wrong_length,
        orientations=drive.orientation,
        config=MechanizationConfig(alignment_policy=AlignmentPolicy.PHONE_FRAME_DIAGNOSTIC),
    )
    assert any("was not used" in note for note in trajectory.notes)
    # The run still completes on the accelerometer and the supplied orientation.
    assert len(trajectory) == len(drive)
