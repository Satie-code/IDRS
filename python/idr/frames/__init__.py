"""Phase 3: sensor frames, orientation and phone→vehicle alignment.

The public surface Phase 4 should build on. Everything here operates on Phase 2
canonical data and produces an estimate *plus* a confidence *plus* the
individual quality metrics behind it — never a bare number.

The chain this package implements::

    canonical sensor data           idr.frames.sources
              ↓
        sensor frame                idr.frames.vectors, .conventions
              ↓
      orientation estimate          idr.frames.orientation (+ .gravity, .magnetometer)
              ↓
       phone → vehicle              idr.frames.alignment, .calibration
              ↓
      vehicle-frame motion          AlignmentEstimate.to_vehicle_frame

What this package deliberately does **not** do: integrate anything into
velocity or position, maintain a filter state or covariance, fuse GNSS, or run
a model. Those begin in Phase 4. The interfaces here are the inputs that phase
consumes.
"""

from __future__ import annotations

from idr.frames.alignment import (
    AlignmentEstimate,
    AlignmentQuality,
    CalibrationStatus,
    ForwardDirectionEstimate,
    check_alignment,
    estimate_alignment,
    estimate_forward_direction,
)
from idr.frames.calibration import CalibrationProgress, CalibrationSession
from idr.frames.conventions import Frame, RotationConvention
from idr.frames.gravity import GravityEstimate, GravityMethod, estimate_gravity
from idr.frames.magnetometer import MagnetometerQuality, assess_magnetometer
from idr.frames.orientation import (
    AxisSupport,
    OrientationEstimate,
    OrientationTrack,
    estimate_orientation,
)
from idr.frames.rotations import FrameRotation, is_rotation_matrix
from idr.frames.sources import SampleTiming, SegmentData, load_segment
from idr.frames.vectors import TimestampedVector3, Vector3, VectorSeries

__all__ = [
    "AlignmentEstimate",
    "AlignmentQuality",
    "AxisSupport",
    "CalibrationProgress",
    "CalibrationSession",
    "CalibrationStatus",
    "ForwardDirectionEstimate",
    "Frame",
    "FrameRotation",
    "GravityEstimate",
    "GravityMethod",
    "MagnetometerQuality",
    "OrientationEstimate",
    "OrientationTrack",
    "RotationConvention",
    "SampleTiming",
    "SegmentData",
    "TimestampedVector3",
    "Vector3",
    "VectorSeries",
    "assess_magnetometer",
    "check_alignment",
    "estimate_alignment",
    "estimate_forward_direction",
    "estimate_gravity",
    "estimate_orientation",
    "is_rotation_matrix",
    "load_segment",
]
