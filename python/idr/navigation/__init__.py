"""Phase 4: strapdown inertial mechanization and the baseline dead-reckoning engine.

The first phase that integrates anything. It turns calibrated smartphone IMU
measurements into a continuously propagated attitude, velocity and position:

.. code-block:: text

    canonical smartphone IMU        idr.frames.sources
              ↓
    orientation estimate            idr.frames.orientation      (Phase 3)
              ↓
    phone → vehicle alignment       idr.frames.alignment        (Phase 3)
              ↓
    vehicle-frame specific force    idr.navigation.mechanization
              ↓
    navigation-frame acceleration   + idr.navigation.gravity
              ↓
    velocity, position              idr.navigation.integration

**This is a baseline, not the navigation system.** It has no filter, no fusion,
no learned correction and no aiding of any kind after initialization. An
unaided inertial solution on consumer MEMS sensors drifts quickly, and the
purpose of building it carefully is to have a trustworthy number for Phase 5 to
improve on — a comparison that is only worth anything if the baseline was not
quietly helped.

What Phase 4 deliberately does not contain: EKF, UKF, factor graph, particle
filter, GNSS/INS fusion, non-holonomic constraints, zero-velocity *corrections*
(detection only), map matching, any neural network, Android, ONNX, production
C++, or a GNSS-outage simulator.
"""

from __future__ import annotations

from idr.navigation.bias import (
    AccelerometerBias,
    GyroscopeBias,
    ImuBias,
    estimate_bias_from_rest,
)
from idr.navigation.geodesy import TangentPlane, origin_from_fixes
from idr.navigation.gravity import (
    ConstantGravity,
    GravityModel,
    WGS84Gravity,
    linear_acceleration_nav,
)
from idr.navigation.integration import (
    InvalidStepAction,
    InvalidTimeStepError,
    StepVerdict,
    TimeStepPolicy,
    trapezoidal_step,
)
from idr.navigation.mechanization import (
    AlignmentPolicy,
    AttitudeMode,
    MechanizationConfig,
    MechanizationError,
    StrapdownMechanization,
    propagate_attitude,
)
from idr.navigation.reference_alignment import (
    ErrorCurves,
    LagEstimate,
    ReferenceComparison,
    ReferenceTrack,
    compare,
    estimate_residual_lag,
)
from idr.navigation.state import (
    InitialState,
    NavigationFlag,
    NavigationState,
)
from idr.navigation.stationarity import StationarityResult, detect_stationarity
from idr.navigation.trajectory import Trajectory, mechanize_series

__all__ = [
    "AccelerometerBias",
    "AlignmentPolicy",
    "AttitudeMode",
    "ConstantGravity",
    "ErrorCurves",
    "GravityModel",
    "GyroscopeBias",
    "ImuBias",
    "InitialState",
    "InvalidStepAction",
    "InvalidTimeStepError",
    "LagEstimate",
    "MechanizationConfig",
    "MechanizationError",
    "NavigationFlag",
    "NavigationState",
    "ReferenceComparison",
    "ReferenceTrack",
    "StationarityResult",
    "StepVerdict",
    "StrapdownMechanization",
    "TangentPlane",
    "TimeStepPolicy",
    "Trajectory",
    "WGS84Gravity",
    "compare",
    "detect_stationarity",
    "estimate_bias_from_rest",
    "estimate_residual_lag",
    "linear_acceleration_nav",
    "mechanize_series",
    "origin_from_fixes",
    "propagate_attitude",
    "trapezoidal_step",
]
