"""Typed 3-vectors that carry their frame.

The failure this prevents is mundane and expensive: a bare ``[x, y, z]`` passed
into a function expecting vehicle-frame data when it actually holds phone-frame
data produces plausible numbers and a wrong answer. Attaching the frame makes
that a raised error instead.

This is deliberately *not* a units framework. There is no dimensional analysis,
no unit arithmetic, and no attempt to track m/s² through a multiplication —
Phase 2 already normalizes units on ingest and records them in
``canonical_schema.json``. What is tracked is the frame, the timestamp and the
provenance, because those are the three things that were actually ambiguous in
the dataset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from idr.frames.conventions import MIN_VECTOR_NORM, Frame


@dataclass(frozen=True)
class Vector3:
    """A 3-vector tagged with the frame its components are expressed in."""

    x: float
    y: float
    z: float
    frame: Frame = Frame.UNKNOWN

    @classmethod
    def from_array(cls, values: np.ndarray | list[float], frame: Frame = Frame.UNKNOWN) -> Vector3:
        array = np.asarray(values, dtype="float64").reshape(-1)
        if array.size != 3:
            raise ValueError(f"a Vector3 needs exactly 3 components, got {array.size}")
        return cls(float(array[0]), float(array[1]), float(array[2]), frame)

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype="float64")

    @property
    def norm(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    @property
    def is_finite(self) -> bool:
        return all(math.isfinite(value) for value in (self.x, self.y, self.z))

    def normalized(self) -> Vector3:
        """Unit vector in the same direction and frame.

        Raises when the vector is too short to have a direction: normalizing
        noise would otherwise manufacture a confident-looking unit vector out
        of nothing, which is exactly the kind of fabricated certainty this
        phase is supposed to avoid.
        """
        magnitude = self.norm
        if not math.isfinite(magnitude) or magnitude < MIN_VECTOR_NORM:
            raise ValueError(
                f"cannot normalize a vector of norm {magnitude!r}: it has no reliable direction"
            )
        return Vector3(self.x / magnitude, self.y / magnitude, self.z / magnitude, self.frame)

    def dot(self, other: Vector3) -> float:
        self._require_same_frame(other, "dot")
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3) -> Vector3:
        self._require_same_frame(other, "cross")
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
            self.frame,
        )

    def angle_to(self, other: Vector3) -> float:
        """Angle in radians between two directions in the same frame.

        Uses ``atan2(‖a×b‖, a·b)`` rather than ``acos`` because ``acos`` loses
        most of its precision for nearly-parallel vectors — the case that
        matters most when checking whether an alignment is consistent.
        """
        self._require_same_frame(other, "angle_to")
        cross = self.cross(other)
        return math.atan2(cross.norm, self.dot(other))

    def scaled(self, factor: float) -> Vector3:
        return Vector3(self.x * factor, self.y * factor, self.z * factor, self.frame)

    def with_frame(self, frame: Frame) -> Vector3:
        """Relabel the frame **without** transforming the components.

        Only correct when the components were already expressed in ``frame``
        and the tag was missing or wrong. To actually change frames, apply a
        rotation.
        """
        return replace(self, frame=frame)

    def _require_same_frame(self, other: Vector3, operation: str) -> None:
        if self.frame != other.frame:
            raise ValueError(
                f"cannot {operation} a {self.frame} vector with a {other.frame} vector; "
                "rotate one into the other's frame first"
            )

    def __repr__(self) -> str:
        return f"Vector3({self.x:.6g}, {self.y:.6g}, {self.z:.6g}, frame={self.frame})"


@dataclass(frozen=True)
class TimestampedVector3:
    """A :class:`Vector3` with the timing and provenance Phase 2 preserved.

    ``analysis_time_s`` is the clock Phase 2 chose as continuous for this
    stream and is the one algorithms should difference. ``source_time_s`` is
    carried verbatim for provenance — it may reset mid-recording, which is
    precisely why it is not the analysis clock.
    """

    value: Vector3
    analysis_time_s: float
    source_time_s: float | None = None
    source: str = ""
    quality_flags: int = 0

    @property
    def frame(self) -> Frame:
        return self.value.frame

    def to_array(self) -> np.ndarray:
        return self.value.to_array()


@dataclass
class VectorSeries:
    """A time-ordered block of same-frame vectors, held column-wise.

    Algorithms in this package work on whole segments, so the array form is the
    working representation and :class:`Vector3` is the interface at the edges.
    ``samples`` is ``(n, 3)``; ``analysis_time_s`` is ``(n,)``.
    """

    samples: np.ndarray
    analysis_time_s: np.ndarray
    frame: Frame = Frame.UNKNOWN
    source_time_s: np.ndarray | None = None
    quality_flags: np.ndarray | None = None
    source: str = ""
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype="float64")
        if self.samples.ndim != 2 or self.samples.shape[1] != 3:
            raise ValueError(f"samples must have shape (n, 3), got {self.samples.shape}")
        self.analysis_time_s = np.asarray(self.analysis_time_s, dtype="float64").reshape(-1)
        if self.analysis_time_s.size != self.samples.shape[0]:
            raise ValueError(
                f"{self.analysis_time_s.size} timestamps for {self.samples.shape[0]} samples"
            )

    def __len__(self) -> int:
        return int(self.samples.shape[0])

    @property
    def norms(self) -> np.ndarray:
        return np.linalg.norm(self.samples, axis=1)

    @property
    def finite_mask(self) -> np.ndarray:
        return np.isfinite(self.samples).all(axis=1) & np.isfinite(self.analysis_time_s)

    def at(self, index: int) -> TimestampedVector3:
        row = self.samples[index]
        return TimestampedVector3(
            value=Vector3(float(row[0]), float(row[1]), float(row[2]), self.frame),
            analysis_time_s=float(self.analysis_time_s[index]),
            source_time_s=(
                float(self.source_time_s[index]) if self.source_time_s is not None else None
            ),
            source=self.source,
            quality_flags=(int(self.quality_flags[index]) if self.quality_flags is not None else 0),
        )

    def subset(self, mask: np.ndarray) -> VectorSeries:
        """A new series holding only the selected rows, metadata preserved."""
        mask = np.asarray(mask, dtype=bool)
        return VectorSeries(
            samples=self.samples[mask],
            analysis_time_s=self.analysis_time_s[mask],
            frame=self.frame,
            source_time_s=None if self.source_time_s is None else self.source_time_s[mask],
            quality_flags=None if self.quality_flags is None else self.quality_flags[mask],
            source=self.source,
            notes=list(self.notes),
        )

    def mean_vector(self) -> Vector3:
        """Component-wise mean over finite samples, tagged with this frame."""
        finite = self.samples[self.finite_mask]
        if finite.size == 0:
            raise ValueError("no finite samples to average")
        return Vector3.from_array(finite.mean(axis=0), self.frame)
