"""Small post-filters for uniformly sampled command trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class FilteredJointTrajectory:
    """Position and derivatives from one local-polynomial filter."""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


def filter_joint_trajectory_savgol(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    *,
    sample_dt_s: float,
    window_length_samples: int,
    polynomial_order: int,
    preserve_endpoints: bool = True,
) -> FilteredJointTrajectory:
    """Smooth position and derive ``dq/ddq`` from the same local polynomial.

    This is intentionally a simple non-causal post-filter, suitable when the full
    planned path is available before publication.  Restoring both endpoint states
    keeps queue splices exact.  Callers must re-check constraints after filtering.
    """

    q = np.asarray(position, dtype=np.float64)
    dq = np.asarray(velocity, dtype=np.float64)
    ddq = np.asarray(acceleration, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] < 3 or q.shape[1] < 1:
        raise ValueError("position must have shape [samples >= 3, dof >= 1]")
    if dq.shape != q.shape or ddq.shape != q.shape:
        raise ValueError("position, velocity, and acceleration shapes must match")
    if not all(np.all(np.isfinite(value)) for value in (q, dq, ddq)):
        raise ValueError("position, velocity, and acceleration must be finite")
    if not np.isfinite(sample_dt_s) or sample_dt_s <= 0.0:
        raise ValueError("sample_dt_s must be finite and positive")
    if window_length_samples < 3 or window_length_samples % 2 != 1:
        raise ValueError("window_length_samples must be an odd integer >= 3")
    if window_length_samples > len(q):
        raise ValueError("window_length_samples must not exceed the sample count")
    if polynomial_order < 2 or polynomial_order >= window_length_samples:
        raise ValueError(
            "polynomial_order must be >= 2 and below the window length"
        )

    filtered = [
        savgol_filter(
            q,
            window_length=window_length_samples,
            polyorder=polynomial_order,
            deriv=derivative,
            delta=sample_dt_s,
            axis=0,
            mode="interp",
        )
        for derivative in range(3)
    ]
    if preserve_endpoints:
        for output, source in zip(filtered, (q, dq, ddq), strict=True):
            output[[0, -1]] = source[[0, -1]]
    return FilteredJointTrajectory(
        *(np.asarray(value, dtype=np.float64) for value in filtered)
    )
