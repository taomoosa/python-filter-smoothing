"""Jerk-limited state-to-state interpolation for a sampled joint path."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RuckigInterpolation:
    """Uniformly sampled trajectory and its correspondence to input nodes."""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    node_sample_indices: np.ndarray
    segment_durations_s: np.ndarray
    calculation_wall_time_s: float

    @property
    def duration_s(self) -> float:
        return float(np.sum(self.segment_durations_s))


def _nodes(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] < 2 or result.shape[1] < 1:
        raise ValueError(f"{name} must have shape [nodes >= 2, dof >= 1]")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _limits(value: np.ndarray, dof: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (dof,):
        raise ValueError(f"{name} must contain one value per joint")
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return result


def interpolate_joint_path(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    *,
    sample_dt_s: float,
    minimum_segment_duration_s: float | np.ndarray,
    max_velocity: np.ndarray,
    max_acceleration: np.ndarray,
    max_jerk: np.ndarray,
    min_position: np.ndarray | None = None,
    max_position: np.ndarray | None = None,
) -> RuckigInterpolation:
    """Connect every input ``q/dq/ddq`` node with a local Ruckig trajectory.

    Ruckig's Community edition cannot calculate multiple intermediate waypoints
    locally.  This function therefore solves one state-to-state problem per input
    interval.  Every input state is reached exactly and acceleration is continuous,
    but the curve between two nodes is not guaranteed to equal the source path.
    Callers responsible for collision safety must validate the returned samples.
    """

    try:
        from ruckig import (
            DurationDiscretization,
            InputParameter,
            Result,
            Ruckig,
            Trajectory,
        )
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise RuntimeError("Ruckig is required for joint-path interpolation") from error

    q = _nodes(position, "position")
    dq = _nodes(velocity, "velocity")
    ddq = _nodes(acceleration, "acceleration")
    if dq.shape != q.shape or ddq.shape != q.shape:
        raise ValueError("position, velocity, and acceleration shapes must match")
    if not math.isfinite(sample_dt_s) or sample_dt_s <= 0.0:
        raise ValueError("sample_dt_s must be finite and positive")
    minimum_durations = np.asarray(minimum_segment_duration_s, dtype=np.float64)
    if minimum_durations.ndim == 0:
        minimum_durations = np.full(len(q) - 1, float(minimum_durations))
    else:
        minimum_durations = minimum_durations.reshape(-1)
    if minimum_durations.shape != (len(q) - 1,):
        raise ValueError("minimum segment durations must be scalar or one per segment")
    if not np.all(np.isfinite(minimum_durations)) or np.any(
        minimum_durations <= 0.0
    ):
        raise ValueError("minimum segment durations must be finite and positive")

    dof = q.shape[1]
    velocity_limit = _limits(max_velocity, dof, "max_velocity")
    acceleration_limit = _limits(max_acceleration, dof, "max_acceleration")
    jerk_limit = _limits(max_jerk, dof, "max_jerk")
    lower = None if min_position is None else np.asarray(min_position, dtype=np.float64)
    upper = None if max_position is None else np.asarray(max_position, dtype=np.float64)
    if (lower is None) != (upper is None):
        raise ValueError("min_position and max_position must be supplied together")
    if lower is not None:
        lower = lower.reshape(-1)
        upper = upper.reshape(-1)
        if lower.shape != (dof,) or upper.shape != (dof,):
            raise ValueError("position limits must contain one value per joint")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("position limits must be finite")
        if np.any(lower >= upper):
            raise ValueError("each minimum position must be below its maximum")

    otg = Ruckig(dof, sample_dt_s)
    output_q: list[list[float]] = []
    output_dq: list[list[float]] = []
    output_ddq: list[list[float]] = []
    node_indices = [0]
    durations: list[float] = []
    started = time.perf_counter()

    for segment in range(len(q) - 1):
        request = InputParameter(dof)
        request.current_position = q[segment]
        request.current_velocity = dq[segment]
        request.current_acceleration = ddq[segment]
        request.target_position = q[segment + 1]
        request.target_velocity = dq[segment + 1]
        request.target_acceleration = ddq[segment + 1]
        request.max_velocity = velocity_limit
        request.max_acceleration = acceleration_limit
        request.max_jerk = jerk_limit
        request.minimum_duration = minimum_durations[segment]
        request.duration_discretization = DurationDiscretization.Discrete
        if lower is not None and upper is not None:
            request.min_position = lower
            request.max_position = upper

        trajectory = Trajectory(dof)
        result = otg.calculate(request, trajectory)
        if result not in {Result.Working, Result.Finished}:
            raise RuntimeError(f"Ruckig failed at segment {segment}: {result}")
        steps = round(float(trajectory.duration) / sample_dt_s)
        if steps < 1 or not math.isclose(
            steps * sample_dt_s,
            float(trajectory.duration),
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError("Ruckig returned a non-discrete segment duration")
        start_step = 0 if segment == 0 else 1
        for step in range(start_step, steps + 1):
            state = trajectory.at_time(step * sample_dt_s)
            output_q.append(state[0])
            output_dq.append(state[1])
            output_ddq.append(state[2])
        durations.append(float(trajectory.duration))
        node_indices.append(node_indices[-1] + steps)

    return RuckigInterpolation(
        position=np.asarray(output_q),
        velocity=np.asarray(output_dq),
        acceleration=np.asarray(output_ddq),
        node_sample_indices=np.asarray(node_indices, dtype=np.int64),
        segment_durations_s=np.asarray(durations),
        calculation_wall_time_s=time.perf_counter() - started,
    )
