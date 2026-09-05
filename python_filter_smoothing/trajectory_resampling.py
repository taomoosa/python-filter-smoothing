"""YAML-selectable resampling of a long MPC joint path."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from .trajectory_filter import filter_joint_trajectory_savgol


@dataclass(frozen=True)
class ResampledJointTrajectory:
    """Uniform command samples plus resampling diagnostics."""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    raw_position: np.ndarray
    raw_velocity: np.ndarray
    raw_acceleration: np.ndarray
    node_sample_indices: np.ndarray
    segment_durations_s: np.ndarray
    interpolation_wall_time_s: float
    filter_wall_time_s: float
    method: str
    time_scale: float

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


def derivative_limit_ratios(
    velocity: np.ndarray,
    acceleration: np.ndarray,
    sample_dt_s: float,
    max_velocity: np.ndarray,
    max_acceleration: np.ndarray,
    max_jerk: np.ndarray,
) -> tuple[float, float, float]:
    """Return maximum velocity, acceleration, and discrete-jerk limit ratios."""

    jerk = np.zeros_like(acceleration)
    if len(jerk) > 1:
        jerk[1:] = np.diff(acceleration, axis=0) / sample_dt_s
        jerk[0] = jerk[1]
    return (
        float(np.max(np.abs(velocity) / max_velocity)),
        float(np.max(np.abs(acceleration) / max_acceleration)),
        float(np.max(np.abs(jerk) / max_jerk)),
    )


def _filter(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    sample_dt_s: float,
    options: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    settings = options["post_filter"]
    if str(settings["method"]) != "savgol_position":
        raise ValueError("resampling.post_filter.method must be savgol_position")
    started = time.perf_counter()
    result = filter_joint_trajectory_savgol(
        position,
        velocity,
        acceleration,
        sample_dt_s=sample_dt_s,
        window_length_samples=int(settings["window_length_samples"]),
        polynomial_order=int(settings["polynomial_order"]),
        preserve_endpoints=True,
    )
    return (
        result.position,
        result.velocity,
        result.acceleration,
        time.perf_counter() - started,
    )


def _hermite_at_scale(
    q: np.ndarray,
    dq: np.ndarray,
    ddq: np.ndarray,
    source_dt_s: float,
    sample_dt_s: float,
    requested_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    steps_per_segment = math.ceil(source_dt_s * requested_scale / sample_dt_s)
    segment_dt = steps_per_segment * sample_dt_s
    scale = segment_dt / source_dt_s
    node_time = np.arange(len(q), dtype=np.float64) * segment_dt
    sample_time = np.arange((len(q) - 1) * steps_per_segment + 1) * sample_dt_s
    scaled_dq = dq / scale
    scaled_dq[0] = dq[0]
    spline = CubicHermiteSpline(node_time, q, scaled_dq, axis=0)
    position = np.asarray(spline(sample_time, 0))
    velocity = np.asarray(spline(sample_time, 1))
    acceleration = np.asarray(spline(sample_time, 2))
    position[[0, -1]] = q[[0, -1]]
    velocity[0] = dq[0]
    velocity[-1] = dq[-1] / scale
    acceleration[0] = ddq[0]
    acceleration[-1] = ddq[-1] / scale**2
    node_indices = np.arange(len(q), dtype=np.int64) * steps_per_segment
    durations = np.full(len(q) - 1, segment_dt, dtype=np.float64)
    return position, velocity, acceleration, node_indices, durations, scale


def _hermite(
    q: np.ndarray,
    dq: np.ndarray,
    ddq: np.ndarray,
    source_dt_s: float,
    sample_dt_s: float,
    limits: tuple[np.ndarray, np.ndarray, np.ndarray],
    options: dict[str, Any],
) -> ResampledJointTrajectory:
    settings = options["hermite"]
    scale = float(settings["initial_time_scale"])
    margin = float(settings["time_scale_margin"])
    maximum_iterations = int(settings["max_time_scale_iterations"])
    if scale < 1.0 or margin <= 1.0 or maximum_iterations < 1:
        raise ValueError("invalid Hermite time-scaling configuration")
    started = time.perf_counter()
    total_filter_wall = 0.0
    for _ in range(maximum_iterations):
        raw_q, raw_dq, raw_ddq, node_indices, durations, scale = _hermite_at_scale(
            q, dq, ddq, source_dt_s, sample_dt_s, scale
        )
        out_q, out_dq, out_ddq, filter_wall = _filter(
            raw_q, raw_dq, raw_ddq, sample_dt_s, options
        )
        total_filter_wall += filter_wall
        ratios = derivative_limit_ratios(
            out_dq, out_ddq, sample_dt_s, *limits
        )
        required = max(ratios[0], math.sqrt(ratios[1]), np.cbrt(ratios[2]))
        if required <= 1.0 + 1.0e-9:
            return ResampledJointTrajectory(
                out_q,
                out_dq,
                out_ddq,
                raw_q,
                raw_dq,
                raw_ddq,
                node_indices,
                durations,
                time.perf_counter() - started - total_filter_wall,
                total_filter_wall,
                "hermite",
                scale,
            )
        scale *= float(required) * margin
    raise RuntimeError("Hermite resampling could not satisfy derivative limits")


def _ruckig(
    q: np.ndarray,
    dq: np.ndarray,
    ddq: np.ndarray,
    source_dt_s: float,
    sample_dt_s: float,
    limits: tuple[np.ndarray, np.ndarray, np.ndarray],
    position_bounds: tuple[np.ndarray, np.ndarray],
    options: dict[str, Any],
) -> ResampledJointTrajectory:
    from .ruckig_interpolation import interpolate_joint_path

    settings = options["ruckig"]
    margin = float(settings["limit_margin"])
    time_scale_margin = float(settings["time_scale_margin"])
    maximum_iterations = int(settings["max_time_scale_iterations"])
    if margin <= 0.0 or margin > 1.0:
        raise ValueError("resampling.ruckig.limit_margin must be in (0, 1]")
    if time_scale_margin <= 1.0 or maximum_iterations < 1:
        raise ValueError("invalid Ruckig time-scaling configuration")
    requested_scale = 1.0
    total_interpolation_wall = 0.0
    total_filter_wall = 0.0
    for _ in range(maximum_iterations):
        scaled_dq = dq / requested_scale
        scaled_ddq = ddq / requested_scale**2
        scaled_dq[0] = dq[0]
        scaled_ddq[0] = ddq[0]
        raw = interpolate_joint_path(
            q,
            scaled_dq,
            scaled_ddq,
            sample_dt_s=sample_dt_s,
            minimum_segment_duration_s=source_dt_s * requested_scale,
            max_velocity=limits[0] * margin,
            max_acceleration=limits[1] * margin,
            max_jerk=limits[2] * margin,
            min_position=position_bounds[0],
            max_position=position_bounds[1],
        )
        total_interpolation_wall += raw.calculation_wall_time_s
        out_q, out_dq, out_ddq, filter_wall = _filter(
            raw.position, raw.velocity, raw.acceleration, sample_dt_s, options
        )
        total_filter_wall += filter_wall
        ratios = derivative_limit_ratios(
            out_dq, out_ddq, sample_dt_s, *limits
        )
        required = max(ratios[0], math.sqrt(ratios[1]), np.cbrt(ratios[2]))
        if required <= 1.0 + 1.0e-9:
            return ResampledJointTrajectory(
                out_q,
                out_dq,
                out_ddq,
                raw.position,
                raw.velocity,
                raw.acceleration,
                raw.node_sample_indices,
                raw.segment_durations_s,
                total_interpolation_wall,
                total_filter_wall,
                "ruckig",
                raw.duration_s / ((len(q) - 1) * source_dt_s),
            )
        requested_scale *= float(required) * time_scale_margin
    raise RuntimeError("Ruckig resampling could not satisfy derivative limits")


def resample_joint_trajectory(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    *,
    source_dt_s: float,
    sample_dt_s: float,
    max_velocity: np.ndarray,
    max_acceleration: np.ndarray,
    max_jerk: np.ndarray,
    min_position: np.ndarray,
    max_position: np.ndarray,
    options: dict[str, Any],
) -> ResampledJointTrajectory:
    """Resample with the configured Hermite default or optional Ruckig path."""

    q = _nodes(position, "position")
    dq = _nodes(velocity, "velocity")
    ddq = _nodes(acceleration, "acceleration")
    if q.shape != dq.shape or q.shape != ddq.shape:
        raise ValueError("position, velocity, and acceleration shapes must match")
    if not math.isfinite(source_dt_s) or source_dt_s <= 0.0:
        raise ValueError("source_dt_s must be finite and positive")
    if not math.isfinite(sample_dt_s) or sample_dt_s <= 0.0:
        raise ValueError("sample_dt_s must be finite and positive")
    limits = (
        _limits(max_velocity, q.shape[1], "max_velocity"),
        _limits(max_acceleration, q.shape[1], "max_acceleration"),
        _limits(max_jerk, q.shape[1], "max_jerk"),
    )
    method = str(options["method"])
    if method == "hermite":
        return _hermite(q, dq, ddq, source_dt_s, sample_dt_s, limits, options)
    if method == "ruckig":
        return _ruckig(
            q,
            dq,
            ddq,
            source_dt_s,
            sample_dt_s,
            limits,
            (np.asarray(min_position), np.asarray(max_position)),
            options,
        )
    raise ValueError("resampling.method must be hermite or ruckig")
