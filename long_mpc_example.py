"""Generate servo commands from long cuRobo MPC paths."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from curobo.config_io import load_yaml
from curobo.types import JointState, Pose, RobotState

from python_filter_smoothing.continuous_trajectory import ContinuousMpcTrajectory
from python_filter_smoothing.mpc_application import (
    CartesianTargetResolver,
    CspaceAcceptanceMode,
    JointLimitEvaluation,
    MpcCommandApplication,
    PoseError,
    PoseProgressPolicy,
    ProgressEvaluation,
    TargetResolution,
    TargetResolutionError,
    TrajectoryConstraintPolicy,
    TrajectoryExecutionMode,
    evaluate_joint_trajectory_limits,
)
from python_filter_smoothing.pose_utils import relative_rot6d_to_quaternion
from python_filter_smoothing.predictive_mpc import MpcHorizon
from python_filter_smoothing.servo_queue import (
    extend_stationary_trajectory,
    slice_joint_trajectory,
)
from python_filter_smoothing.trajectory_resampling import (
    ResampledJointTrajectory,
    connect_initial_state_to_horizon,
    resample_joint_trajectory,
)

DEFAULT_CONFIG = (
    Path(__file__).parent / "python_filter_smoothing/configs/long_mpc_application.yml"
)


def _config_path(value: str, parent: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (parent / path).resolve()


def _artifact_reference(path: str | Path, output_directory: Path) -> str:
    """Store a portable path without exposing the user's absolute directory."""

    return Path(
        os.path.relpath(Path(path).resolve(), start=output_directory.resolve())
    ).as_posix()


def _initial_state(
    controller: ContinuousMpcTrajectory, options: Mapping[str, Any]
) -> JointState:
    """Create the configured start state in the solver's joint order.

    A mapping may override only selected joints from the robot configuration's
    default. A sequence must specify every joint in ``controller.joint_names``.
    """

    default = controller.default_state()
    configured = options.get("initial_joint_positions_rad")
    if configured is None:
        return default

    position = default.position.clone()
    if isinstance(configured, Mapping):
        unknown = set(configured) - set(controller.joint_names)
        if unknown:
            raise ValueError(
                "initial_joint_positions_rad contains unknown joints: "
                f"{sorted(unknown)}"
            )
        indices = {name: index for index, name in enumerate(controller.joint_names)}
        for name, value in configured.items():
            position[0, indices[name]] = float(value)
    else:
        values = torch.as_tensor(
            configured, device=position.device, dtype=position.dtype
        )
        if values.ndim != 1 or len(values) != len(controller.joint_names):
            raise ValueError(
                "initial_joint_positions_rad must contain one value per joint"
            )
        position[0].copy_(values)
    if not bool(torch.all(torch.isfinite(position)).item()):
        raise ValueError("initial_joint_positions_rad must be finite")
    return JointState.from_position(position, joint_names=controller.joint_names)


@dataclass(frozen=True)
class _GeneratedPath:
    state: JointState
    mpc_wall_time_s: float
    resampled: ResampledJointTrajectory
    validation_wall_time_s: float
    constraint_violations: dict[str, int]
    maximum_constraint_values: dict[str, float]
    maximum_node_error_rad: float
    source_duration_s: float
    initial_state_errors: tuple[float, float, float]
    raw_max_jerk_rad_s3: float
    filtered_max_jerk_rad_s3: float
    filter_changes: tuple[float, float, float]
    candidate_feasible: tuple[bool, ...]
    selected_candidate_iterations: int
    initial_pose_error: PoseError
    terminal_pose_error: PoseError
    progress: ProgressEvaluation
    joint_limits: JointLimitEvaluation
    native_full_horizon_feasible: bool
    selected_target_candidate_index: int | None
    selected_target_offset_m: tuple[float, float, float] | None
    target_resolution: TargetResolution


@dataclass(frozen=True)
class _RawCandidate:
    iterations: int
    horizon: MpcHorizon
    position_error_m: float
    rotation_error_rad: float
    progress: ProgressEvaluation | None = None
    selectable_by_application: bool | None = None


class _PathGenerationError(RuntimeError):
    """Candidate rejection reported to the application loop."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _rank_feasible_candidates(
    candidates: list[_RawCandidate],
) -> list[_RawCandidate]:
    """Keep feasible results even when a later solve regresses."""

    return sorted(
        (
            candidate
            for candidate in candidates
            if (
                candidate.horizon.full_horizon_feasible
                if candidate.selectable_by_application is None
                else candidate.selectable_by_application
            )
            and (candidate.progress is None or candidate.progress.accepted)
        ),
        key=lambda candidate: (
            candidate.position_error_m,
            candidate.rotation_error_rad,
            candidate.iterations,
        ),
    )


def _as_command_state(
    trajectory: ResampledJointTrajectory,
    template: JointState,
    joint_names: list[str],
    dt_s: float,
) -> JointState:
    device, dtype = template.position.device, template.position.dtype
    state = JointState.from_position(
        torch.as_tensor(trajectory.position, device=device, dtype=dtype)[None],
        joint_names=joint_names,
    )
    state.velocity = torch.as_tensor(trajectory.velocity, device=device, dtype=dtype)[
        None
    ]
    state.acceleration = torch.as_tensor(
        trajectory.acceleration, device=device, dtype=dtype
    )[None]
    state.jerk = torch.zeros_like(state.acceleration)
    if state.acceleration.shape[1] > 1:
        state.jerk[:, 1:] = torch.diff(state.acceleration, dim=1) / dt_s
        state.jerk[:, 0] = state.jerk[:, 1]
    state.dt = torch.full((1,), dt_s, device=device, dtype=dtype)
    return state


def _validate_with_curobo(
    controller: ContinuousMpcTrajectory,
    state: JointState,
    ignored_constraints: frozenset[str] = frozenset(),
) -> tuple[bool, dict[str, int], dict[str, float]]:
    """Evaluate named cuRobo constraints at every sample.

    Ignoring ``cspace`` is an application decision and does not ignore the separate
    self- or scene-collision constraints. The caller must then enforce physical
    position and derivative limits independently.
    """

    rollout = controller.solver.metrics_rollout
    horizon = int(rollout.horizon)
    violations: dict[str, int] = {}
    maxima: dict[str, float] = {}
    all_feasible = True
    for start in range(0, state.position.shape[1], horizon):
        count = min(horizon, state.position.shape[1] - start)
        chunk = slice_joint_trajectory(state, start, start + count)
        chunk = extend_stationary_trajectory(chunk, horizon)
        robot = RobotState(
            joint_state=chunk,
            cuda_robot_model_state=controller.solver.compute_kinematics(chunk),
        )
        metrics = rollout.compute_metrics_from_state(robot)
        constraints = metrics.costs_and_constraints.constraints
        for name, value in zip(constraints.names, constraints.values, strict=True):
            selected = value[:, :count]
            count_violations = int(torch.count_nonzero(selected > 0.0).item())
            violations[name] = violations.get(name, 0) + count_violations
            maxima[name] = max(maxima.get(name, 0.0), float(selected.max().item()))
            if name not in ignored_constraints and count_violations:
                all_feasible = False
    return all_feasible, violations, maxima


def _targets(
    controller: ContinuousMpcTrajectory,
    options: dict[str, Any],
    initial: JointState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pose = controller.solver.compute_kinematics(initial).tool_poses.to_dict()[
        controller.solver.tool_frames[0]
    ]
    rotation = Pose(position=pose.position, quaternion=pose.quaternion).get_rotation()
    if not isinstance(rotation, torch.Tensor):
        raise TypeError("initial tool pose is missing rotation")
    offsets = torch.as_tensor(
        options["target_offsets_m"],
        device=pose.position.device,
        dtype=pose.position.dtype,
    )
    rotations = torch.as_tensor(
        options["target_rotation_offsets_rot6d"],
        device=pose.position.device,
        dtype=pose.position.dtype,
    )
    if offsets.ndim != 2 or offsets.shape[1] != 3 or len(offsets) == 0:
        raise ValueError("target_offsets_m must have shape [N, 3]")
    if rotations.ndim == 1:
        rotations = rotations.reshape(1, 6).expand(len(offsets), 6)
    if rotations.shape != (len(offsets), 6):
        raise ValueError("target_rotation_offsets_rot6d must have shape [N, 6]")
    return (
        pose.position.reshape(1, 3) + offsets,
        relative_rot6d_to_quaternion(rotation, rotations),
        offsets,
    )


def _maximum_jerk(acceleration: np.ndarray, dt_s: float) -> float:
    if len(acceleration) < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(acceleration, axis=0) / dt_s)))


def _state_error(
    controller: ContinuousMpcTrajectory,
    state: JointState,
    target_position: torch.Tensor,
    target_quaternion: torch.Tensor,
) -> PoseError:
    pose = controller.solver.compute_kinematics(state).tool_poses.to_dict()[
        controller.solver.tool_frames[0]
    ]
    position_error = float(
        torch.linalg.vector_norm(
            pose.position.reshape(-1, 3)[0] - target_position
        ).item()
    )
    actual_quaternion = pose.quaternion.reshape(-1, 4)[0]
    quaternion_dot = torch.abs(torch.dot(actual_quaternion, target_quaternion))
    rotation_error = float(
        (2.0 * torch.acos(torch.clamp(quaternion_dot, 0.0, 1.0))).item()
    )
    return PoseError(position_error, rotation_error)


def _candidate_error(
    controller: ContinuousMpcTrajectory,
    horizon: MpcHorizon,
    target_position: torch.Tensor,
    target_quaternion: torch.Tensor,
) -> PoseError:
    endpoint = JointState.from_position(
        horizon.states.position[:, -1], joint_names=controller.joint_names
    )
    return _state_error(controller, endpoint, target_position, target_quaternion)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mpc-config",
        type=Path,
        help="override long_mpc_config with a mechanism-specific MPC YAML",
    )
    parser.add_argument("--duration", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--execution-mode",
        choices=[mode.value for mode in TrajectoryExecutionMode],
        help="override application.execution_mode",
    )
    parser.add_argument(
        "--cspace-acceptance",
        choices=[mode.value for mode in CspaceAcceptanceMode],
        help="override application.constraint_acceptance.cspace_mode",
    )
    parser.add_argument(
        "--maximum-velocity-ratio",
        type=float,
        help="application acceptance limit relative to the robot limit",
    )
    parser.add_argument(
        "--maximum-acceleration-ratio",
        type=float,
        help="application acceptance limit relative to the robot limit",
    )
    parser.add_argument(
        "--maximum-jerk-ratio",
        type=float,
        help="application acceptance limit relative to the robot limit",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_yaml(str(config_path))
    options = config["example"]
    application_options = config["application"]
    resampling_options = config["resampling"]
    mpc_config_path = (
        args.mpc_config.resolve()
        if args.mpc_config is not None
        else _config_path(config["long_mpc_config"], config_path.parent)
    )
    controller = ContinuousMpcTrajectory(mpc_config_path)
    initial = _initial_state(controller, options)
    controller.setup(initial)
    target_position, target_quaternion, offsets = _targets(controller, options, initial)

    servo_dt = float(options["servo_dt_s"])
    duration_s = float(args.duration or options["duration_s"])
    total_samples = round(duration_s / servo_dt)
    if total_samples < 1 or not math.isclose(total_samples * servo_dt, duration_s):
        raise ValueError("duration must be a positive multiple of servo_dt_s")
    target_period = float(options["target_period_s"])
    target_samples = round(target_period / servo_dt)
    if target_samples < 1 or not math.isclose(target_samples * servo_dt, target_period):
        raise ValueError("target_period_s must be a positive servo multiple")
    execution_mode = TrajectoryExecutionMode(
        args.execution_mode or application_options["execution_mode"]
    )
    connection_delay = float(application_options["planning_connection_delay_s"])
    connection_samples = round(connection_delay / servo_dt)
    if connection_samples < 0 or not math.isclose(
        connection_samples * servo_dt, connection_delay
    ):
        raise ValueError(
            "planning_connection_delay_s must be a nonnegative servo multiple"
        )
    if (
        execution_mode is TrajectoryExecutionMode.FUTURE_QUEUE
        and connection_samples < 1
    ):
        raise ValueError("future_queue requires a positive planning connection delay")
    retry_period = float(application_options["failed_plan_retry_period_s"])
    retry_samples = round(retry_period / servo_dt)
    if retry_samples < 1 or not math.isclose(retry_samples * servo_dt, retry_period):
        raise ValueError("failed_plan_retry_period_s must be a positive servo multiple")
    progress_policy = PoseProgressPolicy.from_mapping(
        application_options.get("progress", {})
    )
    target_resolver = CartesianTargetResolver.from_mapping(
        application_options.get("target_resolution", {})
    )
    constraint_options = dict(application_options.get("constraint_acceptance", {}))
    if args.cspace_acceptance is not None:
        constraint_options["cspace_mode"] = args.cspace_acceptance
    if args.maximum_velocity_ratio is not None:
        constraint_options["maximum_velocity_ratio"] = args.maximum_velocity_ratio
    if args.maximum_acceleration_ratio is not None:
        constraint_options["maximum_acceleration_ratio"] = (
            args.maximum_acceleration_ratio
        )
    if args.maximum_jerk_ratio is not None:
        constraint_options["maximum_jerk_ratio"] = args.maximum_jerk_ratio
    constraint_policy = TrajectoryConstraintPolicy.from_mapping(constraint_options)

    transition = controller.solver.transition_model
    bounds = transition.get_state_bounds()
    if bool(
        torch.any(initial.position < bounds.position[0]).item()
        or torch.any(initial.position > bounds.position[1]).item()
    ):
        raise ValueError("initial_joint_positions_rad exceeds robot position bounds")
    nominal_velocity_limit = transition.max_velocity
    nominal_acceleration_limit = transition.max_acceleration
    nominal_jerk_limit = transition.max_jerk
    resampling_velocity_limit = nominal_velocity_limit * float(
        resampling_options["velocity_limit_scale"]
    )
    resampling_acceleration_limit = nominal_acceleration_limit * float(
        resampling_options["acceleration_limit_scale"]
    )
    resampling_jerk_limit = nominal_jerk_limit * float(
        resampling_options["jerk_limit_scale"]
    )
    numpy_limits = (
        resampling_velocity_limit.detach().cpu().numpy(),
        resampling_acceleration_limit.detach().cpu().numpy(),
        resampling_jerk_limit.detach().cpu().numpy(),
        bounds.position[0].detach().cpu().numpy(),
        bounds.position[1].detach().cpu().numpy(),
    )

    def generate(target_index: int, start: JointState) -> _GeneratedPath:
        controller.setup(start)
        try:
            target_resolution = target_resolver.set_target(
                controller,
                start,
                target_position[target_index],
                target_quaternion[target_index],
            )
        except TargetResolutionError as error:
            raise _PathGenerationError("target_resolution", str(error)) from error
        target_selection = controller.last_target_selection
        initial_pose_error = _state_error(
            controller,
            start,
            target_position[target_index],
            target_quaternion[target_index],
        )
        candidates: list[_RawCandidate] = []
        for index, iterations in enumerate(controller.candidate_iterations):
            if index:
                controller.prepare_candidate(start, iterations)
            horizon = controller.solve_horizon()
            terminal_pose_error = _candidate_error(
                controller,
                horizon,
                target_position[target_index],
                target_quaternion[target_index],
            )
            progress = progress_policy.evaluate(initial_pose_error, terminal_pose_error)
            candidates.append(
                _RawCandidate(
                    iterations,
                    horizon,
                    terminal_pose_error.position_m,
                    terminal_pose_error.rotation_rad,
                    progress,
                    horizon.full_horizon_feasible
                    or constraint_policy.cspace_mode
                    is CspaceAcceptanceMode.PHYSICAL_LIMITS,
                )
            )
        ranked = _rank_feasible_candidates(candidates)
        if not ranked:
            native_feasible = any(
                item.horizon.full_horizon_feasible for item in candidates
            )
            reason = "no_progress" if native_feasible else "mpc_infeasible"
            raise _PathGenerationError(
                reason,
                f"long MPC target {target_index} produced no acceptable candidate",
            )

        mpc_wall = sum(item.horizon.wall_time_s for item in candidates)
        for selected in ranked:
            source = selected.horizon.states
            initial_errors = tuple(
                float(
                    (getattr(source, name)[:, 0] - getattr(start, name))
                    .abs()
                    .max()
                    .item()
                )
                for name in ("position", "velocity", "acceleration")
            )
            names = ("position", "velocity", "acceleration")
            nodes = connect_initial_state_to_horizon(
                tuple(getattr(start, name)[0].detach().cpu().numpy() for name in names),
                tuple(
                    getattr(source, name)[0].detach().cpu().numpy() for name in names
                ),
            )
            try:
                resampled = resample_joint_trajectory(
                    *nodes,
                    source_dt_s=controller.timing.command_dt_s,
                    sample_dt_s=servo_dt,
                    max_velocity=numpy_limits[0],
                    max_acceleration=numpy_limits[1],
                    max_jerk=numpy_limits[2],
                    min_position=numpy_limits[3],
                    max_position=numpy_limits[4],
                    options=resampling_options,
                )
            except (RuntimeError, ValueError):
                continue
            state = _as_command_state(
                resampled, start, controller.joint_names, servo_dt
            )
            validation_started = time.perf_counter()
            valid, violations, maxima = _validate_with_curobo(
                controller,
                state,
                constraint_policy.ignored_curobo_constraints,
            )
            joint_limits = evaluate_joint_trajectory_limits(
                state,
                start,
                dt_s=servo_dt,
                minimum_position=bounds.position[0],
                maximum_position=bounds.position[1],
                maximum_velocity=nominal_velocity_limit,
                maximum_acceleration=nominal_acceleration_limit,
                maximum_jerk=nominal_jerk_limit,
                policy=constraint_policy,
            )
            validation_wall = time.perf_counter() - validation_started
            if not valid or not joint_limits.accepted:
                continue
            node_error = float(
                np.max(
                    np.abs(resampled.position[resampled.node_sample_indices] - nodes[0])
                )
            )
            changes = tuple(
                float(np.max(np.abs(filtered - raw)))
                for filtered, raw in (
                    (resampled.position, resampled.raw_position),
                    (resampled.velocity, resampled.raw_velocity),
                    (resampled.acceleration, resampled.raw_acceleration),
                )
            )
            return _GeneratedPath(
                state=state,
                mpc_wall_time_s=mpc_wall,
                resampled=resampled,
                validation_wall_time_s=validation_wall,
                constraint_violations=violations,
                maximum_constraint_values=maxima,
                maximum_node_error_rad=node_error,
                source_duration_s=(len(nodes[0]) - 1) * controller.timing.command_dt_s,
                initial_state_errors=initial_errors,
                raw_max_jerk_rad_s3=_maximum_jerk(resampled.raw_acceleration, servo_dt),
                filtered_max_jerk_rad_s3=_maximum_jerk(
                    resampled.acceleration, servo_dt
                ),
                filter_changes=changes,
                candidate_feasible=tuple(
                    item.horizon.full_horizon_feasible for item in candidates
                ),
                selected_candidate_iterations=selected.iterations,
                initial_pose_error=initial_pose_error,
                terminal_pose_error=PoseError(
                    selected.position_error_m, selected.rotation_error_rad
                ),
                progress=selected.progress
                or progress_policy.evaluate(
                    initial_pose_error,
                    PoseError(selected.position_error_m, selected.rotation_error_rad),
                ),
                joint_limits=joint_limits,
                native_full_horizon_feasible=(selected.horizon.full_horizon_feasible),
                selected_target_candidate_index=(
                    target_selection.candidate_index
                    if target_selection is not None
                    else None
                ),
                selected_target_offset_m=(
                    tuple(
                        float(value)
                        for value in (
                            target_selection.position_m - target_position[target_index]
                        )
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    if target_selection is not None
                    else None
                ),
                target_resolution=target_resolution,
            )
        raise _PathGenerationError(
            "final_validation",
            f"long MPC target {target_index} had no candidate passing final validation",
        )

    startup_started = time.perf_counter()
    startup = generate(0, initial)
    startup_wall = time.perf_counter() - startup_started
    application = MpcCommandApplication(
        startup.state,
        mode=execution_mode,
        connection_samples=connection_samples,
    )

    q_parts: list[torch.Tensor] = []
    dq_parts: list[torch.Tensor] = []
    ddq_parts: list[torch.Tensor] = []
    target_ids: list[int] = []
    update_records: list[list[float]] = []
    generated = [startup]
    produced = 0
    last_attempt = 0
    accepted_updates = 1
    infeasible_updates = 0
    rejected_updates = 0
    deadline_misses = 0
    rejected_updates_by_reason: dict[str, int] = {}

    def append(commands: JointState) -> None:
        nonlocal produced
        count = min(commands.position.shape[1], total_samples - produced)
        q_parts.append(commands.position[0, :count])
        dq_parts.append(commands.velocity[0, :count])
        ddq_parts.append(commands.acceleration[0, :count])
        target_ids.extend(
            ((produced + sample) // target_samples) % len(offsets)
            for sample in range(count)
        )
        produced += count

    while produced < total_samples:
        target_index = (produced // target_samples) % len(offsets)
        if target_index != last_attempt:
            plan = application.begin_plan()
            start = plan.initial_state
            application_started = time.perf_counter()
            candidate = None
            rejection_reason = ""
            try:
                candidate = generate(target_index, start)
            except _PathGenerationError as error:
                infeasible_updates += int(error.reason == "mpc_infeasible")
                rejection_reason = error.reason
            application_wall = time.perf_counter() - application_started
            elapsed_samples = max(1, math.ceil(application_wall / servo_dt))
            commands_during_planning = application.consume_during_planning(
                min(elapsed_samples, total_samples - produced)
            )
            if commands_during_planning is not None:
                append(commands_during_planning)
            commit = (
                application.publish(plan, candidate.state)
                if candidate is not None and produced < total_samples
                else None
            )
            accepted = bool(commit is not None and commit.accepted)
            on_time = (
                execution_mode is TrajectoryExecutionMode.IMMEDIATE
                or elapsed_samples <= connection_samples
            )
            if accepted:
                last_attempt = target_index
                generated.append(candidate)
                accepted_updates += 1
            else:
                rejected_updates += 1
                rejection_reason = rejection_reason or (
                    commit.reason if commit is not None else "no_candidate"
                )
                rejected_updates_by_reason[rejection_reason] = (
                    rejected_updates_by_reason.get(rejection_reason, 0) + 1
                )
                if (
                    execution_mode is TrajectoryExecutionMode.IMMEDIATE
                    and produced < total_samples
                ):
                    append(
                        application.consume(
                            min(retry_samples, total_samples - produced)
                        )
                    )
            deadline_misses += int(not on_time)
            update_records.append(
                [
                    produced * servo_dt,
                    float(target_index),
                    candidate.mpc_wall_time_s if candidate else math.nan,
                    candidate.selected_candidate_iterations if candidate else math.nan,
                    float(sum(candidate.candidate_feasible)) if candidate else 0.0,
                    candidate.resampled.interpolation_wall_time_s
                    if candidate
                    else math.nan,
                    candidate.resampled.filter_wall_time_s if candidate else math.nan,
                    candidate.validation_wall_time_s if candidate else math.nan,
                    application_wall,
                    float(elapsed_samples),
                    float(on_time),
                    float(candidate is not None),
                    float(accepted),
                    candidate.resampled.duration_s if candidate else math.nan,
                    candidate.initial_pose_error.position_m if candidate else math.nan,
                    candidate.terminal_pose_error.position_m if candidate else math.nan,
                    candidate.progress.position_improvement_m
                    if candidate
                    else math.nan,
                    float(candidate.progress.accepted) if candidate else 0.0,
                    float(candidate.native_full_horizon_feasible) if candidate else 0.0,
                    candidate.joint_limits.maximum_velocity_ratio
                    if candidate
                    else math.nan,
                    candidate.joint_limits.maximum_acceleration_ratio
                    if candidate
                    else math.nan,
                    candidate.joint_limits.maximum_jerk_ratio
                    if candidate
                    else math.nan,
                    float(candidate.constraint_violations.get("cspace", 0))
                    if candidate
                    else 0.0,
                    float(candidate.target_resolution.used_proxy) if candidate else 0.0,
                    float(candidate.target_resolution.ik_attempts)
                    if candidate
                    else 0.0,
                    candidate.target_resolution.retreat_distance_m
                    if candidate
                    else math.nan,
                    candidate.target_resolution.orientation_fraction
                    if candidate
                    else math.nan,
                ]
            )
            continue
        boundary = min(
            total_samples, ((produced // target_samples) + 1) * target_samples
        )
        append(application.consume(boundary - produced))

    q, dq, ddq = map(torch.cat, (q_parts, dq_parts, ddq_parts))
    jerk = torch.zeros_like(ddq)
    if len(ddq) > 1:
        jerk[1:] = torch.diff(ddq, dim=0) / servo_dt
        jerk[0] = jerk[1]
    time_s = torch.arange(len(q), device=q.device, dtype=q.dtype) * servo_dt
    target_tensor = torch.as_tensor(target_ids, device=q.device, dtype=torch.long)
    desired_position = target_position[target_tensor]
    desired_quaternion = target_quaternion[target_tensor]
    tool = controller.solver.compute_kinematics(
        JointState.from_position(q[None], joint_names=controller.joint_names)
    ).tool_poses.to_dict()[controller.solver.tool_frames[0]]
    tool_position = tool.position.reshape(-1, 3)
    if tool_position.shape[0] != len(q):
        raise RuntimeError("forward kinematics returned an unexpected trajectory shape")

    output = args.output or _config_path(
        options["output_directory"], config_path.parent
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    columns = (
        ["time_s", "target_index"]
        + [
            f"{field}_{joint}"
            for field in ("q_rad", "dq_rad_s", "ddq_rad_s2", "jerk_rad_s3")
            for joint in controller.joint_names
        ]
        + [
            "tool_x_m",
            "tool_y_m",
            "tool_z_m",
            "target_x_m",
            "target_y_m",
            "target_z_m",
            "target_qw",
            "target_qx",
            "target_qy",
            "target_qz",
        ]
    )
    values = (
        torch.cat(
            (
                time_s[:, None],
                target_tensor.to(q.dtype)[:, None],
                q,
                dq,
                ddq,
                jerk,
                tool_position,
                desired_position,
                desired_quaternion,
            ),
            dim=1,
        )
        .cpu()
        .numpy()
    )
    np.savetxt(
        output / "trajectory.csv", values, delimiter=",", header=",".join(columns)
    )
    update_columns = [
        "publish_time_s",
        "target_index",
        "long_mpc_wall_time_s",
        "selected_candidate_iterations",
        "feasible_candidate_count",
        "resampling_wall_time_s",
        "post_filter_wall_time_s",
        "validation_wall_time_s",
        "application_wall_time_s",
        "elapsed_servo_samples",
        "met_connection_deadline",
        "path_constraint_feasible",
        "accepted",
        "path_duration_s",
        "initial_position_error_m",
        "terminal_position_error_m",
        "position_improvement_m",
        "progress_accepted",
        "native_full_horizon_feasible",
        "maximum_velocity_ratio",
        "maximum_acceleration_ratio",
        "maximum_jerk_ratio",
        "curobo_cspace_positive_samples",
        "target_proxy_used",
        "target_resolution_ik_attempts",
        "target_retreat_distance_m",
        "target_orientation_fraction",
    ]
    np.savetxt(
        output / "planner_updates.csv",
        np.asarray(update_records),
        delimiter=",",
        header=",".join(update_columns),
    )

    velocity_ratio = dq.abs() / nominal_velocity_limit
    acceleration_ratio = ddq.abs() / nominal_acceleration_limit
    jerk_ratio = jerk.abs() / nominal_jerk_limit
    errors = torch.linalg.vector_norm(tool_position - desired_position, dim=1)
    completed_segment_ends = torch.nonzero(
        target_tensor[1:] != target_tensor[:-1], as_tuple=False
    ).flatten()
    completed_errors = errors[completed_segment_ends]
    method = str(resampling_options["method"])
    summary = {
        "trajectory_generator": f"long_mpc_plus_{method}_savgol",
        "resampling_method": method,
        "samples": len(q),
        "duration_s": len(q) * servo_dt,
        "command_dt_s": servo_dt,
        "optimizer_execution": {
            "setup_cold_iterations": controller.config["optimizer"][
                "cold_start_iterations"
            ],
            "candidate_iterations": list(controller.candidate_iterations),
            "warm_iterations": controller.config["optimizer"]["warm_start_iterations"],
            "fixed_iterations": controller.config["optimizer"]["fixed_iterations"],
            "return_best_action": controller.config["optimizer"]["return_best_action"],
            "use_ik_joint_reference": controller.config["optimizer"]["target_update"][
                "use_ik_joint_reference"
            ],
            "seed_from_ik": controller.config["optimizer"]["target_update"][
                "seed_from_ik"
            ],
            "ik_fallback_seeds": controller.config["optimizer"]["target_update"][
                "ik_fallback_seeds"
            ],
            "ik_position_offsets_m": controller.config["optimizer"][
                "target_update"
            ].get("ik_position_offsets_m", [[0.0, 0.0, 0.0]]),
        },
        "optimizer_weights": {
            name: controller.config["optimizer"][name]
            for name in (
                "tool_pose_weight",
                "non_terminal_tool_pose_weight_factor",
                "cspace_bound_weight",
                "squared_l2_regularization_weight",
                "scene_collision_weight",
                "self_collision_weight",
            )
            if name in controller.config["optimizer"]
        },
        "timing_configuration": controller.config["timing"],
        "collision_configuration": controller.config["collision"],
        "robot_limit_scales": {
            name: controller.config["robot"][name]
            for name in (
                "velocity_limit_scale",
                "acceleration_limit_scale",
                "jerk_limit_scale",
            )
        },
        "joint_names": controller.joint_names,
        "application_config": _artifact_reference(config_path, output),
        "mpc_config": _artifact_reference(mpc_config_path, output),
        "robot": _artifact_reference(controller.config["robot"]["config"], output),
        "scene_model": _artifact_reference(controller.config["scene"], output),
        "initial_joint_positions_rad": initial.position[0].cpu().tolist(),
        "target_offsets_m": offsets.cpu().tolist(),
        "target_period_s": target_period,
        "application_execution_mode": execution_mode.value,
        "planning_connection_delay_s": (
            connection_delay
            if execution_mode is TrajectoryExecutionMode.FUTURE_QUEUE
            else 0.0
        ),
        "failed_plan_retry_period_s": retry_period,
        "progress_policy": {
            "enabled": progress_policy.enabled,
            "minimum_position_improvement_m": progress_policy.minimum_position_improvement_m,
            "position_tolerance_m": progress_policy.position_tolerance_m,
            "check_rotation": progress_policy.check_rotation,
            "minimum_rotation_improvement_rad": progress_policy.minimum_rotation_improvement_rad,
            "rotation_tolerance_rad": progress_policy.rotation_tolerance_rad,
        },
        "target_resolution_policy": {
            "enabled": target_resolver.policy.enabled,
            "coarse_position_samples": (target_resolver.policy.coarse_position_samples),
            "refinement_iterations": target_resolver.policy.refinement_iterations,
            "clearance_m": target_resolver.policy.clearance_m,
            "orientation_fractions": list(target_resolver.policy.orientation_fractions),
        },
        "target_resolutions": [
            {
                "used_proxy": item.target_resolution.used_proxy,
                "reason": item.target_resolution.reason,
                "ik_attempts": item.target_resolution.ik_attempts,
                "retreat_distance_m": item.target_resolution.retreat_distance_m,
                "orientation_fraction": (item.target_resolution.orientation_fraction),
                "selected_position_m": (
                    item.target_resolution.selected.position_m.cpu().tolist()
                ),
            }
            for item in generated
        ],
        "proxy_target_paths": sum(
            item.target_resolution.used_proxy for item in generated
        ),
        "constraint_acceptance_policy": {
            "cspace_mode": constraint_policy.cspace_mode.value,
            "maximum_velocity_ratio": constraint_policy.maximum_velocity_ratio,
            "maximum_acceleration_ratio": constraint_policy.maximum_acceleration_ratio,
            "maximum_jerk_ratio": constraint_policy.maximum_jerk_ratio,
            "position_tolerance_rad": constraint_policy.position_tolerance_rad,
            "collision_constraints": "strict",
        },
        "long_mpc_dt_s": controller.timing.command_dt_s,
        "resampling": resampling_options,
        "startup_application_wall_time_s": startup_wall,
        "accepted_long_paths": accepted_updates,
        "rejected_long_updates": rejected_updates,
        "infeasible_long_updates": infeasible_updates,
        "rejected_updates_by_reason": rejected_updates_by_reason,
        "planning_deadline_misses": deadline_misses,
        "selected_candidate_iterations": [
            item.selected_candidate_iterations for item in generated
        ],
        "selected_target_candidate_indices": [
            item.selected_target_candidate_index for item in generated
        ],
        "selected_target_offsets_m": [
            item.selected_target_offset_m for item in generated
        ],
        "final_candidate_infeasible_paths": sum(
            not item.candidate_feasible[-1] for item in generated
        ),
        "all_accepted_paths_curobo_constraint_feasible": all(
            not any(item.constraint_violations.values()) for item in generated
        ),
        "all_accepted_paths_collision_feasible": all(
            item.constraint_violations.get("scene_collision", 0) == 0
            and item.constraint_violations.get("self_collision", 0) == 0
            for item in generated
        ),
        "accepted_paths_native_infeasible": sum(
            not item.native_full_horizon_feasible for item in generated
        ),
        "accepted_paths_with_curobo_cspace_positive": sum(
            item.constraint_violations.get("cspace", 0) > 0 for item in generated
        ),
        "curobo_constraint_violations_by_component": {
            name: sum(item.constraint_violations.get(name, 0) for item in generated)
            for name in sorted(
                {name for item in generated for name in item.constraint_violations}
            )
        },
        "curobo_maximum_constraint_by_component": {
            name: max(
                item.maximum_constraint_values.get(name, 0.0) for item in generated
            )
            for name in sorted(
                {name for item in generated for name in item.maximum_constraint_values}
            )
        },
        "max_filtered_long_mpc_node_position_error_rad": max(
            item.maximum_node_error_rad for item in generated
        ),
        "max_initial_long_state_error": {
            name: max(item.initial_state_errors[index] for item in generated)
            for index, name in enumerate(
                ("position_rad", "velocity_rad_s", "acceleration_rad_s2")
            )
        },
        "path_duration_s": [item.resampled.duration_s for item in generated],
        "duration_scale_vs_long_mpc": [
            item.resampled.duration_s / item.source_duration_s for item in generated
        ],
        "maximum_interpolation_wall_time_s": max(
            item.resampled.interpolation_wall_time_s for item in generated
        ),
        "maximum_filter_wall_time_s": max(
            item.resampled.filter_wall_time_s for item in generated
        ),
        "max_raw_discrete_jerk_rad_s3": max(
            item.raw_max_jerk_rad_s3 for item in generated
        ),
        "max_filtered_discrete_jerk_rad_s3": max(
            item.filtered_max_jerk_rad_s3 for item in generated
        ),
        "max_filter_change": {
            name: max(item.filter_changes[index] for item in generated)
            for index, name in enumerate(
                ("position_rad", "velocity_rad_s", "acceleration_rad_s2")
            )
        },
        "max_joint_displacement_l2_rad": float(
            torch.linalg.vector_norm(q - q[0], dim=1).max().item()
        ),
        "max_abs_velocity_per_joint_rad_s": dq.abs().amax(dim=0).cpu().tolist(),
        "max_abs_acceleration_per_joint_rad_s2": ddq.abs().amax(dim=0).cpu().tolist(),
        "max_abs_jerk_per_joint_rad_s3": jerk.abs().amax(dim=0).cpu().tolist(),
        "joint_velocity_limit_rad_s": nominal_velocity_limit.cpu().tolist(),
        "joint_acceleration_limit_rad_s2": nominal_acceleration_limit.cpu().tolist(),
        "joint_jerk_limit_rad_s3": nominal_jerk_limit.cpu().tolist(),
        "max_velocity_limit_ratio": float(velocity_ratio.max().item()),
        "max_acceleration_limit_ratio": float(acceleration_ratio.max().item()),
        "max_jerk_limit_ratio": float(jerk_ratio.max().item()),
        "tool_position_error_m": {
            "mean": float(errors.mean().item()),
            "p95": float(torch.quantile(errors, 0.95).item()),
            "maximum": float(errors.max().item()),
            "final": float(errors[-1].item()),
            "completed_segment_end_median": (
                float(torch.median(completed_errors).item())
                if len(completed_errors)
                else None
            ),
            "completed_segment_end_maximum": (
                float(completed_errors.max().item()) if len(completed_errors) else None
            ),
        },
        "collision_validation_note": (
            "cuRobo scene/self constraints are mandatory at every 5 ms sample; "
            "cspace acceptance follows the application policy; continuous swept "
            "collision between samples is not claimed"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"trajectory: {output / 'trajectory.csv'}")


if __name__ == "__main__":
    main()
