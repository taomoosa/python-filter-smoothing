"""Generate servo commands from long cuRobo MPC paths."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from curobo.config_io import load_yaml
from curobo.types import JointState, Pose, RobotState

from python_filter_smoothing.continuous_trajectory import ContinuousMpcTrajectory
from python_filter_smoothing.pose_utils import relative_rot6d_to_quaternion
from python_filter_smoothing.predictive_mpc import MpcHorizon
from python_filter_smoothing.servo_queue import (
    ServoTrajectoryQueue,
    extend_stationary_trajectory,
    slice_joint_trajectory,
)
from python_filter_smoothing.trajectory_resampling import (
    ResampledJointTrajectory,
    resample_joint_trajectory,
)

DEFAULT_CONFIG = (
    Path(__file__).parent
    / "python_filter_smoothing/configs/long_mpc_application.yml"
)


def _config_path(value: str, parent: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (parent / path).resolve()


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


@dataclass(frozen=True)
class _RawCandidate:
    iterations: int
    horizon: MpcHorizon
    position_error_m: float
    rotation_error_rad: float


def _rank_feasible_candidates(
    candidates: list[_RawCandidate],
) -> list[_RawCandidate]:
    """Keep feasible results even when a later solve regresses."""

    return sorted(
        (
            candidate
            for candidate in candidates
            if candidate.horizon.full_horizon_feasible
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
    state.velocity = torch.as_tensor(
        trajectory.velocity, device=device, dtype=dtype
    )[None]
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
    controller: ContinuousMpcTrajectory, state: JointState
) -> tuple[bool, dict[str, int], dict[str, float]]:
    """Evaluate cspace, self-collision, and scene constraints at every sample."""

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
        feasible = metrics.feasible
        if isinstance(feasible, torch.Tensor):
            all_feasible &= bool(torch.all(feasible[:, :count]).item())
        else:
            all_feasible &= bool(feasible)
        constraints = metrics.costs_and_constraints.constraints
        for name, value in zip(constraints.names, constraints.values, strict=True):
            selected = value[:, :count]
            violations[name] = violations.get(name, 0) + int(
                torch.count_nonzero(selected > 0.0).item()
            )
            maxima[name] = max(maxima.get(name, 0.0), float(selected.max().item()))
    return all_feasible, violations, maxima


def _targets(
    controller: ContinuousMpcTrajectory, options: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    initial = controller.default_state()
    pose = controller.solver.compute_kinematics(initial).tool_poses.to_dict()[
        controller.solver.tool_frames[0]
    ]
    rotation = Pose(position=pose.position, quaternion=pose.quaternion).get_rotation()
    if not isinstance(rotation, torch.Tensor):
        raise TypeError("initial tool pose is missing rotation")
    offsets = torch.as_tensor(
        options["target_offsets_m"], device=pose.position.device, dtype=pose.position.dtype
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


def _candidate_error(
    controller: ContinuousMpcTrajectory,
    horizon: MpcHorizon,
    target_position: torch.Tensor,
    target_quaternion: torch.Tensor,
) -> tuple[float, float]:
    endpoint = JointState.from_position(
        horizon.states.position[:, -1], joint_names=controller.joint_names
    )
    pose = controller.solver.compute_kinematics(endpoint).tool_poses.to_dict()[
        controller.solver.tool_frames[0]
    ]
    position_error = float(
        torch.linalg.vector_norm(pose.position.reshape(-1, 3)[0] - target_position).item()
    )
    actual_quaternion = pose.quaternion.reshape(-1, 4)[0]
    quaternion_dot = torch.abs(torch.dot(actual_quaternion, target_quaternion))
    rotation_error = float(
        (2.0 * torch.acos(torch.clamp(quaternion_dot, 0.0, 1.0))).item()
    )
    return position_error, rotation_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_yaml(str(config_path))
    options = config["example"]
    resampling_options = config["resampling"]
    controller = ContinuousMpcTrajectory(
        _config_path(config["long_mpc_config"], config_path.parent)
    )
    initial = controller.default_state()
    controller.setup(initial)
    target_position, target_quaternion, offsets = _targets(controller, options)

    servo_dt = float(options["servo_dt_s"])
    duration_s = float(args.duration or options["duration_s"])
    total_samples = round(duration_s / servo_dt)
    if total_samples < 1 or not math.isclose(total_samples * servo_dt, duration_s):
        raise ValueError("duration must be a positive multiple of servo_dt_s")
    target_period = float(options["target_period_s"])
    target_samples = round(target_period / servo_dt)
    if target_samples < 1 or not math.isclose(target_samples * servo_dt, target_period):
        raise ValueError("target_period_s must be a positive servo multiple")
    connection_delay = float(options["planning_connection_delay_s"])
    connection_samples = round(connection_delay / servo_dt)
    if connection_samples < 1 or not math.isclose(
        connection_samples * servo_dt, connection_delay
    ):
        raise ValueError("planning_connection_delay_s must be a servo multiple")

    transition = controller.solver.transition_model
    bounds = transition.get_state_bounds()
    velocity_limit = transition.max_velocity * float(
        resampling_options["velocity_limit_scale"]
    )
    acceleration_limit = transition.max_acceleration * float(
        resampling_options["acceleration_limit_scale"]
    )
    jerk_limit = transition.max_jerk * float(
        resampling_options["jerk_limit_scale"]
    )
    numpy_limits = (
        velocity_limit.detach().cpu().numpy(),
        acceleration_limit.detach().cpu().numpy(),
        jerk_limit.detach().cpu().numpy(),
        bounds.position[0].detach().cpu().numpy(),
        bounds.position[1].detach().cpu().numpy(),
    )

    def generate(target_index: int, start: JointState) -> _GeneratedPath:
        controller.setup(start)
        controller.set_target(
            target_position[target_index], target_quaternion[target_index]
        )
        candidates: list[_RawCandidate] = []
        for index, iterations in enumerate(controller.candidate_iterations):
            if index:
                controller.prepare_candidate(start, iterations)
            horizon = controller.solve_horizon()
            position_error, rotation_error = _candidate_error(
                controller,
                horizon,
                target_position[target_index],
                target_quaternion[target_index],
            )
            candidates.append(
                _RawCandidate(
                    iterations, horizon, position_error, rotation_error
                )
            )
        ranked = _rank_feasible_candidates(candidates)
        if not ranked:
            raise RuntimeError(
                f"long MPC target {target_index} produced no feasible candidate"
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
            nodes = [
                torch.cat(
                    (getattr(start, name)[:, None], getattr(source, name)), dim=1
                )[0]
                .detach()
                .cpu()
                .numpy()
                for name in ("position", "velocity", "acceleration")
            ]
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
            valid, violations, maxima = _validate_with_curobo(controller, state)
            validation_wall = time.perf_counter() - validation_started
            if not valid:
                continue
            node_error = float(
                np.max(
                    np.abs(
                        resampled.position[resampled.node_sample_indices] - nodes[0]
                    )
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
                source_duration_s=(len(nodes[0]) - 1)
                * controller.timing.command_dt_s,
                initial_state_errors=initial_errors,
                raw_max_jerk_rad_s3=_maximum_jerk(
                    resampled.raw_acceleration, servo_dt
                ),
                filtered_max_jerk_rad_s3=_maximum_jerk(
                    resampled.acceleration, servo_dt
                ),
                filter_changes=changes,
                candidate_feasible=tuple(
                    item.horizon.full_horizon_feasible for item in candidates
                ),
                selected_candidate_iterations=selected.iterations,
            )
        raise RuntimeError(
            f"long MPC target {target_index} had no candidate passing final validation"
        )

    startup_started = time.perf_counter()
    startup = generate(0, initial)
    startup_wall = time.perf_counter() - startup_started
    queue = ServoTrajectoryQueue(startup.state)

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
    deadline_misses = 0

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
            last_attempt = target_index
            start = queue.at_offset(connection_samples)
            application_started = time.perf_counter()
            candidate = None
            try:
                candidate = generate(target_index, start)
            except RuntimeError:
                infeasible_updates += 1
            application_wall = time.perf_counter() - application_started
            elapsed_samples = max(1, math.ceil(application_wall / servo_dt))
            on_time = elapsed_samples <= connection_samples
            accepted = bool(candidate is not None and on_time)
            consume_count = connection_samples if accepted else elapsed_samples
            append(queue.consume(min(consume_count, total_samples - produced)))
            if accepted and produced < total_samples:
                queue.replace(candidate.state)
                generated.append(candidate)
                accepted_updates += 1
            deadline_misses += int(not on_time)
            update_records.append(
                [
                    produced * servo_dt,
                    float(target_index),
                    candidate.mpc_wall_time_s if candidate else math.nan,
                    candidate.selected_candidate_iterations
                    if candidate
                    else math.nan,
                    float(sum(candidate.candidate_feasible)) if candidate else 0.0,
                    candidate.resampled.interpolation_wall_time_s
                    if candidate
                    else math.nan,
                    candidate.resampled.filter_wall_time_s
                    if candidate
                    else math.nan,
                    candidate.validation_wall_time_s if candidate else math.nan,
                    application_wall,
                    float(elapsed_samples),
                    float(on_time),
                    float(candidate is not None),
                    float(accepted),
                    candidate.resampled.duration_s if candidate else math.nan,
                ]
            )
            continue
        boundary = min(total_samples, ((produced // target_samples) + 1) * target_samples)
        append(queue.consume(boundary - produced))

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

    output = args.output or _config_path(options["output_directory"], config_path.parent)
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
    values = torch.cat(
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
    ).cpu().numpy()
    np.savetxt(output / "trajectory.csv", values, delimiter=",", header=",".join(columns))
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
    ]
    np.savetxt(
        output / "planner_updates.csv",
        np.asarray(update_records),
        delimiter=",",
        header=",".join(update_columns),
    )

    velocity_ratio = dq.abs() / velocity_limit
    acceleration_ratio = ddq.abs() / acceleration_limit
    jerk_ratio = jerk.abs() / jerk_limit
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
            "setup_cold_iterations": controller.config["optimizer"]["cold_start_iterations"],
            "candidate_iterations": list(controller.candidate_iterations),
            "warm_iterations": controller.config["optimizer"]["warm_start_iterations"],
            "fixed_iterations": controller.config["optimizer"]["fixed_iterations"],
            "return_best_action": controller.config["optimizer"]["return_best_action"],
            "use_ik_joint_reference": controller.config["optimizer"]["target_update"]["use_ik_joint_reference"],
            "seed_from_ik": controller.config["optimizer"]["target_update"]["seed_from_ik"],
            "ik_fallback_seeds": controller.config["optimizer"]["target_update"]["ik_fallback_seeds"],
        },
        "joint_names": controller.joint_names,
        "robot": controller.config["robot"]["config"],
        "scene_model": controller.config["scene"],
        "target_offsets_m": offsets.cpu().tolist(),
        "target_period_s": target_period,
        "planning_connection_delay_s": connection_delay,
        "long_mpc_dt_s": controller.timing.command_dt_s,
        "resampling": resampling_options,
        "startup_application_wall_time_s": startup_wall,
        "accepted_long_paths": accepted_updates,
        "infeasible_long_updates": infeasible_updates,
        "planning_deadline_misses": deadline_misses,
        "selected_candidate_iterations": [
            item.selected_candidate_iterations for item in generated
        ],
        "final_candidate_infeasible_paths": sum(
            not item.candidate_feasible[-1] for item in generated
        ),
        "all_accepted_paths_curobo_constraint_feasible": True,
        "curobo_constraint_violations_by_component": {
            name: sum(item.constraint_violations.get(name, 0) for item in generated)
            for name in sorted({name for item in generated for name in item.constraint_violations})
        },
        "curobo_maximum_constraint_by_component": {
            name: max(item.maximum_constraint_values.get(name, 0.0) for item in generated)
            for name in sorted({name for item in generated for name in item.maximum_constraint_values})
        },
        "max_filtered_long_mpc_node_position_error_rad": max(
            item.maximum_node_error_rad for item in generated
        ),
        "max_initial_long_state_error": {
            name: max(item.initial_state_errors[index] for item in generated)
            for index, name in enumerate(("position_rad", "velocity_rad_s", "acceleration_rad_s2"))
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
            for index, name in enumerate(("position_rad", "velocity_rad_s", "acceleration_rad_s2"))
        },
        "max_joint_displacement_l2_rad": float(
            torch.linalg.vector_norm(q - q[0], dim=1).max().item()
        ),
        "max_abs_velocity_per_joint_rad_s": dq.abs().amax(dim=0).cpu().tolist(),
        "max_abs_acceleration_per_joint_rad_s2": ddq.abs().amax(dim=0).cpu().tolist(),
        "max_abs_jerk_per_joint_rad_s3": jerk.abs().amax(dim=0).cpu().tolist(),
        "joint_velocity_limit_rad_s": velocity_limit.cpu().tolist(),
        "joint_acceleration_limit_rad_s2": acceleration_limit.cpu().tolist(),
        "joint_jerk_limit_rad_s3": jerk_limit.cpu().tolist(),
        "max_velocity_limit_ratio": float(velocity_ratio.max().item()),
        "max_acceleration_limit_ratio": float(acceleration_ratio.max().item()),
        "max_jerk_limit_ratio": float(jerk_ratio.max().item()),
        "tool_position_error_m": {
            "mean": float(errors.mean().item()),
            "p95": float(torch.quantile(errors, 0.95).item()),
            "maximum": float(errors.max().item()),
            "final": float(errors[-1].item()),
            "completed_segment_end_median": (
                float(torch.median(completed_errors).item()) if len(completed_errors) else None
            ),
            "completed_segment_end_maximum": (
                float(completed_errors.max().item()) if len(completed_errors) else None
            ),
        },
        "collision_validation_note": (
            "cuRobo scene/self/cspace constraints evaluated at every 5 ms sample; "
            "continuous swept collision between samples is not claimed"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"trajectory: {output / 'trajectory.csv'}")


if __name__ == "__main__":
    main()
