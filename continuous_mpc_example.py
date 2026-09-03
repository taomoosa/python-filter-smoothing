"""Minimal feedforward example for the continuous cuRobo MPC generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from curobo.types import Pose

from python_filter_smoothing.continuous_trajectory import ContinuousMpcTrajectory

DEFAULT_CONFIG = (
    Path(__file__).parent / "python_filter_smoothing/configs/continuous_mpc.yml"
)


def _rot6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Convert Zhou 6D rotations (first two matrix columns) to SO(3)."""

    if rotation_6d.ndim != 2 or rotation_6d.shape[1] != 6:
        raise ValueError("rotation 6D values must have shape [N, 6]")
    if not bool(torch.all(torch.isfinite(rotation_6d)).item()):
        raise ValueError("rotation 6D values must be finite")
    first, second = rotation_6d[:, :3], rotation_6d[:, 3:]
    first_norm = torch.linalg.vector_norm(first, dim=1, keepdim=True)
    x_axis = first / torch.clamp(first_norm, min=1.0e-8)
    second_orthogonal = (
        second - torch.sum(x_axis * second, dim=1, keepdim=True) * x_axis
    )
    second_norm = torch.linalg.vector_norm(second_orthogonal, dim=1, keepdim=True)
    if bool(torch.any(first_norm < 1.0e-6).item()) or bool(
        torch.any(second_norm < 1.0e-6).item()
    ):
        raise ValueError("rotation 6D columns must be nonzero and non-collinear")
    y_axis = second_orthogonal / second_norm
    z_axis = torch.linalg.cross(x_axis, y_axis, dim=1)
    return torch.stack((x_axis, y_axis, z_axis), dim=2)


def _relative_rot6d_to_quaternion(
    base_rotation: torch.Tensor, rotation_6d: torch.Tensor
) -> torch.Tensor:
    """Apply tool-local 6D rotation offsets and return wxyz quaternions."""

    rotation = base_rotation.reshape(1, 3, 3) @ _rot6d_to_matrix(rotation_6d)
    matrix = torch.eye(4, device=rotation.device, dtype=rotation.dtype).repeat(
        len(rotation), 1, 1
    )
    matrix[:, :3, :3] = rotation
    quaternion = Pose.from_matrix(matrix).quaternion
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Pose conversion did not return a quaternion")
    return quaternion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--duration", type=float, help="override example.duration_s")
    args = parser.parse_args()

    mpc = ContinuousMpcTrajectory(args.config)
    config = mpc.config
    example = config["example"]
    state = (
        mpc.default_state()
    )  # No feedback: subsequent states come from MPC predictions.
    mpc.setup(state)

    base_pose = mpc.solver.compute_kinematics(state).tool_poses.to_dict()[
        mpc.solver.tool_frames[0]
    ]
    base = base_pose.position.clone()
    base_rotation = Pose(
        position=base_pose.position, quaternion=base_pose.quaternion
    ).get_rotation()
    if not isinstance(base_rotation, torch.Tensor):
        raise TypeError("initial tool pose is missing rotation")
    offsets = torch.as_tensor(
        example["target_offsets_m"], device=base.device, dtype=base.dtype
    )
    if offsets.ndim != 2 or offsets.shape[1] != 3 or len(offsets) == 0:
        raise ValueError("target_offsets_m must be a non-empty N x 3 array")
    if not bool(torch.all(torch.isfinite(offsets)).item()):
        raise ValueError("target_offsets_m must be finite")
    identity_rot6d = torch.as_tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=base.device, dtype=base.dtype
    )
    rotation_offsets = torch.as_tensor(
        example["target_rotation_offsets_rot6d"],
        device=base.device,
        dtype=base.dtype,
    )
    if rotation_offsets.ndim == 1:
        rotation_offsets = rotation_offsets.reshape(1, 6).expand(len(offsets), 6)
    if rotation_offsets.shape != (len(offsets), 6):
        raise ValueError("target_rotation_offsets_rot6d must have shape [N, 6]")
    target_quaternions = _relative_rot6d_to_quaternion(base_rotation, rotation_offsets)
    period = mpc.timing.mpc_period_s
    target_steps = round(float(example["target_period_s"]) / period)
    duration_s = float(
        args.duration if args.duration is not None else example["duration_s"]
    )
    steps = round(duration_s / period)
    if steps < 1 or not np.isclose(steps * period, duration_s):
        raise ValueError("duration must be a positive multiple of mpc_period_s")
    if target_steps < 1 or not np.isclose(
        target_steps * period, example["target_period_s"]
    ):
        raise ValueError("target_period_s must be an integer multiple of mpc_period_s")
    lookahead_steps = round(float(example["target_lookahead_s"]) / period)
    if lookahead_steps < 0 or not np.isclose(
        lookahead_steps * period, float(example["target_lookahead_s"])
    ):
        raise ValueError(
            "target_lookahead_s must be a nonnegative multiple of mpc_period_s"
        )
    target_path = torch.cat((torch.zeros_like(offsets[:1]), offsets, offsets[:1]))
    target_step_distances = torch.linalg.vector_norm(
        target_path[1:] - target_path[:-1], dim=1
    )
    max_target_step_m = float(example["max_target_step_m"])
    if float(target_step_distances.max().item()) > max_target_step_m + 1.0e-6:
        raise ValueError("adjacent target offset exceeds max_target_step_m")

    transition = mpc.solver.transition_model
    max_velocity = transition.max_velocity
    max_acceleration = transition.max_acceleration
    max_jerk = transition.max_jerk
    target_period_s = target_steps * period
    # Sufficient rest-to-rest quintic displacement bounds for one target interval.
    dynamic_joint_step = torch.minimum(
        max_velocity * target_period_s / 1.875,
        torch.minimum(
            max_acceleration * target_period_s**2 / 5.7735026919,
            max_jerk * target_period_s**3 / 60.0,
        ),
    )

    validate_ik = bool(example["validate_targets_with_ik"])
    max_ik_joint_step = None
    if validate_ik:
        ik_positions = []
        for target_id, (offset, quaternion) in enumerate(
            zip(offsets, target_quaternions, strict=True)
        ):
            try:
                ik_state = mpc.set_target(base + offset, quaternion, validate_ik=True)
            except RuntimeError as error:
                raise RuntimeError(
                    f"IK validation failed for target {target_id}: {offset.tolist()}"
                ) from error
            if ik_state is None or not isinstance(ik_state.position, torch.Tensor):
                raise RuntimeError("IK validation did not return a joint solution")
            ik_positions.append(ik_state.position)
        if not isinstance(state.position, torch.Tensor):
            raise TypeError("initial state is missing position")
        ik_path = torch.cat((state.position, *ik_positions, ik_positions[0]), dim=0)
        ik_joint_steps = (ik_path[1:] - ik_path[:-1]).abs()
        max_ik_joint_step = ik_joint_steps.amax(dim=0)
        if bool(torch.any(ik_joint_steps > dynamic_joint_step + 1.0e-6).item()):
            raise ValueError(
                "adjacent IK targets exceed the configured velocity/acceleration/jerk "
                "step bound"
            )

    positions, velocities, accelerations, target_ids = [], [], [], []
    command_targets = []
    command_target_quaternions = []
    solve_times: list[float] = []
    boundary_position_errors: list[float] = []
    boundary_velocity_errors: list[float] = []
    boundary_acceleration_errors: list[float] = []
    prefix_only_acceptances = 0
    feasible_tail_fallbacks = 0
    command_limit_rejections = 0
    window_records: list[list[float]] = []

    def target_at_step(
        target_step: int,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        segment = target_step // target_steps
        target_id = segment % len(offsets)
        if bool(example["smooth_target_transitions"]):
            start_offset = (
                torch.zeros_like(offsets[0])
                if segment == 0
                else offsets[(target_id - 1) % len(offsets)]
            )
            u = (target_step % target_steps + 1) / target_steps
            blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
            target_offset = start_offset + blend * (offsets[target_id] - start_offset)
            start_rotation = (
                identity_rot6d
                if segment == 0
                else rotation_offsets[(target_id - 1) % len(offsets)]
            )
            target_rotation = start_rotation + blend * (
                rotation_offsets[target_id] - start_rotation
            )
        else:
            target_offset = offsets[target_id]
            target_rotation = rotation_offsets[target_id]
        quaternion = _relative_rot6d_to_quaternion(
            base_rotation, target_rotation.reshape(1, 6)
        )[0]
        return target_id, base + target_offset, quaternion

    for step in range(steps):
        target_id, command_target, command_quaternion = target_at_step(step)
        _, solver_target, solver_quaternion = target_at_step(step + lookahead_steps)
        mpc.set_target(solver_target, solver_quaternion)
        try:
            window = mpc.step()
        except RuntimeError as error:
            raise RuntimeError(
                f"MPC failed at step {step} (t={step * period:.3f} s, "
                f"target={target_id})"
            ) from error
        positions.append(window.commands.position.squeeze(0))
        velocities.append(window.commands.velocity.squeeze(0))
        accelerations.append(window.commands.acceleration.squeeze(0))
        target_ids.extend([target_id] * window.commands.position.shape[1])
        command_targets.append(
            command_target.reshape(1, 3).expand(window.commands.position.shape[1], 3)
        )
        command_target_quaternions.append(
            command_quaternion.reshape(1, 4).expand(
                window.commands.position.shape[1], 4
            )
        )
        solve_times.append(window.wall_time_s)
        prefix_only_acceptances += int(
            not window.full_horizon_feasible and not window.used_feasible_tail_fallback
        )
        feasible_tail_fallbacks += int(window.used_feasible_tail_fallback)
        command_limit_rejections += int(window.rejected_by_command_limits)
        window_records.append(
            [
                step * period,
                float(target_id),
                window.wall_time_s,
                float(window.full_horizon_feasible),
                float(window.rejected_by_command_limits),
                float(window.used_feasible_tail_fallback),
                window.initial_position_error_rad,
                window.initial_velocity_error_rad_s,
                window.initial_acceleration_error_rad_s2,
                window.position_boundary_error_rad or 0.0,
                window.velocity_boundary_error_rad_s or 0.0,
                window.acceleration_boundary_error_rad_s2 or 0.0,
            ]
        )
        if window.position_boundary_error_rad is not None:
            boundary_position_errors.append(window.position_boundary_error_rad)
            boundary_velocity_errors.append(window.velocity_boundary_error_rad_s or 0.0)
            boundary_acceleration_errors.append(
                window.acceleration_boundary_error_rad_s2 or 0.0
            )

    q, dq, ddq = map(torch.cat, (positions, velocities, accelerations))
    dt = mpc.timing.command_dt_s
    jerk = torch.empty_like(ddq)
    jerk[1:] = (ddq[1:] - ddq[:-1]) / dt
    jerk[0] = jerk[1]
    time_s = torch.arange(len(q), device=q.device, dtype=q.dtype) * dt

    target_position = torch.cat(command_targets)
    target_quaternion = torch.cat(command_target_quaternions)
    output = args.output or (args.config.resolve().parent / example["output_directory"])
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    columns = (
        ["time_s", "target_index"]
        + [
            f"{field}_{name}"
            for field in ("q_rad", "dq_rad_s", "ddq_rad_s2", "jerk_rad_s3")
            for name in mpc.joint_names
        ]
        + [
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
                torch.as_tensor(target_ids, device=q.device, dtype=q.dtype)[:, None],
                q,
                dq,
                ddq,
                jerk,
                target_position,
                target_quaternion,
            ),
            dim=1,
        )
        .cpu()
        .numpy()
    )
    np.savetxt(
        output / "trajectory.csv", values, delimiter=",", header=",".join(columns)
    )
    window_columns = [
        "time_s",
        "target_index",
        "wall_time_s",
        "full_horizon_feasible",
        "rejected_by_command_limits",
        "used_feasible_tail_fallback",
        "initial_position_error_rad",
        "initial_velocity_error_rad_s",
        "initial_acceleration_error_rad_s2",
        "position_boundary_error_rad",
        "velocity_boundary_error_rad_s",
        "acceleration_boundary_error_rad_s2",
    ]
    np.savetxt(
        output / "mpc_windows.csv",
        np.asarray(window_records),
        delimiter=",",
        header=",".join(window_columns),
    )

    summary = {
        "samples": len(q),
        "duration_s": len(q) * dt,
        "command_dt_s": dt,
        "joint_names": mpc.joint_names,
        "robot": config["robot"]["config"],
        "scene_model": config["scene"],
        "target_offsets_m": example["target_offsets_m"],
        "target_rotation_offsets_rot6d": rotation_offsets.cpu().tolist(),
        "rotation_6d_convention": (
            "Zhou first-two-columns; tool-local relative rotation"
        ),
        "targets_ik_validated": validate_ik,
        "max_adjacent_target_step_m": float(target_step_distances.max().item()),
        "configured_max_target_step_m": max_target_step_m,
        "max_actual_target_step_m": float(
            torch.linalg.vector_norm(target_position[1:] - target_position[:-1], dim=1)
            .max()
            .item()
        ),
        "smooth_target_transitions": bool(example["smooth_target_transitions"]),
        "target_lookahead_s": lookahead_steps * period,
        "quintic_joint_step_limit_per_joint_rad": dynamic_joint_step.cpu().tolist(),
        "max_adjacent_ik_joint_step_per_joint_rad": (
            max_ik_joint_step.cpu().tolist() if max_ik_joint_step is not None else None
        ),
        "target_changes": len(set(target_ids)),
        "target_segments": (steps + target_steps - 1) // target_steps,
        "max_joint_displacement_l2_rad": float(
            torch.linalg.vector_norm(q - q[0], dim=1).max().item()
        ),
        "max_abs_velocity_per_joint_rad_s": dq.abs().amax(dim=0).cpu().tolist(),
        "max_abs_acceleration_per_joint_rad_s2": ddq.abs().amax(dim=0).cpu().tolist(),
        "max_abs_jerk_per_joint_rad_s3": jerk.abs().amax(dim=0).cpu().tolist(),
        "joint_velocity_limit_rad_s": max_velocity.cpu().tolist(),
        "joint_acceleration_limit_rad_s2": max_acceleration.cpu().tolist(),
        "joint_jerk_limit_rad_s3": max_jerk.cpu().tolist(),
        "command_acceptance_velocity_limit_rad_s": (
            mpc.command_limits.velocity.cpu().tolist()
        ),
        "command_acceptance_acceleration_limit_rad_s2": (
            mpc.command_limits.acceleration.cpu().tolist()
        ),
        "command_acceptance_jerk_limit_rad_s3": (
            mpc.command_limits.jerk.cpu().tolist()
        ),
        "max_position_boundary_error_rad": max(boundary_position_errors, default=0.0),
        "max_velocity_boundary_error_rad_s": max(boundary_velocity_errors, default=0.0),
        "max_acceleration_boundary_error_rad_s2": max(
            boundary_acceleration_errors, default=0.0
        ),
        "mpc_deadline_misses": sum(value > mpc.deadline_s for value in solve_times),
        "prefix_only_acceptances": prefix_only_acceptances,
        "feasible_tail_fallbacks": feasible_tail_fallbacks,
        "command_limit_rejections": command_limit_rejections,
        "command_limit_rejection_times_s": [
            record[0] for record in window_records if record[4]
        ],
        "feasible_tail_fallback_times_s": [
            record[0] for record in window_records if record[5]
        ],
        "full_horizon_feasible_solves": (
            steps - prefix_only_acceptances - feasible_tail_fallbacks
        ),
        "executed_prefix_constraints_verified": True,
        "max_mpc_wall_time_s": max(solve_times),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"trajectory: {output / 'trajectory.csv'}")


if __name__ == "__main__":
    main()
