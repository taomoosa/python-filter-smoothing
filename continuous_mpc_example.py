"""Minimal feedforward example for the continuous cuRobo MPC generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from python_filter_smoothing.continuous_trajectory import (
    ContinuousMpcTrajectory,
    load_config,
)

DEFAULT_CONFIG = (
    Path(__file__).parent / "python_filter_smoothing/configs/continuous_mpc.yml"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    example = config["example"]
    mpc = ContinuousMpcTrajectory(args.config)
    state = (
        mpc.default_state()
    )  # No feedback: subsequent states come from MPC predictions.
    mpc.setup(state)

    base = (
        mpc.solver.compute_kinematics(state)
        .tool_poses.to_dict()[mpc.solver.tool_frames[0]]
        .position
    )
    offsets = torch.as_tensor(
        example["target_offsets_m"], device=base.device, dtype=base.dtype
    )
    period = mpc.timing.mpc_period_s
    target_steps = round(float(example["target_period_s"]) / period)
    steps = round(float(example["duration_s"]) / period)
    if target_steps < 1 or not np.isclose(
        target_steps * period, example["target_period_s"]
    ):
        raise ValueError("target_period_s must be an integer multiple of mpc_period_s")

    positions, velocities, accelerations, target_ids = [], [], [], []
    solve_times: list[float] = []
    for step in range(steps):
        target_id = (step // target_steps) % len(offsets)
        if step % target_steps == 0:
            mpc.set_target(base + offsets[target_id])
        window = mpc.step()
        positions.append(window.commands.position.squeeze(0))
        velocities.append(window.commands.velocity.squeeze(0))
        accelerations.append(window.commands.acceleration.squeeze(0))
        target_ids.extend([target_id] * window.commands.position.shape[1])
        solve_times.append(window.wall_time_s)

    q, dq, ddq = map(torch.cat, (positions, velocities, accelerations))
    dt = mpc.timing.command_dt_s
    jerk = torch.empty_like(ddq)
    jerk[1:] = (ddq[1:] - ddq[:-1]) / dt
    jerk[0] = jerk[1]
    time_s = torch.arange(len(q), device=q.device, dtype=q.dtype) * dt

    output = args.output or (args.config.resolve().parent / example["output_directory"])
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    columns = ["time_s", "target_index"] + [
        f"{field}_{name}"
        for field in ("q_rad", "dq_rad_s", "ddq_rad_s2", "jerk_rad_s3")
        for name in mpc.joint_names
    ]
    values = (
        torch.cat(
            (
                time_s[:, None],
                torch.as_tensor(target_ids, device=q.device, dtype=q.dtype)[:, None],
                q,
                dq,
                ddq,
                jerk,
            ),
            dim=1,
        )
        .cpu()
        .numpy()
    )
    np.savetxt(
        output / "trajectory.csv", values, delimiter=",", header=",".join(columns)
    )

    summary = {
        "samples": len(q),
        "duration_s": len(q) * dt,
        "command_dt_s": dt,
        "joint_names": mpc.joint_names,
        "target_offsets_m": example["target_offsets_m"],
        "target_changes": len(set(target_ids)),
        "max_joint_displacement_l2_rad": float(
            torch.linalg.vector_norm(q - q[0], dim=1).max().item()
        ),
        "max_abs_velocity_per_joint_rad_s": dq.abs().amax(dim=0).cpu().tolist(),
        "max_abs_acceleration_per_joint_rad_s2": ddq.abs().amax(dim=0).cpu().tolist(),
        "max_abs_jerk_per_joint_rad_s3": jerk.abs().amax(dim=0).cpu().tolist(),
        "max_mpc_wall_time_s": max(solve_times),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"trajectory: {output / 'trajectory.csv'}")


if __name__ == "__main__":
    main()
