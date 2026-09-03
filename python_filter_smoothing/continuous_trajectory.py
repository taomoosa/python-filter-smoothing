"""Minimal continuous cuRobo MPC trajectory generator; simulation is external."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from curobo.config_io import load_yaml
from curobo.content import (
    get_robot_configs_path,
    get_scene_configs_path,
    get_task_configs_path,
)
from curobo.model_predictive_control import (
    ModelPredictiveControl,
    ModelPredictiveControlCfg,
)
from curobo.types import GoalToolPose, JointState

from .predictive_mpc import (
    MpcCommandLimits,
    MpcCommandWindow,
    PredictedStateMpc,
    PredictiveMpcTiming,
)


def _resolve_config_file(
    value: str, config_directory: Path, built_in_directory: Path
) -> Path:
    path = Path(value)
    candidates = (
        (path,)
        if path.is_absolute()
        else (config_directory / path, built_in_directory / path)
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"configuration file not found: {value}")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and resolve local or cuRobo-provided robot/world files."""

    config_path = Path(path).resolve()
    config = load_yaml(str(config_path))
    config["robot"]["config"] = str(
        _resolve_config_file(
            config["robot"]["config"],
            config_path.parent,
            get_robot_configs_path(),
        )
    )
    config["scene"] = str(
        _resolve_config_file(
            config["scene"], config_path.parent, get_scene_configs_path()
        )
    )
    config["optimizer"]["base_config"] = str(
        _resolve_config_file(
            config["optimizer"]["base_config"],
            config_path.parent,
            get_task_configs_path(),
        )
    )
    return config


def _solver(config: dict[str, Any]) -> ModelPredictiveControl:
    robot_options = config["robot"]
    robot = load_yaml(robot_options["config"])
    cspace = robot["robot_cfg"]["kinematics"]["cspace"]
    cspace["velocity_scale"] = robot_options["velocity_limit_scale"]
    cspace["acceleration_scale"] = robot_options["acceleration_limit_scale"]
    cspace["jerk_scale"] = robot_options["jerk_limit_scale"]

    optimizer_options = config["optimizer"]
    optimizer = load_yaml(optimizer_options["base_config"])
    costs = optimizer["rollout"]["cost_cfg"]
    costs["tool_pose_cfg"]["weight"] = optimizer_options["tool_pose_weight"]
    costs["cspace_cfg"]["weight"] = optimizer_options["cspace_bound_weight"]

    timing = config["timing"]
    collision = config["collision"]
    runtime = config["runtime"]
    solver_config = ModelPredictiveControlCfg.create(
        robot=robot,
        optimizer_configs=[optimizer],
        scene_model=load_yaml(config["scene"]),
        optimization_dt=timing["optimization_dt_s"],
        interpolation_steps=timing["interpolation_steps"],
        num_control_points=timing["control_points"],
        squared_l2_regularization_weight=optimizer_options[
            "squared_l2_regularization_weight"
        ],
        non_terminal_tool_pose_weight_factor=optimizer_options[
            "non_terminal_tool_pose_weight_factor"
        ],
        warm_start_optimization_num_iters=optimizer_options["warm_start_iterations"],
        cold_start_optimization_num_iters=optimizer_options["cold_start_iterations"],
        optimizer_collision_activation_distance=collision["activation_distance_m"],
        self_collision_check=collision["self_collision_check"],
        use_cuda_graph=runtime["use_cuda_graph"],
        random_seed=runtime["random_seed"],
    )
    return ModelPredictiveControl(solver_config)


class ContinuousMpcTrajectory:
    """Convert successive Cartesian goals into continuous q/dq/ddq windows."""

    def __init__(self, config_path: str | Path) -> None:
        self.config = load_config(config_path)
        timing = self.config["timing"]
        mpc_period = float(timing["mpc_period_s"])
        optimization_dt = float(timing["optimization_dt_s"])
        self.timing = PredictiveMpcTiming(
            mpc_period_s=mpc_period,
            optimization_dt_s=optimization_dt,
            interpolation_steps=int(timing["interpolation_steps"]),
            required_feasible_windows=int(timing["required_feasible_windows"]),
        )
        self.deadline_s = float(timing["deadline_s"])
        self.solver = _solver(self.config)
        acceptance = self.config["command_acceptance"]
        acceptance_scales = [
            float(acceptance[name])
            for name in (
                "velocity_limit_scale",
                "acceleration_limit_scale",
                "jerk_limit_scale",
            )
        ]
        if any(not math.isfinite(scale) or scale <= 0.0 for scale in acceptance_scales):
            raise ValueError("command acceptance scales must be finite and positive")
        transition = self.solver.transition_model
        self.command_limits = MpcCommandLimits(
            velocity=transition.max_velocity * acceptance_scales[0],
            acceleration=transition.max_acceleration * acceptance_scales[1],
            jerk=transition.max_jerk * acceptance_scales[2],
        )
        self._planner = PredictedStateMpc(
            self.solver,
            self.timing,
            command_limits=self.command_limits,
        )
        self._goal_request: GoalToolPose | None = None
        self._ik_reference_state: JointState | None = None

    @property
    def joint_names(self) -> list[str]:
        return list(self.solver.joint_names)

    def default_state(self) -> JointState:
        """Return a finite q/dq/ddq state suitable for initial setup."""

        return JointState.from_position(
            self.solver.default_joint_position.clone().unsqueeze(0),
            joint_names=self.joint_names,
        )

    def setup(self, initial_state: JointState) -> None:
        """Initialize from a measured state before the periodic loop starts."""

        self._planner.setup(initial_state)
        self._ik_reference_state = initial_state
        initial_poses = self.solver.compute_kinematics(
            initial_state
        ).tool_poses.to_dict()
        self._goal_request = GoalToolPose.from_poses(
            initial_poses,
            ordered_tool_frames=self.solver.tool_frames,
            num_goalset=1,
        )

    def set_target(
        self,
        position_m: torch.Tensor,
        quaternion_wxyz: torch.Tensor | None = None,
        *,
        validate_ik: bool = False,
    ) -> JointState | None:
        """Update the goal, or validate and return its IK solution without updating."""

        if self._goal_request is None:
            raise RuntimeError("setup() must be called before set_target()")
        current = (
            self._ik_reference_state if validate_ik else self._planner.current_state
        )
        if current is None:
            raise RuntimeError("setup() must be called before set_target()")
        goal = self._goal_request
        goal.position[:, :, 0, :, :].copy_(position_m.reshape(1, 1, 1, 3))
        if quaternion_wxyz is not None:
            goal.quaternion[:, :, 0, :, :].copy_(quaternion_wxyz.reshape(1, 1, 1, 4))
        if validate_ik:
            if not isinstance(current.position, torch.Tensor):
                raise TypeError("current MPC state is missing position")
            result = self.solver.ik_solver.solve_pose(
                goal_tool_poses=goal,
                current_state=current,
                seed_config=current.position.reshape(1, 1, -1).clone(),
                return_seeds=1,
            )
            if not bool(torch.all(result.success).item()):
                position_error = float(result.position_error.min().item())
                rotation_error = float(result.rotation_error.min().item())
                raise RuntimeError(
                    "Cartesian goal IK validation failed "
                    f"(position_error={position_error:.6g} m, "
                    f"rotation_error={rotation_error:.6g} rad)"
                )
            return JointState.from_position(
                result.solution.reshape(-1, len(self.joint_names))[:1].clone(),
                joint_names=self.joint_names,
            )
        if not self.solver.update_goal_tool_poses(goal, run_ik=False):
            raise RuntimeError("Cartesian goal update failed")
        return None

    def step(self) -> MpcCommandWindow:
        """Return one MPC period of continuous commands."""

        return self._planner.step()
