"""Minimal continuous cuRobo MPC trajectory generator; simulation is external."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from curobo.config_io import load_yaml
from curobo.content import get_robot_configs_path, get_task_configs_path
from curobo.model_predictive_control import (
    ModelPredictiveControl,
    ModelPredictiveControlCfg,
)
from curobo.types import GoalToolPose, JointState

from mpc_app.predictive_mpc import (
    MpcCommandWindow,
    PredictedStateMpc,
    PredictiveMpcTiming,
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the application YAML and resolve its scene path."""

    config_path = Path(path).resolve()
    config = load_yaml(str(config_path))
    scene = Path(config["scene"])
    if not scene.is_absolute():
        scene = (config_path.parent / scene).resolve()
    config["scene"] = str(scene)
    return config


def _solver(config: dict[str, Any]) -> ModelPredictiveControl:
    robot_options = config["robot"]
    robot = load_yaml(str(get_robot_configs_path() / robot_options["config"]))
    cspace = robot["robot_cfg"]["kinematics"]["cspace"]
    cspace["velocity_scale"] = robot_options["velocity_limit_scale"]
    cspace["acceleration_scale"] = robot_options["acceleration_limit_scale"]
    cspace["jerk_scale"] = robot_options["jerk_limit_scale"]

    optimizer_options = config["optimizer"]
    optimizer = load_yaml(
        str(get_task_configs_path() / optimizer_options["base_config"])
    )
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
        servo_dt = float(timing["servo_dt_s"])
        optimization_divisor = round(mpc_period / optimization_dt)
        servo_substeps = round(mpc_period / servo_dt)
        if not math.isclose(mpc_period / optimization_divisor, optimization_dt):
            raise ValueError("optimization_dt_s must divide mpc_period_s")
        if not math.isclose(mpc_period / servo_substeps, servo_dt):
            raise ValueError("servo_dt_s must divide mpc_period_s")
        self.timing = PredictiveMpcTiming(
            mpc_period_s=mpc_period,
            interpolation_steps=int(timing["interpolation_steps"]),
            servo_substeps=servo_substeps,
            optimization_dt_divisor=optimization_divisor,
        )
        self.deadline_s = float(timing["deadline_s"])
        self.solver = _solver(self.config)
        self._planner = PredictedStateMpc(self.solver, self.timing)
        self._goal_poses: dict[str, Any] | None = None

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

        self._goal_poses = self.solver.compute_kinematics(
            initial_state
        ).tool_poses.to_dict()
        self._planner.setup(initial_state)

    def set_target(
        self,
        position_m: torch.Tensor,
        quaternion_wxyz: torch.Tensor | None = None,
    ) -> None:
        """Update the Cartesian goal without resetting the warm start."""

        if self._goal_poses is None:
            raise RuntimeError("setup() must be called before set_target()")
        target = self._goal_poses[self.solver.tool_frames[0]]
        target.position.copy_(position_m.reshape_as(target.position))
        if quaternion_wxyz is not None:
            target.quaternion.copy_(quaternion_wxyz.reshape_as(target.quaternion))
        if not self.solver.update_goal_tool_poses(
            GoalToolPose.from_poses(
                self._goal_poses,
                ordered_tool_frames=self.solver.tool_frames,
                num_goalset=1,
            ),
            run_ik=False,
        ):
            raise RuntimeError("Cartesian goal update failed")

    def step(self) -> MpcCommandWindow:
        """Return one MPC period of continuous commands."""

        return self._planner.step()
