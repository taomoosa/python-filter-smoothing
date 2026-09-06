"""Minimal cuRobo MPC full-horizon generator; simulation is external."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from curobo.config_io import load_yaml
from curobo.content import (
    get_robot_configs_path,
    get_scene_configs_path,
    get_task_configs_path,
)
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.model_predictive_control import (
    ModelPredictiveControl,
    ModelPredictiveControlCfg,
)
from curobo.types import GoalToolPose, JointState

from .predictive_mpc import (
    MpcCommandLimits,
    MpcHorizon,
    MpcHorizonProducer,
    MpcTiming,
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


def _positive_iterations(value: Any, name: str) -> int:
    """Validate an application-configured optimizer iteration count."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result != value or result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _iteration_sequence(value: Any, name: str) -> tuple[int, ...]:
    """Validate a nonempty, strictly increasing iteration sequence."""

    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty sequence")
    result = tuple(
        _positive_iterations(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{name} must be unique and strictly increasing")
    return result


def _position_offset_sequence(
    value: Any, name: str
) -> tuple[tuple[float, float, float], ...]:
    """Validate preferred Cartesian offsets for nearby-target IK."""

    if value is None:
        return ((0.0, 0.0, 0.0),)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty sequence")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(f"{name}[{index}] must contain XYZ")
        offset = tuple(float(component) for component in item)
        if not all(math.isfinite(component) for component in offset):
            raise ValueError(f"{name}[{index}] must be finite")
        result.append(offset)
    if result[0] != (0.0, 0.0, 0.0):
        raise ValueError(f"{name}[0] must be the exact target [0, 0, 0]")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate offsets")
    return tuple(result)


def _weight_sequence(value: Any, name: str, length: int) -> list[float]:
    """Validate an optimizer weight vector before constructing cuRobo."""

    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) or item < 0.0 for item in result):
        raise ValueError(f"{name} values must be finite and nonnegative")
    return result


def _weight(value: Any, name: str) -> float:
    """Validate a scalar optimizer weight."""

    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _robot_config(config: dict[str, Any]) -> dict[str, Any]:
    robot_options = config["robot"]
    robot = load_yaml(robot_options["config"])
    cspace = robot["robot_cfg"]["kinematics"]["cspace"]
    cspace["velocity_scale"] = robot_options["velocity_limit_scale"]
    cspace["acceleration_scale"] = robot_options["acceleration_limit_scale"]
    cspace["jerk_scale"] = robot_options["jerk_limit_scale"]
    return robot


def _solver(config: dict[str, Any]) -> ModelPredictiveControl:
    robot = _robot_config(config)

    optimizer_options = config["optimizer"]
    optimizer = load_yaml(optimizer_options["base_config"])
    optimizer["optimizer"]["fixed_iters"] = bool(
        optimizer_options.get("fixed_iterations", True)
    )
    optimizer["optimizer"]["return_best_action"] = bool(
        optimizer_options.get("return_best_action", True)
    )
    costs = optimizer["rollout"]["cost_cfg"]
    costs["tool_pose_cfg"]["weight"] = _weight_sequence(
        optimizer_options["tool_pose_weight"], "optimizer.tool_pose_weight", 2
    )
    costs["cspace_cfg"]["weight"] = _weight_sequence(
        optimizer_options["cspace_bound_weight"],
        "optimizer.cspace_bound_weight",
        5,
    )
    if "scene_collision_weight" in optimizer_options:
        optimizer["rollout"]["constraint_cfg"]["scene_collision_cfg"]["weight"] = (
            _weight(
                optimizer_options["scene_collision_weight"],
                "optimizer.scene_collision_weight",
            )
        )
    if "self_collision_weight" in optimizer_options:
        optimizer["rollout"]["constraint_cfg"]["self_collision_cfg"]["weight"] = (
            _weight(
                optimizer_options["self_collision_weight"],
                "optimizer.self_collision_weight",
            )
        )

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
        squared_l2_regularization_weight=_weight_sequence(
            optimizer_options["squared_l2_regularization_weight"],
            "optimizer.squared_l2_regularization_weight",
            5,
        ),
        non_terminal_tool_pose_weight_factor=optimizer_options[
            "non_terminal_tool_pose_weight_factor"
        ],
        warm_start_optimization_num_iters=_positive_iterations(
            optimizer_options["warm_start_iterations"],
            "optimizer.warm_start_iterations",
        ),
        cold_start_optimization_num_iters=_positive_iterations(
            optimizer_options["cold_start_iterations"],
            "optimizer.cold_start_iterations",
        ),
        optimizer_collision_activation_distance=collision["activation_distance_m"],
        self_collision_check=collision["self_collision_check"],
        use_cuda_graph=runtime["use_cuda_graph"],
        random_seed=runtime["random_seed"],
    )
    return ModelPredictiveControl(solver_config)


def _fallback_ik_solver(
    config: dict[str, Any], num_seeds: int, max_batch_size: int
) -> InverseKinematics:
    runtime = config["runtime"]
    collision = config["collision"]
    solver_config = InverseKinematicsCfg.create(
        robot=_robot_config(config),
        scene_model=load_yaml(config["scene"]),
        num_seeds=num_seeds,
        max_batch_size=max_batch_size,
        optimizer_collision_activation_distance=collision["activation_distance_m"],
        self_collision_check=collision["self_collision_check"],
        use_cuda_graph=runtime["use_cuda_graph"],
        random_seed=runtime["random_seed"],
    )
    return InverseKinematics(solver_config)


@dataclass(frozen=True)
class MpcTargetSelection:
    """IK-selected target; candidate zero is the exact requested pose."""

    candidate_index: int
    position_m: torch.Tensor
    quaternion_wxyz: torch.Tensor
    joint_state: JointState


class ContinuousMpcTrajectory:
    """Convert successive Cartesian goals into complete q/dq/ddq horizons."""

    def __init__(self, config_path: str | Path) -> None:
        self.config = load_config(config_path)
        optimizer_options = self.config["optimizer"]
        target_update = optimizer_options.get("target_update", {})
        self._use_ik_joint_reference = bool(
            target_update.get("use_ik_joint_reference", False)
        )
        self._seed_from_ik = bool(target_update.get("seed_from_ik", False))
        if self._seed_from_ik and not self._use_ik_joint_reference:
            raise ValueError("seed_from_ik requires use_ik_joint_reference")
        self._ik_fallback_seeds = _positive_iterations(
            target_update.get("ik_fallback_seeds", 1),
            "optimizer.target_update.ik_fallback_seeds",
        )
        self._ik_position_offsets_m = _position_offset_sequence(
            target_update.get("ik_position_offsets_m"),
            "optimizer.target_update.ik_position_offsets_m",
        )
        timing = self.config["timing"]
        self.timing = MpcTiming(
            optimization_dt_s=float(timing["optimization_dt_s"]),
            interpolation_steps=int(timing["interpolation_steps"]),
        )
        self.solver = _solver(self.config)
        self._fallback_ik = (
            _fallback_ik_solver(
                self.config,
                self._ik_fallback_seeds,
                len(self._ik_position_offsets_m),
            )
            if self._use_ik_joint_reference
            and (
                self._ik_fallback_seeds > 1
                or len(self._ik_position_offsets_m) > 1
            )
            else None
        )
        self._setup_cold_start_iterations = _positive_iterations(
            optimizer_options["cold_start_iterations"],
            "optimizer.cold_start_iterations",
        )
        self._candidate_iterations = _iteration_sequence(
            target_update.get(
                "candidate_iterations", [optimizer_options["cold_start_iterations"]]
            ),
            "optimizer.target_update.candidate_iterations",
        )
        transition = self.solver.transition_model
        self.command_limits = MpcCommandLimits(
            velocity=transition.max_velocity,
            acceleration=transition.max_acceleration,
            jerk=transition.max_jerk,
        )
        self._planner = MpcHorizonProducer(
            self.solver,
            self.timing,
            command_limits=self.command_limits,
        )
        self._goal_request: GoalToolPose | None = None
        self._joint_reference_state: JointState | None = None
        self._last_target_selection: MpcTargetSelection | None = None

    @property
    def joint_names(self) -> list[str]:
        return list(self.solver.joint_names)

    @property
    def candidate_iterations(self) -> tuple[int, ...]:
        """Independent cold-solve iteration counts configured for one target."""

        return self._candidate_iterations

    @property
    def last_target_selection(self) -> MpcTargetSelection | None:
        """Return the IK pose selected for the active target, if applicable."""

        return self._last_target_selection

    def default_state(self) -> JointState:
        """Return a finite q/dq/ddq state suitable for initial setup."""

        return JointState.from_position(
            self.solver.default_joint_position.clone().unsqueeze(0),
            joint_names=self.joint_names,
        )

    def setup(self, initial_state: JointState) -> None:
        """Initialize from a measured state before the periodic loop starts."""

        self.solver.config.cold_start_optimization_num_iters = (
            self._setup_cold_start_iterations
        )
        self._planner.setup(initial_state)
        self._joint_reference_state = None
        self._last_target_selection = None
        initial_poses = self.solver.compute_kinematics(
            initial_state
        ).tool_poses.to_dict()
        self._goal_request = GoalToolPose.from_poses(
            initial_poses,
            ordered_tool_frames=self.solver.tool_frames,
            num_goalset=1,
        )

    def _target_selection(
        self,
        candidate_index: int,
        joint_position: torch.Tensor,
    ) -> MpcTargetSelection:
        if self._goal_request is None:
            raise RuntimeError("setup() must be called before set_target()")
        offset = torch.as_tensor(
            self._ik_position_offsets_m[candidate_index],
            device=self._goal_request.position.device,
            dtype=self._goal_request.position.dtype,
        )
        return MpcTargetSelection(
            candidate_index=candidate_index,
            position_m=(self._goal_request.position[0, 0, 0, 0] + offset).clone(),
            quaternion_wxyz=self._goal_request.quaternion[0, 0, 0, 0].clone(),
            joint_state=JointState.from_position(
                joint_position.reshape(1, -1).clone(), joint_names=self.joint_names
            ),
        )

    def _nearby_goal_batch(self) -> GoalToolPose:
        if self._goal_request is None:
            raise RuntimeError("setup() must be called before set_target()")
        count = len(self._ik_position_offsets_m)
        position = self._goal_request.position.repeat(count, 1, 1, 1, 1)
        quaternion = self._goal_request.quaternion.repeat(count, 1, 1, 1, 1)
        offsets = torch.as_tensor(
            self._ik_position_offsets_m,
            device=position.device,
            dtype=position.dtype,
        )
        position[:, 0, 0, 0, :] += offsets
        return GoalToolPose(
            tool_frames=list(self._goal_request.tool_frames),
            position=position,
            quaternion=quaternion,
        )

    def _solve_target_ik(self, current: JointState) -> MpcTargetSelection:
        if self._goal_request is None or not isinstance(current.position, torch.Tensor):
            raise RuntimeError("setup() must be called before set_target()")
        solver = self.solver.ik_solver
        result = solver.solve_pose(
            goal_tool_poses=self._goal_request,
            current_state=current,
            seed_config=current.position.reshape(1, 1, -1).clone(),
            return_seeds=1,
        )
        if bool(torch.any(result.success).item()):
            solution = result.solution.reshape(-1, len(self.joint_names))[:1]
            return self._target_selection(0, solution[0])

        if self._fallback_ik is not None:
            self._fallback_ik.reset_seed()
            result = self._fallback_ik.solve_pose(
                goal_tool_poses=self._nearby_goal_batch(),
                current_state=None,
                seed_config=None,
                return_seeds=self._ik_fallback_seeds,
            )
            candidate_count = len(self._ik_position_offsets_m)
            success = result.success.reshape(candidate_count, -1)
            if bool(torch.any(success).item()):
                # Offset order is an application priority: exact target first,
                # then progressively relaxed nearby poses.  Collision along the
                # straight joint seed is deliberately not tested here; avoiding
                # it is the MPC optimization's job.
                candidate_index = int(
                    torch.nonzero(torch.any(success, dim=1), as_tuple=False)[0, 0]
                )
                solutions = result.solution.reshape(
                    candidate_count, -1, len(self.joint_names)
                )[candidate_index]
                bounds = self.solver.transition_model.get_state_bounds().position
                scale = (bounds[1] - bounds[0]).clamp_min(1.0e-6)
                distance = torch.sum(
                    torch.square(
                        (solutions - current.position.reshape(1, -1)) / scale
                    ),
                    dim=-1,
                )
                distance[~success[candidate_index]] = torch.inf
                index = int(torch.argmin(distance).item())
                return self._target_selection(candidate_index, solutions[index])

        position_error = float(result.position_error.min().item())
        rotation_error = float(result.rotation_error.min().item())
        raise RuntimeError(
            "Cartesian goal IK failed "
            f"(position_error={position_error:.6g} m, "
            f"rotation_error={rotation_error:.6g} rad)"
        )

    def set_target(
        self,
        position_m: torch.Tensor,
        quaternion_wxyz: torch.Tensor | None = None,
        *,
        validate_ik: bool = False,
    ) -> JointState | None:
        """Update a Cartesian target and prepare its first independent solve.

        With ``validate_ik=True``, only resolve and return IK without changing the
        active MPC objective.  The configured runtime path instead installs that
        IK result as both a joint reference and, optionally, an optimizer seed.
        """

        if self._goal_request is None:
            raise RuntimeError("setup() must be called before set_target()")
        current = self._planner.current_state
        goal = self._goal_request
        goal.position[:, :, 0, :, :].copy_(position_m.reshape(1, 1, 1, 3))
        if quaternion_wxyz is not None:
            goal.quaternion[:, :, 0, :, :].copy_(quaternion_wxyz.reshape(1, 1, 1, 4))
        selection = (
            self._solve_target_ik(current)
            if validate_ik or self._use_ik_joint_reference
            else None
        )
        joint_reference = selection.joint_state if selection is not None else None
        self._last_target_selection = selection
        if validate_ik:
            return joint_reference
        if selection is not None:
            goal.position[:, :, 0, 0, :].copy_(
                selection.position_m.reshape(1, 1, 3)
            )
            goal.quaternion[:, :, 0, 0, :].copy_(
                selection.quaternion_wxyz.reshape(1, 1, 4)
            )
        if not self.solver.update_goal_tool_poses(goal, run_ik=False):
            raise RuntimeError("Cartesian goal update failed")
        self._joint_reference_state = joint_reference
        self.prepare_candidate(current, self._candidate_iterations[0])
        return joint_reference

    def prepare_candidate(self, initial_state: JointState, iterations: int) -> None:
        """Reset one independent cold solve for the active target."""

        self.solver.config.cold_start_optimization_num_iters = _positive_iterations(
            iterations, "candidate iterations"
        )
        self.solver.reset_robot(initial_state)
        if self._joint_reference_state is not None:
            self.solver.update_goal_state(self._joint_reference_state)
            self.solver.enable_joint_position_tracking()
            if self._seed_from_ik:
                self.solver.update_seed_trajectory_from_goal_state(
                    self._joint_reference_state
                )

    def solve_horizon(self) -> MpcHorizon:
        """Return the complete latest MPC horizon without advancing state."""

        return self._planner.solve_horizon()
