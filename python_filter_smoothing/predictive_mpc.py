"""Compact predicted-state MPC producer independent of robot simulation."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol

import torch
from curobo.model_predictive_control import ModelPredictiveControlResult
from curobo.types import JointState


class MpcSolver(Protocol):
    """Minimum cuRobo solver interface used by the application layer."""

    @property
    def joint_names(self) -> list[str]: ...

    def setup(self, current_state: JointState) -> None: ...

    def optimize_action_sequence(
        self, current_state: JointState
    ) -> ModelPredictiveControlResult: ...


@dataclass(frozen=True)
class PredictiveMpcTiming:
    """Timing contract for one MPC producer and a faster servo consumer."""

    mpc_period_s: float
    optimization_dt_s: float
    interpolation_steps: int
    required_feasible_windows: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.mpc_period_s) or self.mpc_period_s <= 0.0:
            raise ValueError("mpc_period_s must be finite and greater than zero")
        if not math.isfinite(self.optimization_dt_s) or self.optimization_dt_s <= 0.0:
            raise ValueError("optimization_dt_s must be finite and greater than zero")
        if self.interpolation_steps != 4:
            raise ValueError("cuRobo MPC currently requires interpolation_steps=4")
        if self.required_feasible_windows < 1:
            raise ValueError("required_feasible_windows must be at least one")

    @property
    def command_dt_s(self) -> float:
        """Interval of the native MPC state commands."""

        return self.optimization_dt_s

    @property
    def prediction_index(self) -> int:
        """Full-state index exactly one MPC period after the solve input."""

        ratio = self.mpc_period_s / self.optimization_dt_s
        rounded = round(ratio)
        if rounded < 1 or not math.isclose(ratio, rounded, rel_tol=1.0e-9):
            raise ValueError(
                "mpc_period_s must be an integer multiple of optimization_dt_s"
            )
        return rounded


@dataclass(frozen=True)
class MpcCommandWindow:
    """Application-owned command window produced by one synchronous solve."""

    commands: JointState
    next_current_state: JointState
    boundary_reference_state: JointState
    solve_time_s: float
    wall_time_s: float
    initial_position_error_rad: float
    initial_velocity_error_rad_s: float
    initial_acceleration_error_rad_s2: float
    position_boundary_error_rad: float | None
    velocity_boundary_error_rad_s: float | None
    acceleration_boundary_error_rad_s2: float | None
    full_horizon_feasible: bool
    used_feasible_tail_fallback: bool
    rejected_by_command_limits: bool


@dataclass(frozen=True)
class MpcCommandLimits:
    """Per-joint limits used to reject unsafe optimizer solutions."""

    velocity: torch.Tensor
    acceleration: torch.Tensor
    jerk: torch.Tensor


def _state_at(state: JointState, index: int, joint_names: list[str]) -> JointState:
    """Clone one horizon state while preserving position, velocity, and acceleration."""

    if not isinstance(state.position, torch.Tensor) or state.position.ndim != 3:
        raise RuntimeError(
            "MPC state trajectory position must have shape [batch, horizon, dof]"
        )
    if not isinstance(state.velocity, torch.Tensor):
        raise TypeError("MPC state trajectory is missing velocity")
    if not isinstance(state.acceleration, torch.Tensor):
        raise TypeError("MPC state trajectory is missing acceleration")
    if index < 0 or index >= state.position.shape[1]:
        raise RuntimeError(f"MPC state index {index} is outside the returned horizon")
    result = JointState.from_position(
        state.position[:, index, :].clone(), joint_names=joint_names
    )
    result.velocity = state.velocity[:, index, :].clone()
    result.acceleration = state.acceleration[:, index, :].clone()
    return result


def _finite_state(state: JointState) -> bool:
    tensors = (state.position, state.velocity, state.acceleration)
    return all(
        isinstance(value, torch.Tensor)
        and bool(torch.all(torch.isfinite(value)).item())
        for value in tensors
    )


def _state_slice(
    state: JointState, start: int, stop: int, joint_names: list[str]
) -> JointState:
    """Clone the native MPC states before the next predicted boundary."""

    if not isinstance(state.position, torch.Tensor):
        raise TypeError("MPC state trajectory is missing position")
    if not isinstance(state.velocity, torch.Tensor):
        raise TypeError("MPC state trajectory is missing velocity")
    if not isinstance(state.acceleration, torch.Tensor):
        raise TypeError("MPC state trajectory is missing acceleration")
    if start < 0 or stop <= start or stop > state.position.shape[1]:
        raise RuntimeError("native MPC command window is outside the returned horizon")
    result = JointState.from_position(
        state.position[:, start:stop, :].clone(), joint_names=joint_names
    )
    result.velocity = state.velocity[:, start:stop, :].clone()
    result.acceleration = state.acceleration[:, start:stop, :].clone()
    return result


def _norm_difference(first: JointState, second: JointState, field: str) -> float:
    first_value = getattr(first, field)
    second_value = getattr(second, field)
    if not isinstance(first_value, torch.Tensor) or not isinstance(
        second_value, torch.Tensor
    ):
        raise TypeError(f"MPC state is missing {field}")
    return float(torch.linalg.vector_norm(first_value - second_value).item())


def _feasible_prefix_length(solver: MpcSolver) -> int:
    """Count consecutive constraint-feasible states from the horizon start."""

    manager = getattr(solver, "trajectory_execution_manager", None)
    if manager is None or not hasattr(manager, "get_current_metrics"):
        return 0
    feasible = manager.get_current_metrics().feasible
    if not isinstance(feasible, torch.Tensor) or feasible.ndim < 2:
        return 0
    by_state = torch.all(feasible.reshape(-1, feasible.shape[-1]), dim=0)
    first_failure = torch.nonzero(~by_state, as_tuple=False)
    return int(first_failure[0, 0].item()) if len(first_failure) else len(by_state)


def _within_command_limits(
    state: JointState,
    initial_state: JointState,
    dt_s: float,
    limits: MpcCommandLimits,
) -> bool:
    """Check the full candidate horizon before it can become a fallback tail."""

    if not _finite_state(state):
        return False
    velocity = state.velocity
    acceleration = state.acceleration
    initial_acceleration = initial_state.acceleration
    if not isinstance(velocity, torch.Tensor) or not isinstance(
        acceleration, torch.Tensor
    ):
        return False
    if not isinstance(initial_acceleration, torch.Tensor):
        return False
    jerk = (
        torch.diff(
            torch.cat((initial_acceleration[:, None, :], acceleration), dim=1), dim=1
        )
        / dt_s
    )
    return bool(
        torch.all(velocity.abs() <= limits.velocity).item()
        and torch.all(acceleration.abs() <= limits.acceleration).item()
        and torch.all(jerk.abs() <= limits.jerk).item()
    )


class PredictedStateMpc:
    """Solve from the state predicted one MPC period by the previous result.

    Each solve's ``prediction_index`` is exactly one MPC period after its input.
    It supplies the next q/dq/ddq solve input and directly commits the preceding
    native states from cuRobo's full rollout. The class contains no simulation or
    I/O; a real system can run :meth:`step` in a planning worker and atomically
    publish each cloned :class:`MpcCommandWindow` to a command consumer.
    """

    def __init__(
        self,
        solver: MpcSolver,
        timing: PredictiveMpcTiming,
        command_limits: MpcCommandLimits | None = None,
    ) -> None:
        self.solver = solver
        self.timing = timing
        self.command_limits = command_limits
        self._current_state: JointState | None = None
        self._previous_boundary: JointState | None = None
        self._feasible_tail: JointState | None = None
        self._feasible_tail_start = 0
        self._feasible_tail_stop = 0

    @property
    def current_state(self) -> JointState:
        """Return a clone of the state that will initialize the next solve."""

        if self._current_state is None:
            raise RuntimeError("PredictedStateMpc.setup must be called first")
        return self._current_state.clone()

    def setup(self, initial_state: JointState) -> None:
        """Initialize cuRobo and the predicted-state chain from a measured state."""

        if not _finite_state(initial_state):
            raise ValueError("initial_state q/dq/ddq must be finite")
        self._current_state = initial_state.clone()
        self._previous_boundary = None
        self._feasible_tail = None
        self._feasible_tail_start = 0
        self._feasible_tail_stop = 0
        self.solver.setup(self._current_state)

    def step(self) -> MpcCommandWindow:
        """Run one solve and advance q/dq/ddq by exactly one MPC period."""

        if self._current_state is None:
            raise RuntimeError("PredictedStateMpc.setup must be called first")
        started = time.perf_counter()
        result = self.solver.optimize_action_sequence(self._current_state)
        wall_time_s = time.perf_counter() - started
        full_horizon_feasible = result.success is not None and bool(
            torch.all(result.success).item()
        )
        prefix_stop = self.timing.prediction_index + 1
        feasible_prefix_length = _feasible_prefix_length(self.solver)
        prefix_feasible = feasible_prefix_length >= prefix_stop
        used_fallback = False
        state_start = 0
        full_state = (
            result.robot_state_sequence.joint_state
            if result.robot_state_sequence is not None
            else None
        )
        rejected_by_command_limits = bool(
            full_state is not None
            and self.command_limits is not None
            and not _within_command_limits(
                full_state,
                self._current_state,
                self.timing.command_dt_s,
                self.command_limits,
            )
        )
        if rejected_by_command_limits:
            full_horizon_feasible = False
            prefix_feasible = False
        fallback_stop = self._feasible_tail_start + self.timing.prediction_index
        has_fallback_window = (
            self._feasible_tail is not None and fallback_stop < self._feasible_tail_stop
        )
        has_safe_backup = (
            not rejected_by_command_limits
            and feasible_prefix_length
            >= self.timing.required_feasible_windows * self.timing.prediction_index + 1
        )
        candidate_usable = full_horizon_feasible or prefix_feasible
        if (
            not full_horizon_feasible
            and not has_safe_backup
            and has_fallback_window
            and self._feasible_tail is not None
        ):
            full_state = self._feasible_tail
            state_start = self._feasible_tail_start
            self._feasible_tail_start = fallback_stop
            used_fallback = True
        if full_state is None or (
            not full_horizon_feasible and not used_fallback and not prefix_feasible
        ):
            raise RuntimeError("MPC returned an infeasible result")
        if not _finite_state(full_state):
            raise RuntimeError("MPC returned non-finite q/dq/ddq")

        rollout_initial = _state_at(full_state, state_start, self.solver.joint_names)
        initial_position_error = _norm_difference(
            self._current_state, rollout_initial, "position"
        )
        initial_velocity_error = _norm_difference(
            self._current_state, rollout_initial, "velocity"
        )
        initial_acceleration_error = _norm_difference(
            self._current_state, rollout_initial, "acceleration"
        )

        next_current = _state_at(
            full_state,
            state_start + self.timing.prediction_index,
            self.solver.joint_names,
        )
        commands = _state_slice(
            full_state,
            state_start,
            state_start + self.timing.prediction_index,
            self.solver.joint_names,
        )
        boundary = next_current.clone()

        position_error = velocity_error = acceleration_error = None
        if self._previous_boundary is not None:
            first_command = _state_at(commands, 0, self.solver.joint_names)
            position_error = _norm_difference(
                self._previous_boundary, first_command, "position"
            )
            velocity_error = _norm_difference(
                self._previous_boundary, first_command, "velocity"
            )
            acceleration_error = _norm_difference(
                self._previous_boundary, first_command, "acceleration"
            )

        self._current_state = next_current.clone()
        self._previous_boundary = boundary.clone()
        if not used_fallback and candidate_usable:
            self._feasible_tail = full_state.clone()
            self._feasible_tail_start = self.timing.prediction_index
            self._feasible_tail_stop = (
                full_state.position.shape[1]
                if full_horizon_feasible
                else feasible_prefix_length
            )
        elif not used_fallback:
            self._feasible_tail = None
        if used_fallback and hasattr(self.solver, "reset_robot"):
            self.solver.reset_robot(self._current_state)
        return MpcCommandWindow(
            commands=commands,
            next_current_state=next_current,
            boundary_reference_state=boundary,
            solve_time_s=float(result.solve_time),
            wall_time_s=wall_time_s,
            initial_position_error_rad=initial_position_error,
            initial_velocity_error_rad_s=initial_velocity_error,
            initial_acceleration_error_rad_s2=initial_acceleration_error,
            position_boundary_error_rad=position_error,
            velocity_boundary_error_rad_s=velocity_error,
            acceleration_boundary_error_rad_s2=acceleration_error,
            full_horizon_feasible=full_horizon_feasible,
            used_feasible_tail_fallback=used_fallback,
            rejected_by_command_limits=rejected_by_command_limits,
        )
