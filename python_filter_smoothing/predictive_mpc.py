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

    mpc_period_s: float = 0.125
    interpolation_steps: int = 4
    optimization_dt_divisor: int = 5

    def __post_init__(self) -> None:
        if not math.isfinite(self.mpc_period_s) or self.mpc_period_s <= 0.0:
            raise ValueError("mpc_period_s must be finite and greater than zero")
        if self.interpolation_steps != 4:
            raise ValueError("cuRobo MPC currently requires interpolation_steps=4")
        if self.optimization_dt_divisor < 1:
            raise ValueError("optimization_dt_divisor must be greater than zero")

    @property
    def solver_optimization_dt_s(self) -> float:
        """Value passed to cuRobo's current ``optimization_dt`` implementation."""

        return self.mpc_period_s / self.optimization_dt_divisor

    @property
    def command_dt_s(self) -> float:
        """Interval of the native MPC state commands."""

        return self.solver_optimization_dt_s

    @property
    def commands_per_mpc_period(self) -> int:
        """Number of native MPC states committed by each solve."""

        return self.prediction_index

    @property
    def prediction_index(self) -> int:
        """Full-state index exactly one MPC period after the solve input."""

        ratio = self.mpc_period_s / self.solver_optimization_dt_s
        rounded = round(ratio)
        if rounded < 1 or not math.isclose(ratio, rounded, rel_tol=1.0e-9):
            raise ValueError(
                "mpc_period_s must be an integer multiple of solver_optimization_dt_s"
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


def _state_slice(state: JointState, stop: int, joint_names: list[str]) -> JointState:
    """Clone the native MPC states before the next predicted boundary."""

    if not isinstance(state.position, torch.Tensor):
        raise TypeError("MPC state trajectory is missing position")
    if not isinstance(state.velocity, torch.Tensor):
        raise TypeError("MPC state trajectory is missing velocity")
    if not isinstance(state.acceleration, torch.Tensor):
        raise TypeError("MPC state trajectory is missing acceleration")
    if stop < 1 or stop > state.position.shape[1]:
        raise RuntimeError("native MPC command window is outside the returned horizon")
    result = JointState.from_position(
        state.position[:, :stop, :].clone(), joint_names=joint_names
    )
    result.velocity = state.velocity[:, :stop, :].clone()
    result.acceleration = state.acceleration[:, :stop, :].clone()
    return result


def _norm_difference(first: JointState, second: JointState, field: str) -> float:
    first_value = getattr(first, field)
    second_value = getattr(second, field)
    if not isinstance(first_value, torch.Tensor) or not isinstance(
        second_value, torch.Tensor
    ):
        raise TypeError(f"MPC state is missing {field}")
    return float(torch.linalg.vector_norm(first_value - second_value).item())


class PredictedStateMpc:
    """Solve from the state predicted one MPC period by the previous result.

    Each solve's ``prediction_index`` is exactly one MPC period after its input.
    It supplies the next q/dq/ddq solve input and directly commits the preceding
    native states from cuRobo's full rollout. The class contains no simulation or
    I/O; a real system can run :meth:`step` in a planning worker and atomically
    publish each cloned :class:`MpcCommandWindow` to a command consumer.
    """

    def __init__(self, solver: MpcSolver, timing: PredictiveMpcTiming) -> None:
        self.solver = solver
        self.timing = timing
        self._current_state: JointState | None = None
        self._previous_boundary: JointState | None = None

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
        self.solver.setup(self._current_state)

    def step(self) -> MpcCommandWindow:
        """Run one solve and advance q/dq/ddq by exactly one MPC period."""

        if self._current_state is None:
            raise RuntimeError("PredictedStateMpc.setup must be called first")
        started = time.perf_counter()
        result = self.solver.optimize_action_sequence(self._current_state)
        wall_time_s = time.perf_counter() - started
        if result.success is None or not bool(torch.all(result.success).item()):
            raise RuntimeError("MPC returned an infeasible result")
        if result.robot_state_sequence is None:
            raise RuntimeError("MPC did not return robot_state_sequence")
        full_state = result.robot_state_sequence.joint_state
        if not _finite_state(full_state):
            raise RuntimeError("MPC returned non-finite q/dq/ddq")

        rollout_initial = _state_at(full_state, 0, self.solver.joint_names)
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
            full_state, self.timing.prediction_index, self.solver.joint_names
        )
        commands = _state_slice(
            full_state, self.timing.prediction_index, self.solver.joint_names
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
        )
