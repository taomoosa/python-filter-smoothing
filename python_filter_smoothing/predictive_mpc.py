"""Small cuRobo MPC horizon producer with no application fallback policy."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol

import torch
from curobo.model_predictive_control import ModelPredictiveControlResult
from curobo.types import JointState


class MpcSolver(Protocol):
    """Minimum cuRobo solver interface used by the horizon producer."""

    @property
    def joint_names(self) -> list[str]: ...

    def setup(self, current_state: JointState) -> None: ...

    def optimize_action_sequence(
        self, current_state: JointState
    ) -> ModelPredictiveControlResult: ...


@dataclass(frozen=True)
class MpcTiming:
    """Native MPC output timing."""

    optimization_dt_s: float
    interpolation_steps: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.optimization_dt_s) or self.optimization_dt_s <= 0.0:
            raise ValueError("optimization_dt_s must be finite and positive")
        if self.interpolation_steps != 4:
            raise ValueError("cuRobo MPC currently requires interpolation_steps=4")

    @property
    def command_dt_s(self) -> float:
        return self.optimization_dt_s


@dataclass(frozen=True)
class MpcHorizon:
    """One complete, unselected MPC result."""

    states: JointState
    solve_time_s: float
    wall_time_s: float
    full_horizon_feasible: bool
    feasible_prefix_length: int
    rejected_by_command_limits: bool


@dataclass(frozen=True)
class MpcCommandLimits:
    """Per-joint limits used to reject unsafe optimizer output."""

    velocity: torch.Tensor
    acceleration: torch.Tensor
    jerk: torch.Tensor


def _finite_state(state: JointState) -> bool:
    return all(
        isinstance(value, torch.Tensor)
        and bool(torch.all(torch.isfinite(value)).item())
        for value in (state.position, state.velocity, state.acceleration)
    )


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
    if not _finite_state(state):
        return False
    jerk = torch.diff(
        torch.cat((initial_state.acceleration[:, None], state.acceleration), dim=1),
        dim=1,
    ) / dt_s
    return bool(
        torch.all(state.velocity.abs() <= limits.velocity).item()
        and torch.all(state.acceleration.abs() <= limits.acceleration).item()
        and torch.all(jerk.abs() <= limits.jerk).item()
    )


class MpcHorizonProducer:
    """Solve complete horizons; queue ownership and fallback stay outside."""

    def __init__(
        self,
        solver: MpcSolver,
        timing: MpcTiming,
        command_limits: MpcCommandLimits,
    ) -> None:
        self.solver = solver
        self.timing = timing
        self.command_limits = command_limits
        self._current_state: JointState | None = None

    @property
    def current_state(self) -> JointState:
        if self._current_state is None:
            raise RuntimeError("MpcHorizonProducer.setup must be called first")
        return self._current_state.clone()

    def setup(self, initial_state: JointState) -> None:
        if not _finite_state(initial_state):
            raise ValueError("initial_state q/dq/ddq must be finite")
        self._current_state = initial_state.clone()
        self.solver.setup(self._current_state)

    def solve_horizon(self) -> MpcHorizon:
        if self._current_state is None:
            raise RuntimeError("MpcHorizonProducer.setup must be called first")
        started = time.perf_counter()
        result = self.solver.optimize_action_sequence(self._current_state)
        wall_time_s = time.perf_counter() - started
        state = (
            result.robot_state_sequence.joint_state
            if result.robot_state_sequence is not None
            else None
        )
        if state is None:
            raise RuntimeError("MPC returned no state trajectory")
        if not _finite_state(state):
            raise RuntimeError("MPC returned non-finite q/dq/ddq")
        rejected = not _within_command_limits(
            state, self._current_state, self.timing.command_dt_s, self.command_limits
        )
        feasible = (
            result.success is not None
            and bool(torch.all(result.success).item())
            and not rejected
        )
        return MpcHorizon(
            states=state.clone(),
            solve_time_s=float(result.solve_time),
            wall_time_s=wall_time_s,
            full_horizon_feasible=feasible,
            feasible_prefix_length=_feasible_prefix_length(self.solver),
            rejected_by_command_limits=rejected,
        )
