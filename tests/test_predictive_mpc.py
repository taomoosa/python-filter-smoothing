from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from curobo.types import JointState

from python_filter_smoothing.predictive_mpc import (
    MpcCommandLimits,
    PredictedStateMpc,
    PredictiveMpcTiming,
)


class _Solver:
    joint_names: ClassVar[list[str]] = ["j1"]

    def __init__(self, feasible: torch.Tensor) -> None:
        metrics = SimpleNamespace(feasible=feasible)
        self.trajectory_execution_manager = SimpleNamespace(
            get_current_metrics=lambda: metrics
        )

    def setup(self, current_state: JointState) -> None:
        pass

    def optimize_action_sequence(self, current_state: JointState) -> SimpleNamespace:
        position = current_state.position[:, None, :].expand(1, 12, 1).clone()
        full = JointState.from_position(position, joint_names=self.joint_names)
        full.velocity = torch.zeros_like(position)
        full.acceleration = torch.zeros_like(position)
        return SimpleNamespace(
            success=torch.tensor([False]),
            robot_state_sequence=SimpleNamespace(joint_state=full),
            solve_time=0.001,
        )


def _state() -> JointState:
    state = JointState.from_position(torch.zeros(1, 1), joint_names=["j1"])
    state.velocity = torch.zeros_like(state.position)
    state.acceleration = torch.zeros_like(state.position)
    return state


def _timing() -> PredictiveMpcTiming:
    return PredictiveMpcTiming(
        mpc_period_s=0.125,
        optimization_dt_s=0.025,
        interpolation_steps=4,
        required_feasible_windows=4,
    )


def test_accepts_feasible_executable_prefix_when_future_horizon_is_infeasible() -> None:
    feasible = torch.tensor([[True] * 6 + [False] * 6])
    planner = PredictedStateMpc(_Solver(feasible), _timing())
    planner.setup(_state())

    window = planner.step()

    assert not window.full_horizon_feasible
    assert window.commands.position.shape[1] == 5


def test_rejects_infeasible_executable_prefix() -> None:
    feasible = torch.tensor([[True] * 5 + [False] * 7])
    planner = PredictedStateMpc(_Solver(feasible), _timing())
    planner.setup(_state())

    with pytest.raises(RuntimeError, match="infeasible"):
        planner.step()


def test_uses_previous_feasible_tail_when_next_solve_is_infeasible() -> None:
    class SequenceSolver(_Solver):
        def __init__(self) -> None:
            super().__init__(torch.tensor([[False] * 12]))
            self.calls = 0

        def optimize_action_sequence(
            self, current_state: JointState
        ) -> SimpleNamespace:
            index = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1)
            position = current_state.position[:, None, :] + index
            if self.calls:
                position += 100.0
            full = JointState.from_position(position, joint_names=self.joint_names)
            full.velocity = torch.zeros_like(position)
            full.acceleration = torch.zeros_like(position)
            success = torch.tensor([self.calls == 0])
            self.calls += 1
            return SimpleNamespace(
                success=success,
                robot_state_sequence=SimpleNamespace(joint_state=full),
                solve_time=0.001,
            )

        def reset_robot(self, current_state: JointState) -> None:
            pass

    planner = PredictedStateMpc(SequenceSolver(), _timing())
    planner.setup(_state())

    planner.step()
    fallback = planner.step()

    assert fallback.used_feasible_tail_fallback
    assert fallback.commands.position[0, 0, 0].item() == pytest.approx(5.0)
    assert fallback.next_current_state.position[0, 0].item() == pytest.approx(10.0)


def test_rejects_successful_solution_outside_command_limits() -> None:
    class SequenceSolver(_Solver):
        def __init__(self) -> None:
            super().__init__(torch.tensor([[True] * 12]))
            self.calls = 0

        def optimize_action_sequence(
            self, current_state: JointState
        ) -> SimpleNamespace:
            position = current_state.position[:, None, :].expand(1, 12, 1).clone()
            full = JointState.from_position(position, joint_names=self.joint_names)
            full.velocity = torch.zeros_like(position)
            full.acceleration = torch.zeros_like(position)
            if self.calls:
                full.acceleration.fill_(2.0)
            self.calls += 1
            return SimpleNamespace(
                success=torch.tensor([True]),
                robot_state_sequence=SimpleNamespace(joint_state=full),
                solve_time=0.001,
            )

        def reset_robot(self, current_state: JointState) -> None:
            pass

    limit = torch.ones(1)
    planner = PredictedStateMpc(
        SequenceSolver(),
        _timing(),
        MpcCommandLimits(limit, limit, limit),
    )
    planner.setup(_state())

    planner.step()
    fallback = planner.step()

    assert fallback.rejected_by_command_limits
    assert fallback.used_feasible_tail_fallback
