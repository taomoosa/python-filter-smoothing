from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from curobo.types import GoalToolPose, JointState

from python_filter_smoothing.continuous_trajectory import (
    ContinuousMpcTrajectory,
    MpcTargetSelection,
    _iteration_sequence,
    _position_offset_sequence,
    _positive_iterations,
)
from python_filter_smoothing.predictive_mpc import (
    MpcCommandLimits,
    MpcHorizonProducer,
    MpcTiming,
)


def _state(value: float = 0.0) -> JointState:
    state = JointState.from_position(torch.tensor([[value]]), joint_names=["j1"])
    state.velocity = torch.zeros_like(state.position)
    state.acceleration = torch.zeros_like(state.position)
    return state


class _Solver:
    joint_names: ClassVar[list[str]] = ["j1"]

    def __init__(self, success: bool, acceleration: float = 0.0) -> None:
        self.success = success
        self.acceleration = acceleration
        metrics = SimpleNamespace(feasible=torch.tensor([[True] * 6 + [False] * 6]))
        self.trajectory_execution_manager = SimpleNamespace(
            get_current_metrics=lambda: metrics
        )

    def setup(self, current_state: JointState) -> None:
        self.setup_state = current_state.clone()

    def optimize_action_sequence(self, current_state: JointState) -> SimpleNamespace:
        position = current_state.position[:, None, :].expand(1, 12, 1).clone()
        full = JointState.from_position(position, joint_names=self.joint_names)
        full.velocity = torch.zeros_like(position)
        full.acceleration = torch.full_like(position, self.acceleration)
        return SimpleNamespace(
            success=torch.tensor([self.success]),
            robot_state_sequence=SimpleNamespace(joint_state=full),
            solve_time=0.001,
        )


def _producer(solver: _Solver) -> MpcHorizonProducer:
    ones = torch.ones(1)
    return MpcHorizonProducer(
        solver,
        MpcTiming(optimization_dt_s=0.02, interpolation_steps=4),
        MpcCommandLimits(ones, ones, ones),
    )


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_optimizer_iterations_must_be_positive_integers(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="positive integer"):
        _positive_iterations(value, "optimizer.test_iterations")


def test_nearby_ik_offsets_require_exact_target_first() -> None:
    assert _position_offset_sequence(None, "offsets") == ((0.0, 0.0, 0.0),)
    with pytest.raises(ValueError, match="exact target"):
        _position_offset_sequence([[0.01, 0.0, 0.0]], "offsets")


@pytest.mark.parametrize("value", [[], [100, 50], [50, 50]])
def test_candidate_iterations_must_be_increasing(value: list[int]) -> None:
    with pytest.raises(ValueError, match="strictly increasing|nonempty"):
        _iteration_sequence(value, "optimizer.candidates")


def test_target_update_prepares_first_independent_candidate() -> None:
    class TargetSolver:
        def __init__(self) -> None:
            self.config = SimpleNamespace(cold_start_optimization_num_iters=300)
            self.reset_states: list[JointState] = []

        def update_goal_tool_poses(self, goal: object, *, run_ik: bool) -> bool:
            assert not run_ik
            return True

        def reset_robot(self, state: JointState) -> None:
            self.reset_states.append(state.clone())

    solver = TargetSolver()
    application = ContinuousMpcTrajectory.__new__(ContinuousMpcTrajectory)
    application.solver = solver
    application._planner = SimpleNamespace(current_state=_state())
    application._goal_request = SimpleNamespace(
        position=torch.zeros((1, 1, 1, 1, 3)),
        quaternion=torch.zeros((1, 1, 1, 1, 4)),
    )
    application._candidate_iterations = (50, 100, 200)
    application._use_ik_joint_reference = False
    application._seed_from_ik = False
    application._joint_reference_state = None

    application.set_target(
        torch.tensor([0.1, 0.2, 0.3]), torch.tensor([1.0, 0.0, 0.0, 0.0])
    )

    assert solver.config.cold_start_optimization_num_iters == 50
    assert len(solver.reset_states) == 1
    assert application._goal_request.position.flatten().tolist() == pytest.approx(
        [0.1, 0.2, 0.3]
    )


def test_candidate_reset_reapplies_joint_reference_and_unchecked_seed() -> None:
    class TargetSolver:
        def __init__(self) -> None:
            self.config = SimpleNamespace(cold_start_optimization_num_iters=50)
            self.calls: list[str] = []

        def reset_robot(self, state: JointState) -> None:
            self.calls.append("reset")

        def update_goal_state(self, state: JointState) -> None:
            self.calls.append("goal")

        def enable_joint_position_tracking(self) -> None:
            self.calls.append("track")

        def update_seed_trajectory_from_goal_state(self, state: JointState) -> None:
            self.calls.append("seed")

    solver = TargetSolver()
    application = ContinuousMpcTrajectory.__new__(ContinuousMpcTrajectory)
    application.solver = solver
    application._joint_reference_state = _state(0.5)
    application._seed_from_ik = True

    application.prepare_candidate(_state(), 200)

    assert solver.config.cold_start_optimization_num_iters == 200
    assert solver.calls == ["reset", "goal", "track", "seed"]


def test_nearby_ik_uses_first_solvable_offset_and_nearest_joint_branch() -> None:
    class ExactIk:
        def solve_pose(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                success=torch.tensor([[False]]),
                solution=torch.zeros((1, 1, 1)),
                position_error=torch.tensor([[0.1]]),
                rotation_error=torch.tensor([[0.2]]),
            )

    class NearbyIk:
        def reset_seed(self) -> None:
            self.reset = True

        def solve_pose(self, **kwargs: object) -> SimpleNamespace:
            self.goal = kwargs["goal_tool_poses"]
            return SimpleNamespace(
                success=torch.tensor(
                    [[False, False], [True, True], [True, True]]
                ),
                solution=torch.tensor(
                    [[[0.0], [0.0]], [[0.8], [0.3]], [[-0.1], [-0.2]]]
                ),
                position_error=torch.zeros((3, 2)),
                rotation_error=torch.zeros((3, 2)),
            )

    nearby = NearbyIk()
    solver = SimpleNamespace(
        joint_names=["j1"],
        ik_solver=ExactIk(),
        transition_model=SimpleNamespace(
            get_state_bounds=lambda: SimpleNamespace(
                position=torch.tensor([[-1.0], [1.0]])
            )
        ),
    )
    application = ContinuousMpcTrajectory.__new__(ContinuousMpcTrajectory)
    application.solver = solver
    application._fallback_ik = nearby
    application._ik_fallback_seeds = 2
    application._ik_position_offsets_m = (
        (0.0, 0.0, 0.0),
        (0.01, 0.0, 0.0),
        (0.0, 0.01, 0.0),
    )
    application._goal_request = GoalToolPose(
        tool_frames=["tool"],
        position=torch.tensor([[[[[0.4, 0.2, 0.1]]]]]),
        quaternion=torch.tensor([[[[[1.0, 0.0, 0.0, 0.0]]]]]),
    )

    selected = application._solve_target_ik(_state())

    assert selected.candidate_index == 1
    assert selected.position_m.tolist() == pytest.approx([0.41, 0.2, 0.1])
    assert selected.joint_state.position.item() == pytest.approx(0.3)
    assert nearby.goal.position.shape == (3, 1, 1, 1, 3)


def test_selected_nearby_pose_becomes_the_mpc_target() -> None:
    class TargetSolver:
        def update_goal_tool_poses(self, goal: GoalToolPose, *, run_ik: bool) -> bool:
            assert not run_ik
            self.position = goal.position.clone()
            return True

    application = ContinuousMpcTrajectory.__new__(ContinuousMpcTrajectory)
    application.solver = TargetSolver()
    application._planner = SimpleNamespace(current_state=_state())
    application._goal_request = GoalToolPose(
        tool_frames=["tool"],
        position=torch.zeros((1, 1, 1, 1, 3)),
        quaternion=torch.tensor([[[[[1.0, 0.0, 0.0, 0.0]]]]]),
    )
    application._candidate_iterations = (50,)
    application._use_ik_joint_reference = True
    application._joint_reference_state = None
    application._last_target_selection = None
    application.prepare_candidate = lambda state, iterations: None
    selection = MpcTargetSelection(
        candidate_index=1,
        position_m=torch.tensor([0.11, 0.2, 0.3]),
        quaternion_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        joint_state=_state(0.4),
    )
    application._solve_target_ik = lambda state: selection

    joint = application.set_target(
        torch.tensor([0.1, 0.2, 0.3]), torch.tensor([1.0, 0.0, 0.0, 0.0])
    )

    assert joint is selection.joint_state
    assert application.solver.position.flatten().tolist() == pytest.approx(
        [0.11, 0.2, 0.3]
    )
    assert application.last_target_selection is selection


def test_exposes_complete_infeasible_horizon_without_advancing_state() -> None:
    producer = _producer(_Solver(success=False))
    producer.setup(_state())

    horizon = producer.solve_horizon()

    assert not horizon.full_horizon_feasible
    assert horizon.feasible_prefix_length == 6
    assert horizon.states.position.shape[1] == 12
    assert producer.current_state.position.item() == pytest.approx(0.0)


def test_successful_horizon_is_exposed_without_advancing_state() -> None:
    producer = _producer(_Solver(success=True))
    producer.setup(_state(0.2))

    horizon = producer.solve_horizon()

    assert horizon.full_horizon_feasible
    assert not horizon.rejected_by_command_limits
    assert producer.current_state.position.item() == pytest.approx(0.2)


def test_rejects_successful_solution_outside_command_limits() -> None:
    producer = _producer(_Solver(success=True, acceleration=2.0))
    producer.setup(_state())

    horizon = producer.solve_horizon()

    assert not horizon.full_horizon_feasible
    assert horizon.rejected_by_command_limits


def test_timing_requires_curobo_interpolation_setting() -> None:
    with pytest.raises(ValueError, match="interpolation_steps=4"):
        MpcTiming(optimization_dt_s=0.02, interpolation_steps=2)
