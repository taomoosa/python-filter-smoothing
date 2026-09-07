from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from curobo.types import JointState

from python_filter_smoothing.mpc_application import (
    CartesianTargetResolver,
    CspaceAcceptanceMode,
    MpcCommandApplication,
    PoseError,
    PoseProgressPolicy,
    TargetResolutionError,
    TargetResolutionPolicy,
    TrajectoryConstraintPolicy,
    evaluate_joint_trajectory_limits,
)


def _trajectory(values: list[float]) -> JointState:
    position = torch.tensor(values).reshape(1, -1, 1)
    state = JointState.from_position(position, joint_names=["j1"])
    state.velocity = torch.zeros_like(position)
    state.acceleration = torch.zeros_like(position)
    state.jerk = torch.zeros_like(position)
    return state


class _TargetController:
    def __init__(self, minimum_feasible_x: float) -> None:
        self.minimum_feasible_x = minimum_feasible_x
        self.last_target_selection = None
        self.validation_count = 0
        self.installed_position: torch.Tensor | None = None
        self.installed_joint: JointState | None = None

    def tool_pose(self, state: JointState) -> tuple[torch.Tensor, torch.Tensor]:
        del state
        return torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.0, 0.0, 0.0, 0.0])

    def try_target_ik(
        self,
        position_m: torch.Tensor,
        quaternion_wxyz: torch.Tensor,
    ) -> SimpleNamespace | None:
        self.validation_count += 1
        if float(position_m[0]) < self.minimum_feasible_x:
            return None
        joint = _trajectory([float(position_m[0])])
        selection = SimpleNamespace(
            position_m=position_m.clone(),
            quaternion_wxyz=quaternion_wxyz.clone(),
            joint_state=joint,
        )
        self.last_target_selection = selection
        return selection

    def set_target(
        self,
        position_m: torch.Tensor,
        quaternion_wxyz: torch.Tensor | None = None,
        *,
        validate_ik: bool = False,
        joint_reference: JointState | None = None,
    ) -> JointState | None:
        assert quaternion_wxyz is not None
        assert not validate_ik
        assert joint_reference is not None
        self.installed_position = position_m.clone()
        self.installed_joint = joint_reference
        self.last_target_selection = SimpleNamespace(
            position_m=position_m.clone(),
            quaternion_wxyz=quaternion_wxyz.clone(),
        )
        return joint_reference


def test_progress_accepts_improvement_or_target_tolerance() -> None:
    policy = PoseProgressPolicy(
        minimum_position_improvement_m=0.002,
        position_tolerance_m=0.015,
    )

    improved = policy.evaluate(PoseError(0.100, 0.0), PoseError(0.097, 0.0))
    close = policy.evaluate(PoseError(0.016, 0.0), PoseError(0.014, 0.0))

    assert improved.accepted
    assert improved.position_improvement_m == pytest.approx(0.003)
    assert close.accepted


def test_progress_rejects_no_improvement() -> None:
    policy = PoseProgressPolicy()

    result = policy.evaluate(PoseError(0.100, 0.1), PoseError(0.101, 0.0))

    assert not result.accepted
    assert result.reason == "no_position_progress"


def test_rotation_progress_is_optional() -> None:
    initial = PoseError(0.100, 0.2)
    terminal = PoseError(0.090, 0.21)

    assert PoseProgressPolicy(check_rotation=False).evaluate(initial, terminal).accepted
    result = PoseProgressPolicy(check_rotation=True).evaluate(initial, terminal)
    assert not result.accepted
    assert result.reason == "no_rotation_progress"


def test_future_queue_keeps_executing_old_path_during_plan() -> None:
    application = MpcCommandApplication(
        _trajectory([0.0, 1.0, 2.0, 3.0]),
        mode="future_queue",
        connection_samples=2,
    )
    plan = application.begin_plan()

    old_commands = application.consume_during_planning(1)
    commit = application.publish(plan, _trajectory([2.0, 2.5, 3.0]))

    assert old_commands is not None
    assert old_commands.position.item() == pytest.approx(0.0)
    assert commit.accepted
    assert application.consume(3).position.flatten().tolist() == pytest.approx(
        [1.0, 2.0, 2.5]
    )


def test_immediate_mode_ignores_planning_time_and_replaces_now() -> None:
    application = MpcCommandApplication(_trajectory([0.0, 1.0, 2.0]), mode="immediate")
    plan = application.begin_plan()

    assert application.consume_during_planning(10) is None
    commit = application.publish(plan, _trajectory([0.0, 0.5, 1.0]))

    assert commit.accepted
    assert application.consume(3).position.flatten().tolist() == pytest.approx(
        [0.0, 0.5, 1.0]
    )


def test_future_queue_requires_a_connection_window() -> None:
    with pytest.raises(ValueError, match="at least one"):
        MpcCommandApplication(
            _trajectory([0.0, 1.0]), mode="future_queue", connection_samples=0
        )


def test_physical_limit_policy_ignores_only_curobo_cspace() -> None:
    policy = TrajectoryConstraintPolicy(cspace_mode="physical_limits")

    assert policy.cspace_mode is CspaceAcceptanceMode.PHYSICAL_LIMITS
    assert policy.ignored_curobo_constraints == frozenset({"cspace"})


def test_joint_limit_policy_can_allow_five_percent_discrete_jerk() -> None:
    initial = _trajectory([0.0])
    state = _trajectory([0.0, 0.0])
    state.acceleration[:] = torch.tensor([[[0.0], [1.04]]])
    common = {
        "dt_s": 1.0,
        "minimum_position": torch.tensor([-1.0]),
        "maximum_position": torch.tensor([1.0]),
        "maximum_velocity": torch.tensor([1.0]),
        "maximum_acceleration": torch.tensor([2.0]),
        "maximum_jerk": torch.tensor([1.0]),
    }

    strict = evaluate_joint_trajectory_limits(
        state,
        initial,
        policy=TrajectoryConstraintPolicy(maximum_jerk_ratio=1.001),
        **common,
    )
    relaxed = evaluate_joint_trajectory_limits(
        state,
        initial,
        policy=TrajectoryConstraintPolicy(maximum_jerk_ratio=1.05),
        **common,
    )

    assert not strict.accepted
    assert strict.reason == "jerk_limit"
    assert relaxed.accepted
    assert relaxed.maximum_jerk_ratio == pytest.approx(1.04)


def test_target_resolver_installs_exact_ik_target_without_search() -> None:
    controller = _TargetController(minimum_feasible_x=0.4)
    resolver = CartesianTargetResolver()

    result = resolver.set_target(
        controller,
        _trajectory([0.0]),
        torch.tensor([0.6, 0.0, 0.0]),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )

    assert not result.used_proxy
    assert result.reason == "exact"
    assert result.ik_attempts == 1
    assert controller.validation_count == 1
    torch.testing.assert_close(
        controller.installed_position, torch.tensor([0.6, 0.0, 0.0])
    )


def test_target_resolver_finds_nearest_proxy_then_adds_clearance() -> None:
    controller = _TargetController(minimum_feasible_x=0.4)
    resolver = CartesianTargetResolver(
        TargetResolutionPolicy(
            coarse_position_samples=4,
            refinement_iterations=2,
            clearance_m=0.1,
            orientation_fractions=(1.0,),
        )
    )

    result = resolver.set_target(
        controller,
        _trajectory([0.0]),
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )

    assert result.used_proxy
    assert result.reason == "proxy"
    assert result.ik_attempts == 6
    assert result.retreat_distance_m == pytest.approx(0.5375)
    assert result.orientation_fraction == pytest.approx(1.0)
    torch.testing.assert_close(
        controller.installed_position, torch.tensor([0.5375, 0.0, 0.0])
    )
    assert controller.installed_joint is result.selected.joint_state


def test_target_resolver_can_disable_proxy_search() -> None:
    controller = _TargetController(minimum_feasible_x=0.4)
    resolver = CartesianTargetResolver(TargetResolutionPolicy(enabled=False))

    with pytest.raises(TargetResolutionError, match="disabled"):
        resolver.set_target(
            controller,
            _trajectory([0.0]),
            torch.tensor([0.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
        )

    assert controller.validation_count == 1
    assert controller.installed_position is None


@pytest.mark.parametrize(
    "options",
    (
        {"coarse_position_samples": 0},
        {"coarse_position_samples": 2.5},
        {"refinement_iterations": -1},
        {"clearance_m": -0.01},
        {"orientation_fractions": []},
        {"orientation_fractions": [1.1]},
        {"orientation_fractions": [1.0, 1.0]},
    ),
)
def test_target_resolution_policy_rejects_invalid_options(
    options: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        TargetResolutionPolicy.from_mapping(options)
