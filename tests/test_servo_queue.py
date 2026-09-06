from __future__ import annotations

import pytest
import torch
from curobo.types import JointState

from python_filter_smoothing.servo_queue import ServoTrajectoryQueue


def _trajectory(values: list[float], terminal_velocity: float = 0.0) -> JointState:
    position = torch.tensor(values).reshape(1, -1, 1)
    state = JointState.from_position(position, joint_names=["j1"])
    state.velocity = torch.zeros_like(position)
    state.velocity[:, -1] = terminal_velocity
    state.acceleration = torch.zeros_like(position)
    state.jerk = torch.zeros_like(position)
    return state


def test_consumes_commands_and_reports_future_connection_state() -> None:
    queue = ServoTrajectoryQueue(_trajectory([0.0, 1.0, 2.0, 3.0]))

    assert queue.at_offset(2).position.item() == pytest.approx(2.0)
    assert queue.consume(2).position.flatten().tolist() == pytest.approx([0.0, 1.0])
    assert queue.remaining_samples == 2


def test_stationary_tail_can_be_reused_after_path_exhaustion() -> None:
    queue = ServoTrajectoryQueue(_trajectory([0.0, 1.0]))

    assert queue.consume(5).position.flatten().tolist() == pytest.approx(
        [0.0, 1.0, 1.0, 1.0, 1.0]
    )


def test_nonstationary_tail_cannot_be_extended() -> None:
    queue = ServoTrajectoryQueue(_trajectory([0.0, 1.0], terminal_velocity=0.1))

    with pytest.raises(RuntimeError, match="ended before reaching rest"):
        queue.consume(3)


def test_future_plan_splices_after_commands_consumed_while_solving() -> None:
    queue = ServoTrajectoryQueue(_trajectory([0.0, 1.0, 2.0, 3.0, 4.0]))
    ticket = queue.begin_plan(3)
    candidate = _trajectory([3.0, 3.5, 4.0])

    commands_during_solve = queue.consume(2)
    commit = queue.commit_plan(ticket, candidate)
    commands_after_solve = queue.consume(3)

    assert ticket.initial_state.position.item() == pytest.approx(3.0)
    assert commands_during_solve.position.flatten().tolist() == pytest.approx(
        [0.0, 1.0]
    )
    assert commit.accepted
    assert commit.samples_until_splice == 1
    assert commands_after_solve.position.flatten().tolist() == pytest.approx(
        [2.0, 3.0, 3.5]
    )


def test_plan_that_finishes_after_reserved_splice_is_rejected() -> None:
    queue = ServoTrajectoryQueue(_trajectory([0.0, 1.0, 2.0, 3.0]))
    ticket = queue.begin_plan(2)

    queue.consume(3)
    commit = queue.commit_plan(ticket, _trajectory([2.0, 2.5, 3.0]))

    assert not commit.accepted
    assert commit.reason == "late"
    assert queue.current_state.position.item() == pytest.approx(3.0)


def test_failed_plan_leaves_previous_trajectory_available() -> None:
    queue = ServoTrajectoryQueue(_trajectory([0.0, 1.0, 2.0, 3.0]))
    queue.begin_plan(2)

    queue.consume(2)

    assert queue.consume(2).position.flatten().tolist() == pytest.approx([2.0, 3.0])


def test_retry_uses_future_state_after_progress_during_failed_plan() -> None:
    queue = ServoTrajectoryQueue(_trajectory([0.0, 1.0, 2.0, 3.0, 4.0]))
    first = queue.begin_plan(2)

    queue.consume(1)  # The first plan failed; do not commit it.
    retry = queue.begin_plan(2)

    assert first.initial_state.position.item() == pytest.approx(2.0)
    assert retry.initial_state.position.item() == pytest.approx(3.0)


def test_older_ticket_cannot_overwrite_a_newer_committed_plan() -> None:
    queue = ServoTrajectoryQueue(_trajectory([0.0, 1.0, 2.0, 3.0]))
    old = queue.begin_plan(2)
    new = queue.begin_plan(1)

    assert queue.commit_plan(new, _trajectory([1.0, 1.5, 2.0])).accepted
    stale = queue.commit_plan(old, _trajectory([2.0, 2.5, 3.0]))

    assert not stale.accepted
    assert stale.reason == "stale"


@pytest.mark.parametrize("field", ["velocity", "acceleration"])
def test_future_plan_rejects_derivative_discontinuity(field: str) -> None:
    queue = ServoTrajectoryQueue(_trajectory([0.0, 1.0, 2.0]))
    ticket = queue.begin_plan(1)
    candidate = _trajectory([1.0, 1.5, 2.0])
    getattr(candidate, field)[:, 0] = 0.1

    commit = queue.commit_plan(ticket, candidate)

    assert not commit.accepted
    assert commit.reason == "discontinuous"
    assert commit.continuity_errors is not None
    assert queue.consume(2).position.flatten().tolist() == pytest.approx([0.0, 1.0])
