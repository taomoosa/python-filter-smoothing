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


def test_replacement_requires_q_dq_ddq_continuity() -> None:
    queue = ServoTrajectoryQueue(_trajectory([0.0, 1.0, 2.0]))
    queue.consume(1)
    queue.replace(_trajectory([1.0, 1.5, 2.0]))

    assert queue.consume(2).position.flatten().tolist() == pytest.approx([1.0, 1.5])
    with pytest.raises(RuntimeError, match="discontinuous"):
        queue.replace(_trajectory([9.0, 10.0]))


def test_stationary_tail_can_be_reused_after_path_exhaustion() -> None:
    queue = ServoTrajectoryQueue(_trajectory([0.0, 1.0]))

    assert queue.consume(5).position.flatten().tolist() == pytest.approx(
        [0.0, 1.0, 1.0, 1.0, 1.0]
    )


def test_nonstationary_tail_cannot_be_extended() -> None:
    queue = ServoTrajectoryQueue(_trajectory([0.0, 1.0], terminal_velocity=0.1))

    with pytest.raises(RuntimeError, match="ended before reaching rest"):
        queue.consume(3)
