"""Time-indexed, application-owned servo trajectory queue."""

from __future__ import annotations

import torch
from curobo.types import JointState


def slice_joint_trajectory(state: JointState, start: int, stop: int) -> JointState:
    if start < 0 or stop <= start or stop > state.position.shape[1]:
        raise RuntimeError("servo command queue is exhausted")
    result = JointState.from_position(
        state.position[:, start:stop].clone(), joint_names=state.joint_names
    )
    for field in ("velocity", "acceleration", "jerk"):
        value = getattr(state, field)
        if isinstance(value, torch.Tensor):
            setattr(result, field, value[:, start:stop].clone())
    result.dt = state.dt.clone() if isinstance(state.dt, torch.Tensor) else None
    return result


def single_joint_state(state: JointState, index: int) -> JointState:
    sliced = slice_joint_trajectory(state, index, index + 1)
    result = JointState.from_position(
        sliced.position[:, 0], joint_names=sliced.joint_names
    )
    for field in ("velocity", "acceleration", "jerk"):
        value = getattr(sliced, field)
        if isinstance(value, torch.Tensor):
            setattr(result, field, value[:, 0].clone())
    result.dt = sliced.dt
    return result


def extend_stationary_trajectory(state: JointState, samples: int) -> JointState:
    available = state.position.shape[1]
    if samples <= available:
        return state
    if (
        float(state.velocity[:, -1].abs().max().item()) > 1.0e-5
        or float(state.acceleration[:, -1].abs().max().item()) > 1.0e-4
    ):
        raise RuntimeError("command queue ended before reaching rest")
    extra = samples - available
    result = state.clone()
    result.position = torch.cat(
        (state.position, state.position[:, -1:].expand(-1, extra, -1)), dim=1
    )
    for field in ("velocity", "acceleration", "jerk"):
        value = getattr(state, field)
        if isinstance(value, torch.Tensor):
            setattr(
                result,
                field,
                torch.cat((value, value.new_zeros((1, extra, value.shape[-1]))), dim=1),
            )
    return result


class ServoTrajectoryQueue:
    """Consume a verified trajectory while a new one is being generated."""

    def __init__(self, state: JointState) -> None:
        self._state = state.clone()
        self._head = 0

    @property
    def remaining_samples(self) -> int:
        return self._state.position.shape[1] - self._head

    def at_offset(self, offset: int) -> JointState:
        if offset < 0:
            raise ValueError("queue offset must be nonnegative")
        self._state = extend_stationary_trajectory(
            self._state, self._head + offset + 1
        )
        return single_joint_state(self._state, self._head + offset)

    def consume(self, samples: int) -> JointState:
        if samples < 1:
            raise ValueError("samples must be positive")
        self._state = extend_stationary_trajectory(
            self._state, self._head + samples + 1
        )
        result = slice_joint_trajectory(
            self._state, self._head, self._head + samples
        )
        self._head += samples
        return result

    def replace(self, state: JointState, tolerance: float = 2.0e-5) -> None:
        current = self.at_offset(0)
        first = single_joint_state(state, 0)
        errors = [
            float((getattr(current, name) - getattr(first, name)).abs().max().item())
            for name in ("position", "velocity", "acceleration")
        ]
        if max(errors) > tolerance:
            raise RuntimeError(f"new trajectory is discontinuous: {errors}")
        self._state = state.clone()
        self._head = 0
