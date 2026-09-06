"""Time-indexed, application-owned servo trajectory queue."""

from __future__ import annotations

from dataclasses import dataclass

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


def concatenate_joint_trajectories(first: JointState, second: JointState) -> JointState:
    """Join two sampled trajectories without changing either input."""

    result = JointState.from_position(
        torch.cat((first.position, second.position), dim=1),
        joint_names=first.joint_names,
    )
    for field in ("velocity", "acceleration", "jerk"):
        first_value = getattr(first, field)
        second_value = getattr(second, field)
        if isinstance(first_value, torch.Tensor) and isinstance(
            second_value, torch.Tensor
        ):
            setattr(result, field, torch.cat((first_value, second_value), dim=1))
    result.dt = second.dt if second.dt is not None else first.dt
    return result


@dataclass(frozen=True)
class ServoPlanTicket:
    """Snapshot identifying the future state used to initialize one plan."""

    initial_state: JointState
    splice_sample: int
    queue_revision: int


@dataclass(frozen=True)
class ServoPlanCommit:
    """Result of atomically replacing a future queue suffix."""

    accepted: bool
    reason: str
    samples_until_splice: int
    continuity_errors: tuple[float, float, float] | None = None


class ServoTrajectoryQueue:
    """Consume a verified trajectory while a new one is being generated."""

    def __init__(self, state: JointState) -> None:
        self._state = state.clone()
        self._head = 0
        self._consumed_samples = 0
        self._revision = 0

    @property
    def remaining_samples(self) -> int:
        return self._state.position.shape[1] - self._head

    @property
    def consumed_samples(self) -> int:
        return self._consumed_samples

    @property
    def current_state(self) -> JointState:
        return self.at_offset(0)

    def at_offset(self, offset: int) -> JointState:
        if offset < 0:
            raise ValueError("queue offset must be nonnegative")
        self._state = extend_stationary_trajectory(self._state, self._head + offset + 1)
        return single_joint_state(self._state, self._head + offset)

    def consume(self, samples: int) -> JointState:
        if samples < 1:
            raise ValueError("samples must be positive")
        self._state = extend_stationary_trajectory(
            self._state, self._head + samples + 1
        )
        result = slice_joint_trajectory(self._state, self._head, self._head + samples)
        self._head += samples
        self._consumed_samples += samples
        return result

    def begin_plan(self, samples_until_splice: int) -> ServoPlanTicket:
        """Reserve a future connection point and return its q/dq/ddq state."""

        if samples_until_splice < 0:
            raise ValueError("samples_until_splice must be nonnegative")
        return ServoPlanTicket(
            initial_state=self.at_offset(samples_until_splice),
            splice_sample=self._consumed_samples + samples_until_splice,
            queue_revision=self._revision,
        )

    def commit_plan(
        self,
        ticket: ServoPlanTicket,
        state: JointState,
        tolerance: float = 2.0e-5,
    ) -> ServoPlanCommit:
        """Replace the suffix at a reserved point, or reject a stale/late plan."""

        offset = ticket.splice_sample - self._consumed_samples
        if ticket.queue_revision != self._revision:
            return ServoPlanCommit(False, "stale", max(0, offset))
        if offset < 0:
            return ServoPlanCommit(False, "late", 0)

        expected = self.at_offset(offset)
        first = single_joint_state(state, 0)
        errors = [
            float((getattr(expected, name) - getattr(first, name)).abs().max().item())
            for name in ("position", "velocity", "acceleration")
        ]
        if max(errors) > tolerance:
            return ServoPlanCommit(False, "discontinuous", offset, tuple(errors))

        if offset:
            prefix = slice_joint_trajectory(
                self._state, self._head, self._head + offset
            )
            self._state = concatenate_joint_trajectories(prefix, state)
        else:
            self._state = state.clone()
        self._head = 0
        self._revision += 1
        return ServoPlanCommit(True, "accepted", offset, tuple(errors))
