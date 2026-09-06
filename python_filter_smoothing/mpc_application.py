"""Application-side policy for publishing MPC trajectories to a servo loop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import torch
from curobo.types import JointState

from .servo_queue import ServoPlanCommit, ServoPlanTicket, ServoTrajectoryQueue


class TrajectoryExecutionMode(str, Enum):
    """How an application makes a completed MPC trajectory visible to the servo."""

    IMMEDIATE = "immediate"
    FUTURE_QUEUE = "future_queue"


class CspaceAcceptanceMode(str, Enum):
    """How the application interprets cuRobo's aggregate cspace constraint."""

    STRICT = "strict"
    PHYSICAL_LIMITS = "physical_limits"


@dataclass(frozen=True)
class PoseError:
    """Cartesian error to the requested target."""

    position_m: float
    rotation_rad: float


@dataclass(frozen=True)
class ProgressEvaluation:
    """Application quality-gate result, independent of MPC feasibility."""

    accepted: bool
    reason: str
    position_improvement_m: float
    rotation_improvement_rad: float


@dataclass(frozen=True)
class JointLimitEvaluation:
    """Independent physical-limit check for one sampled command trajectory."""

    accepted: bool
    reason: str
    maximum_position_violation_rad: float
    maximum_velocity_ratio: float
    maximum_acceleration_ratio: float
    maximum_jerk_ratio: float


@dataclass(frozen=True)
class TrajectoryConstraintPolicy:
    """Application acceptance limits; collision constraints always remain strict."""

    cspace_mode: CspaceAcceptanceMode = CspaceAcceptanceMode.STRICT
    maximum_velocity_ratio: float = 1.001
    maximum_acceleration_ratio: float = 1.001
    maximum_jerk_ratio: float = 1.001
    position_tolerance_rad: float = 1.0e-6

    def __post_init__(self) -> None:
        object.__setattr__(self, "cspace_mode", CspaceAcceptanceMode(self.cspace_mode))
        if min(
            self.maximum_velocity_ratio,
            self.maximum_acceleration_ratio,
            self.maximum_jerk_ratio,
        ) <= 0.0:
            raise ValueError("maximum derivative ratios must be positive")
        if self.position_tolerance_rad < 0.0:
            raise ValueError("position_tolerance_rad must be nonnegative")

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any]) -> TrajectoryConstraintPolicy:
        """Load the policy from an application YAML mapping."""

        return cls(
            cspace_mode=CspaceAcceptanceMode(options.get("cspace_mode", "strict")),
            maximum_velocity_ratio=float(
                options.get("maximum_velocity_ratio", 1.001)
            ),
            maximum_acceleration_ratio=float(
                options.get("maximum_acceleration_ratio", 1.001)
            ),
            maximum_jerk_ratio=float(options.get("maximum_jerk_ratio", 1.001)),
            position_tolerance_rad=float(
                options.get("position_tolerance_rad", 1.0e-6)
            ),
        )

    @property
    def ignored_curobo_constraints(self) -> frozenset[str]:
        """Return only constraints replaced by an independent application check."""

        if self.cspace_mode is CspaceAcceptanceMode.PHYSICAL_LIMITS:
            return frozenset({"cspace"})
        return frozenset()


def evaluate_joint_trajectory_limits(
    state: JointState,
    initial_state: JointState,
    *,
    dt_s: float,
    minimum_position: torch.Tensor,
    maximum_position: torch.Tensor,
    maximum_velocity: torch.Tensor,
    maximum_acceleration: torch.Tensor,
    maximum_jerk: torch.Tensor,
    policy: TrajectoryConstraintPolicy,
) -> JointLimitEvaluation:
    """Check actual q/dq/ddq/discrete-jerk without using cuRobo cost values."""

    tensors = (state.position, state.velocity, state.acceleration)
    if dt_s <= 0.0 or any(value is None or value.ndim != 3 for value in tensors):
        raise ValueError("trajectory q/dq/ddq must be [batch, samples, dof]")
    if initial_state.acceleration is None:
        raise ValueError("initial_state acceleration is required")
    finite = all(bool(torch.all(torch.isfinite(value)).item()) for value in tensors)
    acceleration = state.acceleration
    initial_acceleration = initial_state.acceleration.reshape(
        acceleration.shape[0], 1, acceleration.shape[-1]
    )
    jerk = torch.diff(torch.cat((initial_acceleration, acceleration), dim=1), dim=1)
    jerk = jerk / dt_s
    lower_violation = (minimum_position - state.position).clamp_min(0.0)
    upper_violation = (state.position - maximum_position).clamp_min(0.0)
    position_violation = float(
        torch.maximum(lower_violation, upper_violation).max().item()
    )
    ratios = (
        float((state.velocity.abs() / maximum_velocity).max().item()),
        float((state.acceleration.abs() / maximum_acceleration).max().item()),
        float((jerk.abs() / maximum_jerk).max().item()),
    )
    if not finite:
        reason = "non_finite"
    elif position_violation > policy.position_tolerance_rad:
        reason = "position_limit"
    elif ratios[0] > policy.maximum_velocity_ratio:
        reason = "velocity_limit"
    elif ratios[1] > policy.maximum_acceleration_ratio:
        reason = "acceleration_limit"
    elif ratios[2] > policy.maximum_jerk_ratio:
        reason = "jerk_limit"
    else:
        reason = "accepted"
    return JointLimitEvaluation(
        reason == "accepted",
        reason,
        position_violation,
        ratios[0],
        ratios[1],
        ratios[2],
    )


@dataclass(frozen=True)
class PoseProgressPolicy:
    """Require a candidate endpoint to approach the Cartesian target."""

    enabled: bool = True
    minimum_position_improvement_m: float = 0.002
    position_tolerance_m: float = 0.015
    check_rotation: bool = False
    minimum_rotation_improvement_rad: float = 0.01
    rotation_tolerance_rad: float = 0.08

    def __post_init__(self) -> None:
        values = (
            self.minimum_position_improvement_m,
            self.position_tolerance_m,
            self.minimum_rotation_improvement_rad,
            self.rotation_tolerance_rad,
        )
        if any(value < 0.0 for value in values):
            raise ValueError("progress thresholds must be nonnegative")

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any]) -> PoseProgressPolicy:
        """Load the policy from an application YAML mapping."""

        return cls(
            enabled=bool(options.get("enabled", True)),
            minimum_position_improvement_m=float(
                options.get("minimum_position_improvement_m", 0.002)
            ),
            position_tolerance_m=float(options.get("position_tolerance_m", 0.015)),
            check_rotation=bool(options.get("check_rotation", False)),
            minimum_rotation_improvement_rad=float(
                options.get("minimum_rotation_improvement_rad", 0.01)
            ),
            rotation_tolerance_rad=float(
                options.get("rotation_tolerance_rad", 0.08)
            ),
        )

    def evaluate(self, initial: PoseError, terminal: PoseError) -> ProgressEvaluation:
        """Accept a path that improves each enabled component or reaches tolerance."""

        position_improvement = initial.position_m - terminal.position_m
        rotation_improvement = initial.rotation_rad - terminal.rotation_rad
        if not self.enabled:
            return ProgressEvaluation(
                True, "disabled", position_improvement, rotation_improvement
            )

        position_ok = (
            terminal.position_m <= self.position_tolerance_m
            or position_improvement >= self.minimum_position_improvement_m
        )
        rotation_ok = not self.check_rotation or (
            terminal.rotation_rad <= self.rotation_tolerance_rad
            or rotation_improvement >= self.minimum_rotation_improvement_rad
        )
        if not position_ok:
            reason = "no_position_progress"
        elif not rotation_ok:
            reason = "no_rotation_progress"
        else:
            reason = "accepted"
        return ProgressEvaluation(
            position_ok and rotation_ok,
            reason,
            position_improvement,
            rotation_improvement,
        )


@dataclass(frozen=True)
class ApplicationPlan:
    """State and queue ticket captured before starting one MPC calculation."""

    initial_state: JointState
    ticket: ServoPlanTicket


class MpcCommandApplication:
    """Application-owned trajectory publisher with selectable queue semantics.

    ``future_queue`` reserves a future state and keeps consuming the old trajectory
    during calculation. ``immediate`` plans from the current state and ignores
    calculation time, which is useful for offline or blocking examples.
    """

    def __init__(
        self,
        initial_trajectory: JointState,
        *,
        mode: str | TrajectoryExecutionMode = TrajectoryExecutionMode.FUTURE_QUEUE,
        connection_samples: int = 0,
    ) -> None:
        self.mode = TrajectoryExecutionMode(mode)
        if connection_samples < 0:
            raise ValueError("connection_samples must be nonnegative")
        if self.mode is TrajectoryExecutionMode.FUTURE_QUEUE and connection_samples < 1:
            raise ValueError("future_queue requires at least one connection sample")
        self.connection_samples = connection_samples
        self._queue = ServoTrajectoryQueue(initial_trajectory)

    @property
    def current_state(self) -> JointState:
        return self._queue.current_state

    def begin_plan(self) -> ApplicationPlan:
        """Capture the q/dq/ddq state from which the application wants a plan."""

        offset = (
            self.connection_samples
            if self.mode is TrajectoryExecutionMode.FUTURE_QUEUE
            else 0
        )
        ticket = self._queue.begin_plan(offset)
        return ApplicationPlan(ticket.initial_state, ticket)

    def consume_during_planning(self, samples: int) -> JointState | None:
        """Advance the old path only when planning and servo execution overlap."""

        if self.mode is TrajectoryExecutionMode.IMMEDIATE:
            return None
        return self.consume(samples)

    def publish(
        self, plan: ApplicationPlan, trajectory: JointState, tolerance: float = 2.0e-5
    ) -> ServoPlanCommit:
        """Atomically publish a validated trajectory at the requested connection."""

        return self._queue.commit_plan(plan.ticket, trajectory, tolerance)

    def consume(self, samples: int) -> JointState:
        """Return the next servo commands from the currently published path."""

        return self._queue.consume(samples)
