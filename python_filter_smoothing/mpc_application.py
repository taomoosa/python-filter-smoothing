"""Application-side policy for publishing MPC trajectories to a servo loop."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol

import torch
from curobo.types import JointState

from .servo_queue import ServoPlanCommit, ServoPlanTicket, ServoTrajectoryQueue


class TrajectoryExecutionMode(str, Enum):
    """How an application makes a completed MPC trajectory visible to the servo."""

    IMMEDIATE = "immediate"
    FUTURE_QUEUE = "future_queue"


class PathGenerationMode(str, Enum):
    """How the application obtains a validated joint trajectory."""

    MPC = "mpc"
    DIRECT_RUCKIG = "direct_ruckig"
    DIRECT_THEN_MPC = "direct_then_mpc"


class CspaceAcceptanceMode(str, Enum):
    """How the application interprets cuRobo's aggregate cspace constraint."""

    STRICT = "strict"
    PHYSICAL_LIMITS = "physical_limits"


@dataclass(frozen=True)
class ValidatedCartesianTarget:
    """One Cartesian target with an IK solution accepted by the controller."""

    position_m: torch.Tensor
    quaternion_wxyz: torch.Tensor
    joint_state: JointState


@dataclass(frozen=True)
class TargetResolution:
    """Requested and selected targets plus application telemetry."""

    requested_position_m: torch.Tensor
    requested_quaternion_wxyz: torch.Tensor
    selected: ValidatedCartesianTarget
    used_proxy: bool
    reason: str
    ik_attempts: int
    retreat_distance_m: float
    orientation_fraction: float


@dataclass(frozen=True)
class TargetResolutionPolicy:
    """Generic IK-feasible proxy search from a requested pose toward the robot."""

    enabled: bool = True
    coarse_position_samples: int = 8
    refinement_iterations: int = 4
    clearance_m: float = 0.02
    orientation_fractions: tuple[float, ...] = (1.0, 0.75, 0.5, 0.0)

    def __post_init__(self) -> None:
        if (
            isinstance(self.coarse_position_samples, bool)
            or not isinstance(self.coarse_position_samples, int)
            or self.coarse_position_samples < 1
        ):
            raise ValueError("coarse_position_samples must be a positive integer")
        if (
            isinstance(self.refinement_iterations, bool)
            or not isinstance(self.refinement_iterations, int)
            or self.refinement_iterations < 0
        ):
            raise ValueError("refinement_iterations must be a nonnegative integer")
        if not math.isfinite(self.clearance_m) or self.clearance_m < 0.0:
            raise ValueError("clearance_m must be finite and nonnegative")
        if not self.orientation_fractions:
            raise ValueError("orientation_fractions must not be empty")
        if any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in self.orientation_fractions
        ):
            raise ValueError("orientation_fractions must be within [0, 1]")
        if len(set(self.orientation_fractions)) != len(self.orientation_fractions):
            raise ValueError("orientation_fractions must not contain duplicates")

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any]) -> TargetResolutionPolicy:
        """Load the target-resolution policy from application YAML."""

        return cls(
            enabled=bool(options.get("enabled", True)),
            coarse_position_samples=options.get("coarse_position_samples", 8),
            refinement_iterations=options.get("refinement_iterations", 4),
            clearance_m=float(options.get("clearance_m", 0.02)),
            orientation_fractions=tuple(
                float(value)
                for value in options.get("orientation_fractions", (1.0, 0.75, 0.5, 0.0))
            ),
        )


class TargetResolutionError(RuntimeError):
    """No IK-feasible exact or proxy target could be found."""


class _TargetSelection(Protocol):
    position_m: torch.Tensor
    quaternion_wxyz: torch.Tensor
    joint_state: JointState


class _TargetController(Protocol):
    """Small controller surface used by the application-side target resolver."""

    @property
    def last_target_selection(self) -> _TargetSelection | None: ...

    def tool_pose(self, state: JointState) -> tuple[torch.Tensor, torch.Tensor]: ...

    def try_target_ik(
        self,
        position_m: torch.Tensor,
        quaternion_wxyz: torch.Tensor,
    ) -> _TargetSelection | None: ...

    def set_target(
        self,
        position_m: torch.Tensor,
        quaternion_wxyz: torch.Tensor | None = None,
        *,
        validate_ik: bool = False,
        joint_reference: JointState | None = None,
    ) -> JointState | None: ...


def _quaternion_slerp(
    start_wxyz: torch.Tensor, end_wxyz: torch.Tensor, fraction: float
) -> torch.Tensor:
    """Shortest-path quaternion interpolation for one WXYZ pair."""

    start = start_wxyz.reshape(4)
    end = end_wxyz.reshape(4)
    start = start / torch.linalg.vector_norm(start).clamp_min(1.0e-12)
    end = end / torch.linalg.vector_norm(end).clamp_min(1.0e-12)
    dot = torch.dot(start, end)
    if bool((dot < 0.0).item()):
        end = -end
        dot = -dot
    dot = torch.clamp(dot, 0.0, 1.0)
    if bool((dot > 0.9995).item()):
        result = start + fraction * (end - start)
        return result / torch.linalg.vector_norm(result).clamp_min(1.0e-12)
    angle = torch.acos(dot)
    scale = torch.sin(angle).clamp_min(1.0e-12)
    return (
        torch.sin((1.0 - fraction) * angle) / scale * start
        + torch.sin(fraction * angle) / scale * end
    )


_TargetValidator = Callable[
    [torch.Tensor, torch.Tensor], ValidatedCartesianTarget | None
]


class CartesianTargetResolver:
    """Install an IK-feasible exact or proxy Cartesian goal.

    The exact pose is tried first. On failure, candidates are sampled on the line
    from the requested position toward the current collision-free tool position.
    The nearest feasible interval is refined, then shifted toward the current pose
    by the configured clearance. Requested orientation is preferred and relaxed
    only after exhausting the positional search for that orientation.
    """

    def __init__(self, policy: TargetResolutionPolicy | None = None) -> None:
        self.policy = policy or TargetResolutionPolicy()

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any]) -> CartesianTargetResolver:
        """Load and construct a resolver from application YAML."""

        return cls(TargetResolutionPolicy.from_mapping(options))

    def set_target(
        self,
        controller: _TargetController,
        current_state: JointState,
        requested_position_m: torch.Tensor,
        requested_quaternion_wxyz: torch.Tensor,
    ) -> TargetResolution:
        """Resolve, validate, and install one target on an MPC controller."""

        requested_position = requested_position_m.reshape(3).clone()
        requested_quaternion = requested_quaternion_wxyz.reshape(4).clone()
        current_position, current_quaternion = controller.tool_pose(current_state)
        current_position = current_position.reshape(3)
        current_quaternion = current_quaternion.reshape(4)
        values = (
            requested_position,
            requested_quaternion,
            current_position,
            current_quaternion,
        )
        if not all(bool(torch.all(torch.isfinite(value)).item()) for value in values):
            raise ValueError("requested and current Cartesian poses must be finite")
        if (
            float(torch.linalg.vector_norm(requested_quaternion).item()) <= 1.0e-12
            or float(torch.linalg.vector_norm(current_quaternion).item()) <= 1.0e-12
        ):
            raise ValueError("requested and current quaternions must be nonzero")

        attempts = 0

        def validate(
            position: torch.Tensor, quaternion: torch.Tensor
        ) -> ValidatedCartesianTarget | None:
            nonlocal attempts
            attempts += 1
            selection = controller.try_target_ik(position, quaternion)
            if selection is None:
                return None
            return ValidatedCartesianTarget(
                position_m=selection.position_m.clone(),
                quaternion_wxyz=selection.quaternion_wxyz.clone(),
                joint_state=selection.joint_state,
            )

        selected, orientation_fraction = self._resolve(
            requested_position,
            requested_quaternion,
            current_position,
            current_quaternion,
            validate,
        )
        result = self._resolution(
            requested_position,
            requested_quaternion,
            selected,
            attempts,
            orientation_fraction,
        )
        controller.set_target(
            result.selected.position_m,
            result.selected.quaternion_wxyz,
            joint_reference=result.selected.joint_state,
        )
        return result

    def _resolve(
        self,
        requested_position: torch.Tensor,
        requested_quaternion: torch.Tensor,
        current_position: torch.Tensor,
        current_quaternion: torch.Tensor,
        validate: _TargetValidator,
    ) -> tuple[ValidatedCartesianTarget, float]:
        exact = validate(requested_position, requested_quaternion)
        if exact is not None:
            return exact, 1.0
        if not self.policy.enabled:
            raise TargetResolutionError(
                "exact Cartesian target has no IK solution and proxy search is disabled"
            )

        direction = current_position - requested_position
        distance = float(torch.linalg.vector_norm(direction).item())
        for orientation_fraction in self.policy.orientation_fractions:
            quaternion = _quaternion_slerp(
                current_quaternion, requested_quaternion, orientation_fraction
            )
            lower_fraction = 0.0
            best: ValidatedCartesianTarget | None = None
            upper_fraction = 0.0
            for index in range(1, self.policy.coarse_position_samples + 1):
                fraction = index / self.policy.coarse_position_samples
                candidate = validate(
                    requested_position + fraction * direction, quaternion
                )
                if candidate is not None:
                    best = candidate
                    upper_fraction = fraction
                    break
                lower_fraction = fraction
            if best is None:
                continue

            for _ in range(self.policy.refinement_iterations):
                fraction = 0.5 * (lower_fraction + upper_fraction)
                candidate = validate(
                    requested_position + fraction * direction, quaternion
                )
                if candidate is None:
                    lower_fraction = fraction
                else:
                    best = candidate
                    upper_fraction = fraction

            if distance > 1.0e-12 and self.policy.clearance_m > 0.0:
                clearance_fraction = min(
                    1.0, upper_fraction + self.policy.clearance_m / distance
                )
                candidate = validate(
                    requested_position + clearance_fraction * direction,
                    quaternion,
                )
                if candidate is not None:
                    best = candidate
            return best, orientation_fraction

        raise TargetResolutionError("no IK-feasible Cartesian proxy found")

    @staticmethod
    def _resolution(
        requested_position: torch.Tensor,
        requested_quaternion: torch.Tensor,
        selected: ValidatedCartesianTarget,
        attempts: int,
        orientation_fraction: float,
    ) -> TargetResolution:
        retreat = float(
            torch.linalg.vector_norm(selected.position_m - requested_position).item()
        )
        requested = requested_quaternion / torch.linalg.vector_norm(
            requested_quaternion
        ).clamp_min(1.0e-12)
        actual = selected.quaternion_wxyz / torch.linalg.vector_norm(
            selected.quaternion_wxyz
        ).clamp_min(1.0e-12)
        orientation_changed = float(torch.abs(torch.dot(requested, actual)).item()) < (
            1.0 - 1.0e-6
        )
        used_proxy = retreat > 1.0e-6 or orientation_changed
        return TargetResolution(
            requested_position_m=requested_position.clone(),
            requested_quaternion_wxyz=requested_quaternion.clone(),
            selected=selected,
            used_proxy=used_proxy,
            reason="proxy" if used_proxy else "exact",
            ik_attempts=attempts,
            retreat_distance_m=retreat,
            orientation_fraction=orientation_fraction,
        )


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
        if (
            min(
                self.maximum_velocity_ratio,
                self.maximum_acceleration_ratio,
                self.maximum_jerk_ratio,
            )
            <= 0.0
        ):
            raise ValueError("maximum derivative ratios must be positive")
        if self.position_tolerance_rad < 0.0:
            raise ValueError("position_tolerance_rad must be nonnegative")

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any]) -> TrajectoryConstraintPolicy:
        """Load the policy from an application YAML mapping."""

        return cls(
            cspace_mode=CspaceAcceptanceMode(options.get("cspace_mode", "strict")),
            maximum_velocity_ratio=float(options.get("maximum_velocity_ratio", 1.001)),
            maximum_acceleration_ratio=float(
                options.get("maximum_acceleration_ratio", 1.001)
            ),
            maximum_jerk_ratio=float(options.get("maximum_jerk_ratio", 1.001)),
            position_tolerance_rad=float(options.get("position_tolerance_rad", 1.0e-6)),
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
            rotation_tolerance_rad=float(options.get("rotation_tolerance_rad", 0.08)),
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
