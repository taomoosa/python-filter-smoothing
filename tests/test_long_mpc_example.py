from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from curobo.types import JointState

from long_mpc_example import (
    _artifact_reference,
    _initial_state,
    _rank_feasible_candidates,
    _RawCandidate,
)
from python_filter_smoothing.continuous_trajectory import _weight_sequence
from python_filter_smoothing.mpc_application import ProgressEvaluation


def _fake_controller() -> SimpleNamespace:
    names = ["joint_a", "joint_b", "joint_c"]
    return SimpleNamespace(
        joint_names=names,
        default_state=lambda: JointState.from_position(
            torch.tensor([[0.1, 0.2, 0.3]]), joint_names=names
        ),
    )


def test_artifact_reference_does_not_expose_absolute_path(tmp_path: Path) -> None:
    output = tmp_path / "artifacts" / "run"
    config = tmp_path / "configs" / "robot.yml"

    reference = _artifact_reference(config, output)

    assert reference == "../../configs/robot.yml"
    assert not Path(reference).is_absolute()


def test_initial_state_accepts_partial_joint_name_mapping() -> None:
    state = _initial_state(
        _fake_controller(), {"initial_joint_positions_rad": {"joint_b": -0.4}}
    )

    torch.testing.assert_close(state.position, torch.tensor([[0.1, -0.4, 0.3]]))


def test_initial_state_accepts_complete_joint_order_sequence() -> None:
    state = _initial_state(
        _fake_controller(), {"initial_joint_positions_rad": [0.4, 0.5, 0.6]}
    )

    torch.testing.assert_close(state.position, torch.tensor([[0.4, 0.5, 0.6]]))


@pytest.mark.parametrize(
    "configured",
    ({"missing_joint": 0.0}, [0.1, 0.2], [0.1, float("nan"), 0.3]),
)
def test_initial_state_rejects_invalid_joint_configuration(configured: object) -> None:
    with pytest.raises(ValueError):
        _initial_state(
            _fake_controller(), {"initial_joint_positions_rad": configured}
        )


def test_optimizer_weight_validation_is_mechanism_profile_friendly() -> None:
    assert _weight_sequence([1, 2, 3, 4, 0], "weights", 5) == [
        1.0,
        2.0,
        3.0,
        4.0,
        0.0,
    ]
    with pytest.raises(ValueError):
        _weight_sequence([1, 2, -3, 4, 0], "weights", 5)


def test_archive_keeps_earlier_feasible_result_when_later_solve_regresses() -> None:
    candidates = [
        _RawCandidate(50, SimpleNamespace(full_horizon_feasible=True), 0.20, 0.10),
        _RawCandidate(100, SimpleNamespace(full_horizon_feasible=True), 0.10, 0.10),
        _RawCandidate(200, SimpleNamespace(full_horizon_feasible=False), 0.01, 0.01),
    ]

    ranked = _rank_feasible_candidates(candidates)

    assert [candidate.iterations for candidate in ranked] == [100, 50]


def test_archive_rejects_feasible_candidate_without_application_progress() -> None:
    rejected = ProgressEvaluation(False, "no_position_progress", -0.01, 0.0)
    accepted = ProgressEvaluation(True, "accepted", 0.01, 0.0)
    candidates = [
        _RawCandidate(
            50, SimpleNamespace(full_horizon_feasible=True), 0.05, 0.0, rejected
        ),
        _RawCandidate(
            100, SimpleNamespace(full_horizon_feasible=True), 0.06, 0.0, accepted
        ),
    ]

    ranked = _rank_feasible_candidates(candidates)

    assert [candidate.iterations for candidate in ranked] == [100]
