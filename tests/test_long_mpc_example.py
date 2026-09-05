from types import SimpleNamespace

from long_mpc_example import _rank_feasible_candidates, _RawCandidate


def test_archive_keeps_earlier_feasible_result_when_later_solve_regresses() -> None:
    candidates = [
        _RawCandidate(50, SimpleNamespace(full_horizon_feasible=True), 0.20, 0.10),
        _RawCandidate(100, SimpleNamespace(full_horizon_feasible=True), 0.10, 0.10),
        _RawCandidate(200, SimpleNamespace(full_horizon_feasible=False), 0.01, 0.01),
    ]

    ranked = _rank_feasible_candidates(candidates)

    assert [candidate.iterations for candidate in ranked] == [100, 50]
