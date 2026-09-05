from __future__ import annotations

import numpy as np
import pytest

from python_filter_smoothing.trajectory_resampling import (
    derivative_limit_ratios,
    resample_joint_trajectory,
)


def _options(method: str) -> dict[str, object]:
    return {
        "method": method,
        "hermite": {
            "initial_time_scale": 1.0,
            "time_scale_margin": 1.02,
            "max_time_scale_iterations": 8,
        },
        "ruckig": {
            "limit_margin": 1.0,
            "time_scale_margin": 1.02,
            "max_time_scale_iterations": 12,
        },
        "post_filter": {
            "method": "savgol_position",
            "window_length_samples": 5,
            "polynomial_order": 2,
        },
    }


@pytest.mark.parametrize("method", ["hermite", "ruckig"])
def test_resampler_is_selectable_and_respects_derivative_limits(method: str) -> None:
    q = np.asarray([[0.0], [0.01], [0.02], [0.025]])
    dq = np.asarray([[0.0], [0.1], [0.05], [0.0]])
    ddq = np.zeros_like(q)
    limits = (np.asarray([1.0]), np.asarray([10.0]), np.asarray([500.0]))

    result = resample_joint_trajectory(
        q,
        dq,
        ddq,
        source_dt_s=0.1,
        sample_dt_s=0.005,
        max_velocity=limits[0],
        max_acceleration=limits[1],
        max_jerk=limits[2],
        min_position=np.asarray([-1.0]),
        max_position=np.asarray([1.0]),
        options=_options(method),
    )

    assert result.method == method
    assert result.position.shape == result.velocity.shape == result.acceleration.shape
    assert np.array_equal(result.position[[0, -1]], q[[0, -1]])
    ratios = derivative_limit_ratios(
        result.velocity, result.acceleration, 0.005, *limits
    )
    assert max(ratios) <= 1.0 + 1.0e-8


def test_unknown_resampler_is_rejected() -> None:
    values = np.zeros((2, 1))
    with pytest.raises(ValueError, match="hermite or ruckig"):
        resample_joint_trajectory(
            values,
            values,
            values,
            source_dt_s=0.1,
            sample_dt_s=0.005,
            max_velocity=np.ones(1),
            max_acceleration=np.ones(1),
            max_jerk=np.ones(1),
            min_position=-np.ones(1),
            max_position=np.ones(1),
            options=_options("bad"),
        )
