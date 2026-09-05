from __future__ import annotations

import numpy as np
import pytest

from python_filter_smoothing.trajectory_filter import filter_joint_trajectory_savgol


def test_savgol_joint_filter_reduces_jerk_and_preserves_endpoint_states() -> None:
    dt = 0.1
    time_s = np.arange(9) * dt
    position = np.column_stack((time_s**3, -(time_s**3)))
    velocity = np.column_stack((3.0 * time_s**2, -(3.0 * time_s**2)))
    acceleration = np.column_stack((6.0 * time_s, -(6.0 * time_s)))
    position[4] += [0.02, -0.02]

    result = filter_joint_trajectory_savgol(
        position,
        velocity,
        acceleration,
        sample_dt_s=dt,
        window_length_samples=5,
        polynomial_order=2,
    )

    assert result.position.shape == position.shape
    assert np.array_equal(result.position[[0, -1]], position[[0, -1]])
    assert np.array_equal(result.velocity[[0, -1]], velocity[[0, -1]])
    assert np.array_equal(result.acceleration[[0, -1]], acceleration[[0, -1]])
    assert np.max(np.abs(result.position[1:-1] - position[1:-1])) > 0.0


def test_savgol_joint_filter_preserves_a_quadratic_and_its_derivatives() -> None:
    dt = 0.1
    time_s = np.arange(9) * dt
    position = (0.3 + 0.4 * time_s + 0.5 * time_s**2)[:, None]
    velocity = (0.4 + time_s)[:, None]
    acceleration = np.ones((9, 1))

    result = filter_joint_trajectory_savgol(
        position,
        velocity,
        acceleration,
        sample_dt_s=dt,
        window_length_samples=5,
        polynomial_order=2,
    )

    assert np.allclose(result.position, position)
    assert np.allclose(result.velocity, velocity)
    assert np.allclose(result.acceleration, acceleration)


@pytest.mark.parametrize("window", [2, 4, 11])
def test_savgol_joint_filter_rejects_invalid_window(window: int) -> None:
    with pytest.raises(ValueError, match="window"):
        filter_joint_trajectory_savgol(
            np.zeros((9, 1)),
            np.zeros((9, 1)),
            np.zeros((9, 1)),
            sample_dt_s=0.1,
            window_length_samples=window,
            polynomial_order=2,
        )
