from __future__ import annotations

import numpy as np
import pytest

from python_filter_smoothing.ruckig_interpolation import interpolate_joint_path


def test_interpolation_reaches_every_state_and_respects_limits() -> None:
    q = np.asarray([[0.0, 0.0], [0.08, -0.04], [0.12, -0.02]])
    dq = np.zeros_like(q)
    ddq = np.zeros_like(q)
    dt = 0.005
    limits = {
        "max_velocity": np.asarray([1.0, 1.0]),
        "max_acceleration": np.asarray([4.0, 4.0]),
        "max_jerk": np.asarray([40.0, 40.0]),
    }

    result = interpolate_joint_path(
        q,
        dq,
        ddq,
        sample_dt_s=dt,
        minimum_segment_duration_s=0.02,
        min_position=np.asarray([-1.0, -1.0]),
        max_position=np.asarray([1.0, 1.0]),
        **limits,
    )

    assert np.allclose(result.position[result.node_sample_indices], q)
    assert np.allclose(result.velocity[result.node_sample_indices], dq)
    assert np.allclose(result.acceleration[result.node_sample_indices], ddq)
    assert result.position.shape == result.velocity.shape == result.acceleration.shape
    assert result.position.shape[0] == result.node_sample_indices[-1] + 1
    assert np.allclose(
        result.segment_durations_s / dt,
        np.round(result.segment_durations_s / dt),
    )
    assert np.all(
        np.max(np.abs(result.velocity), axis=0)
        <= limits["max_velocity"] + 1.0e-9
    )
    assert np.all(
        np.max(np.abs(result.acceleration), axis=0)
        <= limits["max_acceleration"] + 1.0e-9
    )
    discrete_jerk = np.diff(result.acceleration, axis=0) / dt
    assert np.all(
        np.max(np.abs(discrete_jerk), axis=0) <= limits["max_jerk"] + 1.0e-7
    )


def test_nonzero_node_derivatives_remain_continuous() -> None:
    q = np.asarray([[0.0], [0.02], [0.04]])
    dq = np.asarray([[0.10], [0.10], [0.00]])
    ddq = np.asarray([[0.0], [0.0], [0.0]])

    result = interpolate_joint_path(
        q,
        dq,
        ddq,
        sample_dt_s=0.005,
        minimum_segment_duration_s=0.2,
        max_velocity=np.asarray([1.0]),
        max_acceleration=np.asarray([2.0]),
        max_jerk=np.asarray([20.0]),
    )

    assert np.allclose(result.position[result.node_sample_indices], q)
    assert np.allclose(result.velocity[result.node_sample_indices], dq)
    assert np.allclose(result.acceleration[result.node_sample_indices], ddq)


def test_rejects_mismatched_node_shapes() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        interpolate_joint_path(
            np.zeros((2, 2)),
            np.zeros((3, 2)),
            np.zeros((2, 2)),
            sample_dt_s=0.005,
            minimum_segment_duration_s=0.02,
            max_velocity=np.ones(2),
            max_acceleration=np.ones(2),
            max_jerk=np.ones(2),
        )
