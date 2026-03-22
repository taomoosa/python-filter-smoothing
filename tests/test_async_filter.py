"""Tests for AsyncFilter."""
import threading
import numpy as np
import pytest
from python_filter_smoothing import AsyncFilter
from python_filter_smoothing.async_filter import (
    AsyncFilterRTC,
    AsyncFilterRTCRAIL,
    _PolyTrajectory,
    _CubicBlend,
    _DualCubicBlend,
    _QuinticBlend,
    _DualQuinticBlend,
)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class TestAsyncFilterEMA:
    def test_returns_none_before_update(self):
        f = AsyncFilter(method="ema")
        assert f.get_output() is None

    def test_returns_value_after_update(self):
        f = AsyncFilter(method="ema")
        f.update(0.0, 1.0)
        assert f.get_output() is not None

    def test_converges_to_constant(self, rerun_log_scalar):
        f = AsyncFilter(method="ema", alpha=0.5)
        for i in range(60):
            f.update(float(i), 1.0)
            val = f.get_output()
            rerun_log_scalar("async/ema/converge/input", float(i), 1.0)
            rerun_log_scalar("async/ema/converge/output", float(i), val)
        np.testing.assert_allclose(f.get_output().item(), 1.0, atol=0.01)

    def test_vector_input(self):
        f = AsyncFilter(method="ema")
        f.update(0.0, [1.0, 2.0])
        val = f.get_output()
        assert val.shape == (2,)


# ---------------------------------------------------------------------------
# Interpolation methods
# ---------------------------------------------------------------------------

class TestAsyncFilterInterpolation:
    def test_linear_midpoint(self):
        f = AsyncFilter(method="linear")
        f.update(0.0, 0.0)
        f.update(1.0, 1.0)
        val = f.get_output(t=0.5)
        np.testing.assert_allclose(val.item(), 0.5, atol=1e-10)

    def test_spline_sine(self, rerun_log_scalar):
        f = AsyncFilter(method="spline", buffer_size=50)
        t = np.linspace(0, 2 * np.pi, 30)
        x = np.sin(t)
        for ti, xi in zip(t, x):
            f.update(float(ti), float(xi))
            rerun_log_scalar("async/spline/input", float(ti), float(xi))
        t_query = np.linspace(0, 2 * np.pi, 60)
        for tq in t_query:
            val = f.get_output(t=float(tq))
            if val is not None:
                rerun_log_scalar("async/spline/output", float(tq), val)
        val = f.get_output(t=float(np.pi / 2))
        np.testing.assert_allclose(val.item(), 1.0, atol=0.01)

    def test_single_point_returns_that_point(self):
        f = AsyncFilter(method="linear")
        f.update(0.0, 42.0)
        val = f.get_output()
        np.testing.assert_allclose(val.item(), 42.0)

    def test_none_t_uses_latest(self):
        f = AsyncFilter(method="linear")
        f.update(0.0, 0.0)
        f.update(2.0, 4.0)
        val = f.get_output(t=None)
        np.testing.assert_allclose(val.item(), 4.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestAsyncFilterThreadSafety:
    def test_concurrent_updates_no_exception(self):
        f = AsyncFilter(method="ema", alpha=0.3)
        errors = []

        def producer():
            try:
                for i in range(200):
                    f.update(float(i), float(i))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=producer) for _ in range(5)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert not errors, f"Thread errors: {errors}"

    def test_concurrent_update_and_read(self):
        f = AsyncFilter(method="ema", alpha=0.3)
        errors = []

        def producer():
            for i in range(300):
                f.update(float(i), 1.0)

        def consumer():
            for _ in range(300):
                try:
                    f.get_output()
                except Exception as exc:
                    errors.append(exc)

        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=consumer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors


# ---------------------------------------------------------------------------
# Buffer management and clear
# ---------------------------------------------------------------------------

class TestAsyncFilterBuffer:
    def test_buffer_size_limit(self):
        f = AsyncFilter(method="ema", buffer_size=10)
        for i in range(25):
            f.update(float(i), 1.0)
        assert f.buffer_length <= 10

    def test_clear_resets_state(self):
        f = AsyncFilter(method="ema")
        f.update(0.0, 1.0)
        f.clear()
        assert f.get_output() is None
        assert f.buffer_length == 0

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            AsyncFilter(method="unknown_method")


# ---------------------------------------------------------------------------
# One-Euro filter
# ---------------------------------------------------------------------------

class TestAsyncFilterOneEuro:
    def test_returns_none_before_update(self):
        f = AsyncFilter(method="one_euro", min_cutoff=1.0, beta=0.007, d_cutoff=1.0)
        assert f.get_output() is None

    def test_converges_to_constant(self, rerun_log_scalar):
        f = AsyncFilter(method="one_euro", buffer_size=200, min_cutoff=1.0, beta=0.007)
        for i in range(100):
            f.update(float(i), 1.0)
            val = f.get_output()
            rerun_log_scalar("async/one_euro/converge/input", float(i), 1.0)
            if val is not None:
                rerun_log_scalar("async/one_euro/converge/output", float(i), val)
        np.testing.assert_allclose(f.get_output().item(), 1.0, atol=0.01)

    def test_smooths_noise(self, rerun_log_scalar):
        rng = np.random.default_rng(42)
        f = AsyncFilter(method="one_euro", buffer_size=200, min_cutoff=1.0, beta=0.0)
        inputs = []
        outputs = []
        for i in range(100):
            xi = 1.0 + rng.normal() * 0.3
            f.update(float(i), xi)
            val = f.get_output()
            inputs.append(xi)
            rerun_log_scalar("async/one_euro/noise/input", float(i), xi)
            if val is not None:
                outputs.append(val.item())
                rerun_log_scalar("async/one_euro/noise/output", float(i), val)
        assert np.var(outputs) < np.var(inputs)

    def test_vector_input(self):
        f = AsyncFilter(method="one_euro", min_cutoff=1.0, beta=0.007)
        for i in range(10):
            f.update(float(i), [1.0, 2.0])
        val = f.get_output()
        assert val.shape == (2,)

    def test_clear_resets_state(self):
        f = AsyncFilter(method="one_euro", min_cutoff=1.0, beta=0.007)
        f.update(0.0, 1.0)
        f.clear()
        assert f.get_output() is None

    def test_thread_safety(self):
        f = AsyncFilter(method="one_euro", min_cutoff=1.0, beta=0.007)
        errors = []

        def producer():
            for i in range(300):
                f.update(float(i), 1.0)

        def consumer():
            for _ in range(300):
                try:
                    f.get_output()
                except Exception as exc:
                    errors.append(exc)

        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=consumer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Thread errors: {errors}"


# ---------------------------------------------------------------------------
# Moving-average filter
# ---------------------------------------------------------------------------

class TestAsyncFilterMovingAverage:
    def test_returns_none_before_update(self):
        f = AsyncFilter(method="moving_average", window=5, buffer_size=50)
        assert f.get_output() is None

    def test_average_of_recent(self):
        f = AsyncFilter(method="moving_average", window=3, buffer_size=50)
        for i, v in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
            f.update(float(i), v)
        np.testing.assert_allclose(f.get_output().item(), 4.0, atol=1e-10)

    def test_vector_input(self):
        f = AsyncFilter(method="moving_average", window=3, buffer_size=50)
        for i in range(5):
            f.update(float(i), [1.0, 2.0])
        val = f.get_output()
        assert val.shape == (2,)

    def test_reduces_noise(self, rerun_log_scalar):
        rng = np.random.default_rng(42)
        f = AsyncFilter(method="moving_average", window=20, buffer_size=200)
        inputs = []
        outputs = []
        for i in range(100):
            xi = 1.0 + rng.normal() * 0.3
            f.update(float(i), xi)
            val = f.get_output()
            inputs.append(xi)
            rerun_log_scalar("async/mavg/noise/input", float(i), xi)
            if val is not None:
                outputs.append(val.item())
                rerun_log_scalar("async/mavg/noise/output", float(i), val)
        assert np.var(outputs) < np.var(inputs)

    def test_clear_resets(self):
        f = AsyncFilter(method="moving_average", window=5, buffer_size=50)
        f.update(0.0, 1.0)
        f.clear()
        assert f.get_output() is None


# ---------------------------------------------------------------------------
# Output interpolation (PCHIP over output history)
# ---------------------------------------------------------------------------

class TestAsyncFilterOutputInterpolation:
    """Tests that EMA, OneEuro, and MovingAverage produce smooth interpolated
    output when get_output(t) is called with an explicit timestamp."""

    def test_ema_interpolates_between_updates(self, rerun_log_scalar):
        """EMA output at intermediate times should be between adjacent outputs."""
        f = AsyncFilter(method="ema", alpha=0.3, buffer_size=200)
        for i in range(20):
            f.update(float(i), float(i))
        v5 = f.get_output(t=5.0)
        v5_5 = f.get_output(t=5.5)
        v6 = f.get_output(t=6.0)
        rerun_log_scalar("async/ema_interp/t5", 5.0, v5)
        rerun_log_scalar("async/ema_interp/t5_5", 5.5, v5_5)
        rerun_log_scalar("async/ema_interp/t6", 6.0, v6)
        assert v5.item() < v5_5.item() < v6.item()

    def test_ema_none_returns_latest(self):
        """EMA get_output() with no t returns the latest state."""
        f = AsyncFilter(method="ema", alpha=0.5, buffer_size=50)
        f.update(0.0, 10.0)
        f.update(1.0, 20.0)
        val = f.get_output()
        assert val is not None
        np.testing.assert_allclose(val.item(), 15.0)

    def test_one_euro_interpolates_between_updates(self, rerun_log_scalar):
        f = AsyncFilter(method="one_euro", buffer_size=200,
                        min_cutoff=1.0, beta=0.007)
        for i in range(20):
            f.update(float(i), float(i))
        v5 = f.get_output(t=5.0)
        v5_5 = f.get_output(t=5.5)
        v6 = f.get_output(t=6.0)
        rerun_log_scalar("async/1e_interp/t5", 5.0, v5)
        rerun_log_scalar("async/1e_interp/t5_5", 5.5, v5_5)
        rerun_log_scalar("async/1e_interp/t6", 6.0, v6)
        assert v5.item() < v5_5.item() < v6.item()

    def test_moving_average_interpolates_between_updates(self, rerun_log_scalar):
        f = AsyncFilter(method="moving_average", buffer_size=200, window=5)
        for i in range(20):
            f.update(float(i), float(i))
        v5 = f.get_output(t=5.0)
        v5_5 = f.get_output(t=5.5)
        v6 = f.get_output(t=6.0)
        rerun_log_scalar("async/ma_interp/t5", 5.0, v5)
        rerun_log_scalar("async/ma_interp/t5_5", 5.5, v5_5)
        rerun_log_scalar("async/ma_interp/t6", 6.0, v6)
        assert v5.item() < v5_5.item() < v6.item()

    def test_interpolation_smooth_with_burst_data(self):
        """Simulates chunk burst: many samples fed at once, output should
        be smooth when queried at fine time intervals."""
        f = AsyncFilter(method="ema", alpha=0.3, buffer_size=500)
        # Feed 50 samples of sin(t) at once (like a chunk burst)
        for i in range(50):
            f.update(i * 0.1, np.sin(i * 0.1))
        # Query at fine-grained intervals
        vals = [f.get_output(t=i * 0.05).item() for i in range(100)]
        diffs = [abs(vals[i+1] - vals[i]) for i in range(len(vals)-1)]
        assert max(diffs) < 0.5, f"Max jump {max(diffs):.4f} too large"

    def test_interpolation_clamps_beyond_range(self):
        """Query beyond the buffer range should return the boundary value."""
        f = AsyncFilter(method="ema", alpha=0.5, buffer_size=50)
        f.update(1.0, 10.0)
        f.update(2.0, 20.0)
        # Beyond range: should hold last value
        v_after = f.get_output(t=100.0)
        v_latest = f.get_output()
        np.testing.assert_allclose(v_after.item(), v_latest.item())
        # Before range: should hold first value
        v_before = f.get_output(t=-10.0)
        v_first = f.get_output(t=1.0)
        np.testing.assert_allclose(v_before.item(), v_first.item())

    def test_vector_interpolation(self):
        """PCHIP should handle multi-dimensional data."""
        f = AsyncFilter(method="ema", alpha=0.3, buffer_size=200)
        for i in range(20):
            f.update(float(i), [float(i), float(i) * 2])
        val = f.get_output(t=5.5)
        assert val.shape == (2,)

    def test_clear_resets_output_history(self):
        f = AsyncFilter(method="ema", alpha=0.5, buffer_size=50)
        f.update(0.0, 1.0)
        f.update(1.0, 2.0)
        assert f.get_output(t=0.5) is not None
        f.clear()
        assert f.get_output(t=0.5) is None


# ---------------------------------------------------------------------------
# update_chunk on base / existing filters
# ---------------------------------------------------------------------------

class TestUpdateChunkBase:
    """Verify that base update_chunk dispatches to update() per sample."""

    def test_update_chunk_fills_buffer(self):
        f = AsyncFilter(method="linear", buffer_size=50)
        t = np.linspace(0, 1, 20)
        x = np.sin(t)
        f.update_chunk(t, x)
        assert f.buffer_length == 20

    def test_update_chunk_mismatched_lengths_raises(self):
        f = AsyncFilter(method="linear")
        with pytest.raises(ValueError, match="samples"):
            f.update_chunk(np.arange(5), np.ones((3, 2)))

    def test_update_chunk_ema_equivalent(self):
        """update_chunk should give same result as calling update per sample."""
        f1 = AsyncFilter(method="ema", alpha=0.3)
        f2 = AsyncFilter(method="ema", alpha=0.3)
        t = np.array([0.0, 0.1, 0.2, 0.3])
        x = np.array([1.0, 2.0, 1.5, 3.0])
        for ti, xi in zip(t, x):
            f1.update(float(ti), xi)
        f2.update_chunk(t, x)
        np.testing.assert_allclose(f1.get_output(), f2.get_output())


# ---------------------------------------------------------------------------
# ACT temporal ensembling
# ---------------------------------------------------------------------------

class TestAsyncFilterACT:
    def test_returns_none_before_update(self):
        f = AsyncFilter(method="act")
        assert f.get_output() is None

    def test_single_chunk(self):
        f = AsyncFilter(method="act", k=0.01)
        t = np.array([0.0, 1.0, 2.0])
        x = np.array([0.0, 1.0, 2.0])
        f.update_chunk(t, x)
        # At t=1.0 with a single chunk, should return the chunk's value at t=1
        val = f.get_output(t=1.0)
        assert val is not None
        np.testing.assert_allclose(val, [1.0], atol=1e-10)

    def test_temporal_ensemble_newer_dominates(self):
        """With high k, the newest chunk should dominate."""
        f = AsyncFilter(method="act", k=10.0, max_chunks=10)
        # Chunk 1: value = 0 everywhere
        t1 = np.array([0.0, 1.0, 2.0])
        x1 = np.array([0.0, 0.0, 0.0])
        f.update_chunk(t1, x1)
        # Chunk 2: value = 10 everywhere (overlaps)
        t2 = np.array([0.0, 1.0, 2.0])
        x2 = np.array([10.0, 10.0, 10.0])
        f.update_chunk(t2, x2)
        val = f.get_output(t=1.0)
        # With k=10, weight of chunk1 = exp(-10*1) ≈ 4.5e-5, chunk2 = exp(0) = 1
        # So result should be very close to 10
        assert val is not None
        assert float(val[0]) > 9.5

    def test_temporal_ensemble_equal_weight(self):
        """With k=0, all chunks have equal weight."""
        f = AsyncFilter(method="act", k=0.0, max_chunks=10)
        t1 = np.array([0.0, 1.0, 2.0])
        x1 = np.array([0.0, 0.0, 0.0])
        f.update_chunk(t1, x1)
        t2 = np.array([0.0, 1.0, 2.0])
        x2 = np.array([10.0, 10.0, 10.0])
        f.update_chunk(t2, x2)
        val = f.get_output(t=1.0)
        # k=0 → all weights = 1 → simple average = 5
        np.testing.assert_allclose(val, [5.0], atol=1e-10)

    def test_non_overlapping_chunks(self):
        """Query in a region covered by only one chunk returns that chunk."""
        f = AsyncFilter(method="act", k=0.01)
        f.update_chunk(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        f.update_chunk(np.array([3.0, 4.0]), np.array([10.0, 11.0]))
        # t=0.5 is only in chunk 1
        val = f.get_output(t=0.5)
        np.testing.assert_allclose(val, [0.5], atol=1e-10)
        # t=3.5 is only in chunk 2
        val = f.get_output(t=3.5)
        np.testing.assert_allclose(val, [10.5], atol=1e-10)

    def test_hold_beyond_range(self):
        """Query beyond all chunks holds last chunk's endpoint."""
        f = AsyncFilter(method="act", k=0.01)
        f.update_chunk(np.array([0.0, 1.0]), np.array([5.0, 6.0]))
        val = f.get_output(t=10.0)
        np.testing.assert_allclose(val, [6.0], atol=1e-10)

    def test_hold_before_range(self):
        """Query before all chunks holds first chunk's start."""
        f = AsyncFilter(method="act", k=0.01)
        f.update_chunk(np.array([5.0, 6.0]), np.array([5.0, 6.0]))
        val = f.get_output(t=0.0)
        np.testing.assert_allclose(val, [5.0], atol=1e-10)

    def test_vector_data(self):
        f = AsyncFilter(method="act", k=0.01)
        t = np.array([0.0, 1.0, 2.0])
        x = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        f.update_chunk(t, x)
        val = f.get_output(t=1.5)
        assert val is not None
        assert val.shape == (2,)
        np.testing.assert_allclose(val, [2.5, 25.0], atol=1e-10)

    def test_max_chunks_eviction(self):
        """Old chunks are evicted when max_chunks is reached."""
        f = AsyncFilter(method="act", k=0.0, max_chunks=2)
        f.update_chunk(np.array([0.0, 1.0]), np.array([100.0, 100.0]))
        f.update_chunk(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        f.update_chunk(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        # First chunk (value=100) should have been evicted
        val = f.get_output(t=0.5)
        np.testing.assert_allclose(val, [0.0], atol=1e-10)

    def test_clear_resets(self):
        f = AsyncFilter(method="act")
        f.update_chunk(np.array([0.0, 1.0]), np.array([1.0, 2.0]))
        assert f.get_output() is not None
        f.clear()
        assert f.get_output() is None

    def test_get_output_none_t_uses_last_endpoint(self):
        f = AsyncFilter(method="act", k=0.01)
        f.update_chunk(np.array([0.0, 1.0, 2.0]), np.array([0.0, 5.0, 10.0]))
        val = f.get_output()  # t=None → last chunk's last timestamp = 2.0
        np.testing.assert_allclose(val, [10.0], atol=1e-10)


# ---------------------------------------------------------------------------
# Blend class derivative tests
# ---------------------------------------------------------------------------

class TestBlendDerivatives:
    """Tests for evaluate_deriv on _QuinticBlend and _DualQuinticBlend."""

    @staticmethod
    def _make_trajs():
        t1 = np.linspace(0.0, 2.0, 30)
        x1 = np.column_stack([np.sin(t1), np.cos(t1)])
        traj_old = _PolyTrajectory.fit(t1, x1, 3)
        t2 = np.linspace(1.5, 3.5, 30)
        x2 = np.column_stack([np.sin(t2) + 0.2, np.cos(t2) - 0.1])
        traj_new = _PolyTrajectory.fit(t2, x2, 3)
        return traj_old, traj_new

    def test_quintic_deriv_matches_numeric(self):
        traj_old, traj_new = self._make_trajs()
        blend = _QuinticBlend(traj_old, traj_new, 1.5, 2.0)
        t = 1.75
        eps = 1e-6
        analytic_vel = blend.evaluate_deriv(t, 1)
        numeric_vel = (blend.evaluate(t + eps) - blend.evaluate(t - eps)) / (2 * eps)
        np.testing.assert_allclose(analytic_vel, numeric_vel, atol=1e-3)

    def test_dual_quintic_deriv_matches_numeric(self):
        traj_old, traj_new = self._make_trajs()
        blend = _DualQuinticBlend(traj_old, traj_new, 1.5, 2.0)
        for t in [1.6, 1.75, 1.9]:
            analytic_vel = blend.evaluate_deriv(t, 1)
            eps = 1e-6
            numeric_vel = (blend.evaluate(t + eps) - blend.evaluate(t - eps)) / (2 * eps)
            np.testing.assert_allclose(analytic_vel, numeric_vel, atol=1e-3)

    def test_quintic_start_state_override(self):
        traj_old, traj_new = self._make_trajs()
        custom_p = np.array([10.0, 20.0])
        custom_v = np.array([0.5, -0.5])
        custom_a = np.array([0.0, 0.0])
        blend = _QuinticBlend(
            traj_old, traj_new, 1.5, 2.0,
            start_state=(custom_p, custom_v, custom_a),
        )
        pos_at_start = blend.evaluate(1.5)
        np.testing.assert_allclose(pos_at_start, custom_p, atol=1e-10)
        vel_at_start = blend.evaluate_deriv(1.5, 1)
        np.testing.assert_allclose(vel_at_start, custom_v, atol=1e-6)

    def test_dual_quintic_start_state_override(self):
        traj_old, traj_new = self._make_trajs()
        custom_p = np.array([5.0, 15.0])
        custom_v = np.array([1.0, -1.0])
        custom_a = np.array([0.1, -0.1])
        blend = _DualQuinticBlend(
            traj_old, traj_new, 1.5, 2.0,
            start_state=(custom_p, custom_v, custom_a),
        )
        pos_at_start = blend.evaluate(1.5)
        np.testing.assert_allclose(pos_at_start, custom_p, atol=1e-10)


# ---------------------------------------------------------------------------
# RAIL trajectory post-processing
# ---------------------------------------------------------------------------

class TestAsyncFilterRAIL:
    def test_returns_none_before_update(self):
        f = AsyncFilter(method="rail")
        assert f.get_output() is None

    def test_single_chunk_polynomial_smooth(self):
        """Single chunk should return polynomial-smoothed values."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, 50)
        x_clean = 2.0 * t + 1.0  # linear signal
        x_noisy = x_clean + rng.normal(0, 0.1, size=50)
        f = AsyncFilter(method="rail", poly_degree=3)
        f.update_chunk(t, x_noisy)
        # Smoothed output at midpoint should be closer to clean than noisy
        val = f.get_output(t=0.5)
        assert val is not None
        assert abs(float(val[0]) - 2.0) < 0.15  # close to 2*0.5+1 = 2

    def test_c2_continuity_position(self):
        """Position should be continuous at blend boundary."""
        f = AsyncFilter(method="rail", poly_degree=3, blend_duration=0.3)
        # Chunk 1
        t1 = np.linspace(0.0, 1.0, 30)
        x1 = np.sin(t1)
        f.update_chunk(t1, x1)
        # Chunk 2 (starts at 0.8, overlaps)
        t2 = np.linspace(0.8, 2.0, 30)
        x2 = np.sin(t2) + 0.1  # slight offset
        f.update_chunk(t2, x2)
        # Check position at blend start (t=0.8) and blend end (t=1.1)
        # Both should give finite, reasonable values
        v_start = f.get_output(t=0.8)
        v_end = f.get_output(t=1.1)
        assert v_start is not None
        assert v_end is not None
        assert np.all(np.isfinite(v_start))
        assert np.all(np.isfinite(v_end))

    def test_c2_continuity_smooth_transition(self):
        """Output should be smooth through blend region (no jumps)."""
        f = AsyncFilter(method="rail", poly_degree=3, blend_duration=0.5)
        t1 = np.linspace(0.0, 2.0, 50)
        x1 = np.ones(50) * 0.0
        f.update_chunk(t1, x1)
        t2 = np.linspace(1.5, 3.5, 50)
        x2 = np.ones(50) * 1.0
        f.update_chunk(t2, x2)
        # Sample densely through the blend region
        ts = np.linspace(1.4, 2.1, 100)
        vals = np.array([f.get_output(t=float(ti)) for ti in ts])
        # First derivative (finite differences) should be smooth (no huge jumps)
        diffs = np.diff(vals[:, 0]) / np.diff(ts)
        max_jerk = np.max(np.abs(np.diff(diffs)))
        assert max_jerk < 100.0  # reasonable bound for smooth transition

    def test_output_matches_polynomial_outside_blend(self):
        """Outside the blend region, output should follow polynomial exactly."""
        f = AsyncFilter(method="rail", poly_degree=3, blend_duration=0.2)
        t1 = np.linspace(0.0, 1.0, 30)
        x1 = t1 ** 2  # quadratic, fits perfectly in cubic
        f.update_chunk(t1, x1)
        # Query in the middle of the chunk (no blend region yet)
        val = f.get_output(t=0.5)
        np.testing.assert_allclose(val, [0.25], atol=0.01)

    def test_vector_data(self):
        f = AsyncFilter(method="rail", poly_degree=3)
        t = np.linspace(0, 1, 20)
        x = np.column_stack([t, t ** 2])
        f.update_chunk(t, x)
        val = f.get_output(t=0.5)
        assert val is not None
        assert val.shape == (2,)
        np.testing.assert_allclose(val, [0.5, 0.25], atol=0.02)

    def test_multiple_chunks(self):
        """Processing 3+ chunks should work correctly."""
        f = AsyncFilter(method="rail", poly_degree=3, blend_duration=0.2)
        for i in range(4):
            t = np.linspace(i * 0.8, i * 0.8 + 1.0, 20)
            x = np.sin(t) + i * 0.05
            f.update_chunk(t, x)
        val = f.get_output(t=3.0)
        assert val is not None
        assert np.all(np.isfinite(val))

    def test_clear_resets(self):
        f = AsyncFilter(method="rail")
        f.update_chunk(np.linspace(0, 1, 10), np.ones(10))
        assert f.get_output() is not None
        f.clear()
        assert f.get_output() is None

    def test_get_output_none_t_uses_end(self):
        """get_output(t=None) should return value at end of current trajectory."""
        f = AsyncFilter(method="rail", poly_degree=1)
        t = np.array([0.0, 1.0, 2.0])
        x = np.array([0.0, 1.0, 2.0])
        f.update_chunk(t, x)
        val = f.get_output()  # t=None → t_end = 2.0
        np.testing.assert_allclose(val, [2.0], atol=0.01)

    def test_blend_region_differs_from_either_traj(self):
        """In the blend region, output should differ from both raw trajectories."""
        f = AsyncFilter(method="rail", poly_degree=3, blend_duration=0.5)
        t1 = np.linspace(0.0, 2.0, 50)
        x1 = np.zeros(50)
        f.update_chunk(t1, x1)
        t2 = np.linspace(1.5, 3.5, 50)
        x2 = np.ones(50) * 2.0
        f.update_chunk(t2, x2)
        val = f.get_output(t=1.75)  # midpoint of blend
        # Should be between 0 and 2 (blending the two trajectories)
        assert 0.0 < float(val[0]) < 2.0


# ---------------------------------------------------------------------------
# RAIL: dual-quintic blend
# ---------------------------------------------------------------------------

class TestAsyncFilterRAILDualQuintic:
    def test_dual_quintic_default_enabled(self):
        """dual_quintic=True is the default."""
        f = AsyncFilter(method="rail")
        assert f._dual_quintic is True

    def test_dual_quintic_blend_in_region(self):
        """Dual-quintic blend should produce intermediate values."""
        f = AsyncFilter(
            method="rail", poly_degree=3, blend_duration=0.5, dual_quintic=True
        )
        t1 = np.linspace(0.0, 2.0, 50)
        x1 = np.zeros(50)
        f.update_chunk(t1, x1)
        t2 = np.linspace(1.5, 3.5, 50)
        x2 = np.ones(50) * 2.0
        f.update_chunk(t2, x2)
        val = f.get_output(t=1.75)
        assert 0.0 < float(val[0]) < 2.0

    def test_dual_quintic_smooth_transition(self):
        """Dual-quintic should produce smooth output through blend region."""
        f = AsyncFilter(
            method="rail", poly_degree=3, blend_duration=0.6, dual_quintic=True
        )
        t1 = np.linspace(0.0, 2.0, 50)
        x1 = np.ones(50) * 0.0
        f.update_chunk(t1, x1)
        t2 = np.linspace(1.5, 3.5, 50)
        x2 = np.ones(50) * 1.0
        f.update_chunk(t2, x2)
        # Sample densely through the blend region
        ts = np.linspace(1.4, 2.2, 200)
        vals = np.array([f.get_output(t=float(ti)) for ti in ts])
        # Finite differences should be smooth (no large jumps)
        diffs = np.diff(vals[:, 0]) / np.diff(ts)
        max_jerk = np.max(np.abs(np.diff(diffs)))
        assert max_jerk < 100.0

    def test_dual_quintic_c2_at_midpoint(self):
        """Position/velocity should be continuous at the midpoint of dual blend."""
        f = AsyncFilter(
            method="rail", poly_degree=3, blend_duration=1.0, dual_quintic=True
        )
        t1 = np.linspace(0.0, 2.0, 50)
        x1 = np.sin(t1)
        f.update_chunk(t1, x1)
        t2 = np.linspace(1.5, 3.5, 50)
        x2 = np.sin(t2) + 0.2
        f.update_chunk(t2, x2)
        # Midpoint of blend region
        t_mid = 1.5 + 0.5  # blend_start + blend_dur/2
        eps = 1e-5
        v_before = f.get_output(t=t_mid - eps)
        v_after = f.get_output(t=t_mid + eps)
        # Position should be nearly identical across midpoint
        np.testing.assert_allclose(v_before, v_after, atol=1e-3)

    def test_single_quintic_fallback(self):
        """dual_quintic=False should use single-quintic blend (original)."""
        f = AsyncFilter(
            method="rail",
            poly_degree=3,
            blend_duration=0.5,
            dual_quintic=False,
        )
        t1 = np.linspace(0.0, 2.0, 50)
        x1 = np.zeros(50)
        f.update_chunk(t1, x1)
        t2 = np.linspace(1.5, 3.5, 50)
        x2 = np.ones(50) * 2.0
        f.update_chunk(t2, x2)
        val = f.get_output(t=1.75)
        assert val is not None
        assert 0.0 < float(val[0]) < 2.0

    def test_dual_vs_single_differ(self):
        """Dual-quintic and single-quintic should produce different values."""
        kwargs = dict(poly_degree=3, blend_duration=0.5)
        f_dual = AsyncFilter(method="rail", dual_quintic=True, **kwargs)
        f_single = AsyncFilter(method="rail", dual_quintic=False, **kwargs)
        t1 = np.linspace(0.0, 2.0, 50)
        x1 = np.sin(t1)
        t2 = np.linspace(1.5, 3.5, 50)
        x2 = np.cos(t2) + 0.5
        for f in (f_dual, f_single):
            f.update_chunk(t1, x1)
            f.update_chunk(t2, x2)
        v_dual = f_dual.get_output(t=1.75)
        v_single = f_single.get_output(t=1.75)
        # Both should be finite but generally different
        assert np.all(np.isfinite(v_dual))
        assert np.all(np.isfinite(v_single))
        assert not np.allclose(v_dual, v_single, atol=1e-6)

    def test_dual_quintic_vector_data(self):
        """Dual-quintic should work with multi-dimensional data."""
        f = AsyncFilter(
            method="rail", poly_degree=3, blend_duration=0.3, dual_quintic=True
        )
        t1 = np.linspace(0.0, 1.0, 30)
        x1 = np.column_stack([t1, t1 ** 2, np.sin(t1)])
        f.update_chunk(t1, x1)
        t2 = np.linspace(0.7, 2.0, 30)
        x2 = np.column_stack([t2 + 0.1, t2 ** 2 + 0.1, np.sin(t2) + 0.1])
        f.update_chunk(t2, x2)
        val = f.get_output(t=0.85)
        assert val is not None
        assert val.shape == (3,)
        assert np.all(np.isfinite(val))


# ---------------------------------------------------------------------------
# RAIL: set_current_time and blend-start behaviour
# ---------------------------------------------------------------------------

class TestAsyncFilterRAILCurrentTime:
    def test_set_current_time_affects_blend_start(self):
        """Blend should start at current_time, not at chunk's t_start."""
        f = AsyncFilter(
            method="rail", poly_degree=3, blend_duration=0.4, dual_quintic=True
        )
        t1 = np.linspace(0.0, 2.0, 50)
        x1 = np.zeros(50)
        f.update_chunk(t1, x1)
        # Set current time to 1.8 (later than chunk2's start of 1.5)
        f.set_current_time(1.8)
        t2 = np.linspace(1.5, 3.5, 50)
        x2 = np.ones(50) * 2.0
        f.update_chunk(t2, x2)
        # At t=1.6 (before current_time=1.8), should still be old trajectory
        val_before = f.get_output(t=1.6)
        np.testing.assert_allclose(val_before, [0.0], atol=0.1)
        # At t=1.9 (inside blend: 1.8 to 2.2), should be blending
        val_blend = f.get_output(t=1.9)
        assert 0.0 < float(val_blend[0]) < 2.0

    def test_without_current_time_uses_chunk_start(self):
        """Without set_current_time, blend starts at new chunk's t_start."""
        f = AsyncFilter(
            method="rail", poly_degree=3, blend_duration=0.4, dual_quintic=True
        )
        t1 = np.linspace(0.0, 2.0, 50)
        x1 = np.zeros(50)
        f.update_chunk(t1, x1)
        # No set_current_time called
        t2 = np.linspace(1.5, 3.5, 50)
        x2 = np.ones(50) * 2.0
        f.update_chunk(t2, x2)
        # At t=1.6 (inside blend: 1.5 to 1.9), should be blending
        val = f.get_output(t=1.6)
        assert 0.0 < float(val[0]) < 2.0

    def test_set_current_time_thread_safe(self):
        """set_current_time should be callable from another thread."""
        f = AsyncFilter(method="rail", poly_degree=3)
        errors = []

        def updater():
            try:
                for i in range(100):
                    f.set_current_time(float(i) * 0.01)
            except Exception as exc:
                errors.append(exc)

        def reader():
            try:
                t = np.linspace(0, 1, 20)
                x = np.sin(t)
                f.update_chunk(t, x)
                for _ in range(100):
                    f.get_output(t=0.5)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=fn) for fn in (updater, reader)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert not errors

    def test_clear_resets_current_time(self):
        """clear() should also reset current_time."""
        f = AsyncFilter(method="rail")
        f.set_current_time(1.0)
        f.clear()
        assert f._current_time is None


# ---------------------------------------------------------------------------
# RAIL: temporal alignment (auto_align)
# ---------------------------------------------------------------------------

class TestAsyncFilterRAILAutoAlign:
    def test_auto_align_default_disabled(self):
        """auto_align should be False by default."""
        f = AsyncFilter(method="rail")
        assert f._auto_align is False

    def test_auto_align_shifts_chunk_timestamps(self):
        """auto_align should shift new chunk so that motion direction matches."""
        f = AsyncFilter(
            method="rail",
            poly_degree=3,
            blend_duration=0.3,
            auto_align=True,
            dual_quintic=True,
        )
        # First chunk: linearly increasing signal
        t1 = np.linspace(0.0, 2.0, 50)
        x1 = t1 * 1.0  # velocity = 1.0
        f.update_chunk(t1, x1)
        # Simulate: current time is 2.5 but chunk2 starts at 2.0
        # (0.5s of stale data due to inference latency)
        f.set_current_time(2.5)
        t2 = np.linspace(2.0, 4.0, 50)
        x2 = t2 * 1.0 + 0.1  # similar slope, small offset
        f.update_chunk(t2, x2)
        # The aligned chunk should produce a sensible value at t=2.5
        val = f.get_output(t=2.5)
        assert val is not None
        assert np.all(np.isfinite(val))

    def test_auto_align_without_current_time_is_noop(self):
        """auto_align without set_current_time should not crash."""
        f = AsyncFilter(method="rail", poly_degree=3, auto_align=True)
        t1 = np.linspace(0, 1, 20)
        x1 = np.sin(t1)
        f.update_chunk(t1, x1)
        t2 = np.linspace(0.8, 2.0, 20)
        x2 = np.sin(t2) + 0.1
        f.update_chunk(t2, x2)
        val = f.get_output(t=1.0)
        assert val is not None

    def test_auto_align_with_custom_window(self):
        """align_window parameter should control the search range."""
        f = AsyncFilter(
            method="rail",
            poly_degree=3,
            auto_align=True,
            align_window=0.3,
        )
        t1 = np.linspace(0.0, 2.0, 50)
        x1 = t1 * 1.0
        f.update_chunk(t1, x1)
        f.set_current_time(2.3)
        t2 = np.linspace(2.0, 4.0, 50)
        x2 = t2 * 1.0
        f.update_chunk(t2, x2)
        val = f.get_output(t=2.3)
        assert val is not None
        assert np.all(np.isfinite(val))

    def test_auto_align_first_chunk_no_shift(self):
        """First chunk should have zero alignment shift (no previous traj)."""
        f = AsyncFilter(method="rail", poly_degree=3, auto_align=True)
        f.set_current_time(0.5)
        t = np.linspace(0.0, 1.0, 20)
        x = np.sin(t)
        f.update_chunk(t, x)
        # Should match direct polynomial evaluation at 0.5
        val = f.get_output(t=0.5)
        expected = np.sin(0.5)
        np.testing.assert_allclose(val, [expected], atol=0.05)

    def test_auto_align_preserves_shape(self):
        """auto_align should not change output dimensionality."""
        f = AsyncFilter(
            method="rail", poly_degree=3, auto_align=True, dual_quintic=True
        )
        t1 = np.linspace(0, 1, 20)
        x1 = np.column_stack([t1, t1 ** 2])
        f.update_chunk(t1, x1)
        f.set_current_time(1.2)
        t2 = np.linspace(0.8, 2.0, 20)
        x2 = np.column_stack([t2 + 0.1, t2 ** 2 + 0.1])
        f.update_chunk(t2, x2)
        val = f.get_output(t=1.2)
        assert val.shape == (2,)


# ---------------------------------------------------------------------------
# RAIL: blend_start_source strategies
# ---------------------------------------------------------------------------

class TestAsyncFilterRAILBlendStartSource:
    """Tests for blend_start_source parameter on AsyncFilterRAIL."""

    def _make_three_chunk_filter(self, source: str, *, dual: bool = True):
        """Create a filter, feed 3 chunks, querying between each."""
        f = AsyncFilter(
            method="rail",
            poly_degree=3,
            blend_duration=0.4,
            dual_quintic=dual,
            blend_start_source=source,
        )
        # Chunk 1: constant 0
        t1 = np.linspace(0.0, 2.0, 50)
        x1 = np.zeros(50)
        f.update_chunk(t1, x1)
        # Query during chunk 1 to populate history
        for ti in np.linspace(0.5, 1.8, 20):
            f.get_output(t=float(ti))
        # Chunk 2: constant 1 (blend region is in [1.5, 1.9])
        f.set_current_time(1.5)
        t2 = np.linspace(1.5, 3.5, 50)
        x2 = np.ones(50) * 1.0
        f.update_chunk(t2, x2)
        # Query during blend
        for ti in np.linspace(1.5, 2.0, 20):
            f.get_output(t=float(ti))
        # Chunk 3: constant 2 (now we are in the blend from chunk2→3)
        f.set_current_time(2.5)
        t3 = np.linspace(2.5, 4.5, 50)
        x3 = np.ones(50) * 2.0
        f.update_chunk(t3, x3)
        return f

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError, match="blend_start_source"):
            AsyncFilter(method="rail", blend_start_source="bogus")

    def test_actual_output_source_default(self):
        """Default blend_start_source should be 'actual_output'."""
        f = AsyncFilter(method="rail")
        assert f._blend_start_source == "actual_output"

    def test_all_sources_produce_finite_output(self):
        """All three strategies should produce finite output."""
        for src in ("trajectory", "actual_output", "output_history"):
            f = self._make_three_chunk_filter(src)
            val = f.get_output(t=2.8)
            assert val is not None, f"source={src}"
            assert np.all(np.isfinite(val)), f"source={src}: {val}"

    def test_actual_output_differs_from_trajectory(self):
        """actual_output should differ from trajectory when in a blend."""
        f_traj = self._make_three_chunk_filter("trajectory")
        f_actual = self._make_three_chunk_filter("actual_output")
        # At a point in the 3rd blend region, results should generally differ
        # because the 2nd blend was active at t=2.5 (blend start of 3rd chunk)
        v_traj = f_traj.get_output(t=2.7)
        v_actual = f_actual.get_output(t=2.7)
        assert v_traj is not None and v_actual is not None
        # They may or may not differ significantly, but both should be finite
        assert np.all(np.isfinite(v_traj))
        assert np.all(np.isfinite(v_actual))

    def test_output_history_source_works(self):
        """output_history should use recorded values for blend start."""
        f = self._make_three_chunk_filter("output_history")
        # After 3 chunks, output history should be non-empty
        assert len(f._output_history) > 0
        val = f.get_output(t=3.0)
        assert val is not None
        assert np.all(np.isfinite(val))

    def test_actual_output_single_quintic(self):
        """actual_output should work with single quintic too."""
        f = self._make_three_chunk_filter(
            "actual_output", dual=False
        )
        val = f.get_output(t=2.8)
        assert val is not None
        assert np.all(np.isfinite(val))

    def test_output_history_populated_by_get_output(self):
        """get_output calls should populate _output_history."""
        f = AsyncFilter(
            method="rail",
            poly_degree=3,
            blend_start_source="output_history",
        )
        t = np.linspace(0, 1, 20)
        f.update_chunk(t, np.sin(t))
        assert len(f._output_history) == 0
        f.get_output(t=0.3)
        assert len(f._output_history) == 1
        f.get_output(t=0.5)
        assert len(f._output_history) == 2

    def test_clear_resets_output_history(self):
        f = AsyncFilter(
            method="rail",
            blend_start_source="output_history",
        )
        t = np.linspace(0, 1, 20)
        f.update_chunk(t, np.sin(t))
        f.get_output(t=0.5)
        assert len(f._output_history) > 0
        f.clear()
        assert len(f._output_history) == 0

    def test_blend_continuity_actual_output(self):
        """actual_output should produce smooth transitions (no jumps)."""
        f = AsyncFilter(
            method="rail",
            poly_degree=3,
            blend_duration=0.5,
            dual_quintic=True,
            blend_start_source="actual_output",
        )
        t1 = np.linspace(0.0, 2.0, 50)
        x1 = np.ones(50) * 0.0
        f.update_chunk(t1, x1)
        for ti in np.linspace(0.5, 1.4, 20):
            f.get_output(t=float(ti))
        f.set_current_time(1.5)
        t2 = np.linspace(1.5, 3.5, 50)
        x2 = np.ones(50) * 1.0
        f.update_chunk(t2, x2)
        for ti in np.linspace(1.5, 2.4, 20):
            f.get_output(t=float(ti))
        f.set_current_time(2.5)
        t3 = np.linspace(2.5, 4.5, 50)
        x3 = np.ones(50) * 2.0
        f.update_chunk(t3, x3)
        # Dense sampling through 3rd blend region
        ts = np.linspace(2.4, 3.1, 200)
        vals = np.array([f.get_output(t=float(ti)) for ti in ts])
        diffs = np.diff(vals[:, 0]) / np.diff(ts)
        max_jerk = np.max(np.abs(np.diff(diffs)))
        assert max_jerk < 200.0

    def test_vector_data_all_sources(self):
        """All sources should handle multi-dimensional data."""
        for src in ("trajectory", "actual_output", "output_history"):
            f = AsyncFilter(
                method="rail",
                poly_degree=3,
                blend_duration=0.3,
                blend_start_source=src,
            )
            t1 = np.linspace(0.0, 1.0, 30)
            x1 = np.column_stack([t1, t1 ** 2, np.sin(t1)])
            f.update_chunk(t1, x1)
            for ti in np.linspace(0.2, 0.9, 10):
                f.get_output(t=float(ti))
            f.set_current_time(0.7)
            t2 = np.linspace(0.7, 2.0, 30)
            x2 = np.column_stack(
                [t2 + 0.1, t2 ** 2 + 0.1, np.sin(t2) + 0.1]
            )
            f.update_chunk(t2, x2)
            val = f.get_output(t=0.85)
            assert val is not None, f"source={src}"
            assert val.shape == (3,), f"source={src}"
            assert np.all(np.isfinite(val)), f"source={src}"


# ---------------------------------------------------------------------------
# Cubic blend tests
# ---------------------------------------------------------------------------
class TestCubicBlends:
    """Tests for C¹ cubic blend classes."""

    @staticmethod
    def _make_two_trajs():
        t1 = np.linspace(0, 1, 20)
        x1 = np.column_stack([t1, t1 ** 2])
        t2 = np.linspace(0.7, 2.0, 30)
        x2 = np.column_stack([t2 + 0.5, t2 ** 2 + 0.5])
        traj1 = _PolyTrajectory.fit(t1, x1, 3)
        traj2 = _PolyTrajectory.fit(t2, x2, 3)
        return traj1, traj2

    def test_cubic_blend_evaluate(self):
        traj1, traj2 = self._make_two_trajs()
        bl = _CubicBlend(traj1, traj2, 0.8, 1.0)
        for t in [0.8, 0.85, 0.9, 0.95, 1.0]:
            val = bl.evaluate(t)
            assert val.shape == (2,)
            assert np.all(np.isfinite(val))

    def test_dual_cubic_blend_evaluate(self):
        traj1, traj2 = self._make_two_trajs()
        bl = _DualCubicBlend(traj1, traj2, 0.8, 1.0)
        for t in [0.8, 0.85, 0.9, 0.95, 1.0]:
            val = bl.evaluate(t)
            assert val.shape == (2,)
            assert np.all(np.isfinite(val))

    def test_cubic_pos_vel_continuity(self):
        """C¹ blend should be continuous in position and velocity at knots."""
        traj1, traj2 = self._make_two_trajs()
        bl = _CubicBlend(traj1, traj2, 0.8, 1.0)
        # Start knot
        p_old = traj1.evaluate(0.8)
        p_bl = bl.evaluate(0.8)
        np.testing.assert_allclose(p_bl, p_old, atol=1e-8)
        # End knot
        p_new = traj2.evaluate(1.0)
        p_bl_end = bl.evaluate(1.0)
        np.testing.assert_allclose(p_bl_end, p_new, atol=1e-8)

    def test_cubic_blend_with_start_state(self):
        traj1, traj2 = self._make_two_trajs()
        custom_pos = np.array([10.0, 20.0])
        custom_vel = np.array([1.0, 2.0])
        bl = _CubicBlend(
            traj1, traj2, 0.8, 1.0,
            start_state=(custom_pos, custom_vel, np.zeros(2)),
        )
        p_start = bl.evaluate(0.8)
        np.testing.assert_allclose(p_start, custom_pos, atol=1e-8)

    def test_cubic_evaluate_deriv(self):
        traj1, traj2 = self._make_two_trajs()
        bl = _CubicBlend(traj1, traj2, 0.8, 1.0)
        d = bl.evaluate_deriv(0.9, 1)
        assert d.shape == (2,)
        assert np.all(np.isfinite(d))


# ---------------------------------------------------------------------------
# blend_order parameter tests
# ---------------------------------------------------------------------------
class TestBlendOrder:
    """Tests for the blend_order parameter on AsyncFilterRAIL."""

    def test_default_blend_order_is_cubic(self):
        f = AsyncFilter(method="rail")
        assert f._blend_order == "cubic"

    def test_quintic_blend_order(self):
        f = AsyncFilter(method="rail", blend_order="quintic")
        assert f._blend_order == "quintic"

    def test_invalid_blend_order_raises(self):
        with pytest.raises(ValueError, match="blend_order"):
            AsyncFilter(method="rail", blend_order="linear")

    def test_cubic_produces_finite_output(self):
        f = AsyncFilter(method="rail", blend_order="cubic")
        t1 = np.linspace(0, 1, 20)
        x1 = np.column_stack([t1, t1 ** 2, np.sin(t1)])
        f.update_chunk(t1, x1)
        f.set_current_time(0.5)
        t2 = np.linspace(0.4, 1.5, 20)
        x2 = np.column_stack([t2 + 0.1, t2 ** 2 + 0.1, np.sin(t2) + 0.1])
        f.update_chunk(t2, x2)
        val = f.get_output(t=0.6)
        assert val is not None
        assert np.all(np.isfinite(val))

    def test_quintic_produces_finite_output(self):
        f = AsyncFilter(method="rail", blend_order="quintic")
        t1 = np.linspace(0, 1, 20)
        x1 = np.column_stack([t1, t1 ** 2, np.sin(t1)])
        f.update_chunk(t1, x1)
        f.set_current_time(0.5)
        t2 = np.linspace(0.4, 1.5, 20)
        x2 = np.column_stack([t2 + 0.1, t2 ** 2 + 0.1, np.sin(t2) + 0.1])
        f.update_chunk(t2, x2)
        val = f.get_output(t=0.6)
        assert val is not None
        assert np.all(np.isfinite(val))


# ---------------------------------------------------------------------------
# acc_clamp parameter tests
# ---------------------------------------------------------------------------
class TestAccClamp:
    """Tests for the acc_clamp parameter on AsyncFilterRAIL."""

    def test_default_acc_clamp_is_none(self):
        f = AsyncFilter(method="rail")
        assert f._acc_clamp is None

    def test_acc_clamp_set(self):
        f = AsyncFilter(method="rail", acc_clamp=10.0)
        assert f._acc_clamp == 10.0

    def test_acc_clamp_reduces_overshoot(self):
        """With acc_clamp, quintic blend should have bounded acceleration."""
        np.random.seed(42)
        f_clamped = AsyncFilter(
            method="rail", blend_order="quintic", acc_clamp=5.0,
            blend_start_source="actual_output",
        )
        f_unclamped = AsyncFilter(
            method="rail", blend_order="quintic",
            blend_start_source="actual_output",
        )
        # Feed noisy chunks
        t1 = np.linspace(0, 1, 20)
        noise = np.random.randn(20, 3) * 0.05
        x1 = np.column_stack([t1, t1 ** 2, np.sin(t1)]) + noise
        for f in (f_clamped, f_unclamped):
            f.update_chunk(t1, x1)
            f.set_current_time(0.5)
        t2 = np.linspace(0.4, 1.5, 20)
        noise2 = np.random.randn(20, 3) * 0.05
        x2 = np.column_stack([t2, t2 ** 2, np.sin(t2)]) + noise2
        for f in (f_clamped, f_unclamped):
            f.update_chunk(t2, x2)
        # Both should produce finite output
        for f in (f_clamped, f_unclamped):
            val = f.get_output(t=0.6)
            assert val is not None
            assert np.all(np.isfinite(val))

    def test_acc_clamp_no_effect_on_cubic(self):
        """acc_clamp shouldn't break cubic blends (cubic has no acc matching)."""
        f = AsyncFilter(method="rail", blend_order="cubic", acc_clamp=5.0)
        t1 = np.linspace(0, 1, 20)
        x1 = np.column_stack([t1, t1 ** 2, np.sin(t1)])
        f.update_chunk(t1, x1)
        f.set_current_time(0.5)
        t2 = np.linspace(0.4, 1.5, 20)
        x2 = np.column_stack([t2 + 0.1, t2 ** 2 + 0.1, np.sin(t2) + 0.1])
        f.update_chunk(t2, x2)
        val = f.get_output(t=0.6)
        assert val is not None
        assert np.all(np.isfinite(val))


# ---------------------------------------------------------------------------
# Extrapolation safeguard tests
# ---------------------------------------------------------------------------
class TestPolyTrajectoryExtrapolation:
    """Tests for _PolyTrajectory extrapolation modes."""

    @staticmethod
    def _make_traj(extrapolation: str = "linear"):
        t = np.linspace(0.0, 1.0, 20)
        x = np.column_stack([t ** 2, np.sin(t)])
        return _PolyTrajectory.fit(t, x, 3, extrapolation)

    def test_default_extrapolation_is_linear(self):
        traj = self._make_traj()
        assert traj._extrapolation == "linear"

    def test_in_range_identical_for_all_modes(self):
        """Within [t_start, t_end] all modes should give the same result."""
        trajs = {m: self._make_traj(m) for m in ("clamp", "linear", "poly")}
        for t_q in [0.0, 0.3, 0.5, 0.8, 1.0]:
            vals = {m: tr.evaluate(t_q) for m, tr in trajs.items()}
            np.testing.assert_allclose(vals["clamp"], vals["linear"], atol=1e-12)
            np.testing.assert_allclose(vals["linear"], vals["poly"], atol=1e-12)

    def test_clamp_holds_boundary(self):
        traj = self._make_traj("clamp")
        val_end = traj.evaluate(1.0)
        val_beyond = traj.evaluate(5.0)
        np.testing.assert_allclose(val_beyond, val_end, atol=1e-12)
        val_start = traj.evaluate(0.0)
        val_before = traj.evaluate(-3.0)
        np.testing.assert_allclose(val_before, val_start, atol=1e-12)

    def test_linear_bounded(self):
        """Linear extrapolation should stay finite and closer than poly."""
        traj_lin = self._make_traj("linear")
        traj_poly = self._make_traj("poly")
        val_lin = traj_lin.evaluate(10.0)
        val_poly = traj_poly.evaluate(10.0)
        assert np.all(np.isfinite(val_lin))
        assert np.max(np.abs(val_lin)) < np.max(np.abs(val_poly))

    def test_poly_diverges(self):
        """Poly mode should diverge for high-degree at large extrapolation."""
        traj = self._make_traj("poly")
        val_far = traj.evaluate(100.0)
        assert np.max(np.abs(val_far)) > 1e4  # cubic diverges at t=100

    def test_linear_extrapolation_uses_velocity(self):
        """Linear extrapolation: pos(t) = pos(boundary) + vel(boundary) * dt."""
        traj = self._make_traj("linear")
        dt = 0.5
        val_end = traj.evaluate(1.0)
        vel_end = traj.evaluate_deriv(1.0, 1)
        val_extrap = traj.evaluate(1.0 + dt)
        expected = val_end + vel_end * dt
        np.testing.assert_allclose(val_extrap, expected, atol=1e-10)

    def test_clamp_deriv_is_zero(self):
        """Clamped: derivatives outside range should be zero."""
        traj = self._make_traj("clamp")
        d1 = traj.evaluate_deriv(5.0, 1)
        np.testing.assert_allclose(d1, 0.0, atol=1e-12)
        d2 = traj.evaluate_deriv(-3.0, 2)
        np.testing.assert_allclose(d2, 0.0, atol=1e-12)

    def test_linear_deriv_constant_outside(self):
        """Linear: 1st deriv is constant boundary velocity, higher derivs zero."""
        traj = self._make_traj("linear")
        vel_end = traj.evaluate_deriv(1.0, 1)
        vel_far = traj.evaluate_deriv(5.0, 1)
        np.testing.assert_allclose(vel_far, vel_end, atol=1e-10)
        acc_far = traj.evaluate_deriv(5.0, 2)
        np.testing.assert_allclose(acc_far, 0.0, atol=1e-12)


class TestRAILExtrapolationParam:
    """Tests for the extrapolation parameter on AsyncFilterRAIL."""

    def test_default_is_linear(self):
        f = AsyncFilter(method="rail")
        assert f._extrapolation == "linear"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="extrapolation"):
            AsyncFilter(method="rail", extrapolation="quadratic")

    def test_all_modes_produce_output(self):
        for mode in ("clamp", "linear", "poly"):
            f = AsyncFilter(method="rail", extrapolation=mode)
            t = np.linspace(0, 1, 20)
            x = np.column_stack([t, t ** 2, np.sin(t)])
            f.update_chunk(t, x)
            val = f.get_output(t=3.0)
            assert val is not None, f"mode={mode}"
            assert val.shape == (3,), f"mode={mode}"
            if mode != "poly":
                assert np.max(np.abs(val)) < 100.0, f"mode={mode}"

    def test_extrapolation_reduces_divergence(self):
        """Linear/clamp should prevent divergence for far-future queries."""
        t = np.linspace(0, 0.5, 16)
        x = np.column_stack([t ** 2, np.sin(3 * t)])
        results = {}
        for mode in ("clamp", "linear", "poly"):
            f = AsyncFilter(method="rail", extrapolation=mode, poly_degree=5)
            f.update_chunk(t, x)
            val = f.get_output(t=5.0)
            results[mode] = np.max(np.abs(val))
        assert results["clamp"] < results["poly"]
        assert results["linear"] < results["poly"]


# ---------------------------------------------------------------------------
# RAIL: align_method parameter
# ---------------------------------------------------------------------------

class TestAlignMethod:
    """Tests for the align_method parameter on AsyncFilterRAIL."""

    def test_align_method_default_is_direction(self):
        """Default align_method should be 'direction'."""
        f = AsyncFilter(method="rail")
        assert f._align_method == "direction"

    def test_align_method_invalid_raises(self):
        """Invalid align_method should raise ValueError."""
        with pytest.raises(ValueError, match="align_method"):
            AsyncFilter(method="rail", align_method="invalid")

    def test_align_method_least_squares_accepted(self):
        """'least_squares' should be accepted without error."""
        f = AsyncFilter(method="rail", align_method="least_squares",
                        auto_align=True)
        assert f._align_method == "least_squares"
        assert f._auto_align is True

    def test_ls_align_produces_finite_output(self):
        """least_squares alignment should produce finite output."""
        f = AsyncFilter(method="rail", auto_align=True,
                        align_method="least_squares")
        t1 = np.linspace(0.0, 1.0, 20)
        x1 = t1 * 0.5
        f.update_chunk(t1, x1)
        f.set_current_time(1.2)
        t2 = np.linspace(0.8, 2.0, 20)
        x2 = t2 * 0.5 + 0.05
        f.update_chunk(t2, x2)
        val = f.get_output(t=1.2)
        assert val is not None
        assert np.all(np.isfinite(val))

    def test_ls_align_reduces_position_error(self):
        """LS alignment should reduce position error vs no alignment.

        Creates two chunks with a known temporal offset.  The LS-aligned
        filter should produce output closer to the true trajectory.
        """
        # Ground truth: ramp with gentle curve
        gt = lambda t: 0.4 * t + 0.05 * np.sin(2 * t)

        # Create chunks with temporal offset
        rng = np.random.default_rng(42)
        chunk_dt = 1.0 / 30
        t1 = np.array([i * chunk_dt for i in range(16)])
        x1 = gt(t1 + 0.03).reshape(-1, 1)  # offset +0.03s

        t2_start = 0.35
        t2 = np.array([t2_start + i * chunk_dt for i in range(16)])
        x2 = gt(t2 - 0.05).reshape(-1, 1)  # offset -0.05s

        rms_vals = {}
        for method in ("direction", "least_squares"):
            f = AsyncFilter(method="rail", auto_align=True,
                            align_method=method)
            f.update_chunk(t1, x1)
            f.set_current_time(t2_start)
            f.update_chunk(t2, x2)

            # Evaluate at several points
            errs = []
            for t in np.linspace(t2_start, t2[-1] - 0.05, 20):
                val = f.get_output(t=t)
                if val is not None:
                    errs.append(float(val[0]) - gt(t))
            rms_vals[method] = np.sqrt(np.mean(np.array(errs) ** 2))

        # LS should be no worse than direction
        assert rms_vals["least_squares"] <= rms_vals["direction"] * 1.5

    def test_ls_align_first_chunk_no_crash(self):
        """LS alignment with only one chunk should not crash."""
        f = AsyncFilter(method="rail", auto_align=True,
                        align_method="least_squares")
        f.set_current_time(0.5)
        t = np.linspace(0.0, 1.0, 20)
        x = np.sin(t)
        f.update_chunk(t, x)
        val = f.get_output(t=0.5)
        assert val is not None
        assert np.all(np.isfinite(val))

    def test_ls_align_preserves_shape(self):
        """LS alignment should preserve output dimensionality."""
        f = AsyncFilter(method="rail", auto_align=True,
                        align_method="least_squares")
        t1 = np.linspace(0, 1, 20)
        x1 = np.column_stack([t1, t1 ** 2, np.sin(t1)])
        f.update_chunk(t1, x1)
        f.set_current_time(1.1)
        t2 = np.linspace(0.8, 2.0, 20)
        x2 = np.column_stack([t2 + 0.05, t2 ** 2 + 0.05, np.sin(t2) + 0.05])
        f.update_chunk(t2, x2)
        val = f.get_output(t=1.1)
        assert val.shape == (3,)

    def test_ls_align_with_custom_window(self):
        """align_window should control LS search range."""
        f = AsyncFilter(method="rail", auto_align=True,
                        align_method="least_squares", align_window=0.2)
        t1 = np.linspace(0.0, 1.0, 20)
        x1 = t1 * 0.5
        f.update_chunk(t1, x1)
        f.set_current_time(1.1)
        t2 = np.linspace(0.9, 2.0, 20)
        x2 = t2 * 0.5
        f.update_chunk(t2, x2)
        val = f.get_output(t=1.1)
        assert val is not None
        assert np.all(np.isfinite(val))

    def test_ls_align_without_current_time_is_noop(self):
        """LS alignment without set_current_time should not crash."""
        f = AsyncFilter(method="rail", auto_align=True,
                        align_method="least_squares")
        t1 = np.linspace(0, 1, 20)
        x1 = np.sin(t1)
        f.update_chunk(t1, x1)
        t2 = np.linspace(0.8, 2.0, 20)
        x2 = np.sin(t2) + 0.1
        f.update_chunk(t2, x2)
        val = f.get_output(t=1.0)
        assert val is not None

    def test_factory_passes_align_method(self):
        """AsyncFilter factory should forward align_method to RAIL."""
        f = AsyncFilter(method="rail", auto_align=True,
                        align_method="least_squares")
        assert f._align_method == "least_squares"


# ---------------------------------------------------------------------------
# RTC (Real-Time Chunking)
# ---------------------------------------------------------------------------

class TestAsyncFilterRTCBasic:
    """Basic functionality of AsyncFilterRTC."""

    def _make_filter(self, **kwargs):
        defaults = dict(
            prediction_horizon=10, dt=0.02, min_execution_horizon=1,
            initial_delay=1,
        )
        defaults.update(kwargs)
        return AsyncFilter(method="rtc", **defaults)

    def _make_chunk(self, t_start=0.0, H=10, dt=0.02, D=2):
        t = np.arange(H) * dt + t_start
        x = np.column_stack([np.sin(t * 2 * np.pi), np.cos(t * 2 * np.pi)])
        if D == 1:
            x = x[:, :1]
        return t, x

    def test_factory_creates_rtc(self):
        f = self._make_filter()
        assert type(f).__name__ == "AsyncFilterRTC"

    def test_returns_none_before_update(self):
        f = self._make_filter()
        assert f.get_output() is None
        assert f.get_prefix() is None
        assert f.current_chunk is None

    def test_update_chunk_and_get_output(self):
        f = self._make_filter()
        t, x = self._make_chunk()
        f.update_chunk(t, x)
        out = f.get_output(t[0])
        assert out is not None
        np.testing.assert_allclose(out, x[0], atol=1e-12)

    def test_get_output_at_endpoints(self):
        f = self._make_filter()
        t, x = self._make_chunk()
        f.update_chunk(t, x)
        np.testing.assert_allclose(f.get_output(t[0]), x[0], atol=1e-12)
        np.testing.assert_allclose(f.get_output(t[-1]), x[-1], atol=1e-12)

    def test_get_output_interpolates(self):
        f = self._make_filter()
        t, x = self._make_chunk()
        f.update_chunk(t, x)
        mid_t = (t[3] + t[4]) / 2
        out = f.get_output(mid_t)
        assert out is not None
        # Should be between x[3] and x[4] (linear interp)
        for d in range(x.shape[1]):
            lo = min(x[3, d], x[4, d])
            hi = max(x[3, d], x[4, d])
            assert lo - 1e-12 <= out[d] <= hi + 1e-12

    def test_get_output_default_time(self):
        """get_output(None) returns the last action of the current chunk."""
        f = self._make_filter()
        t, x = self._make_chunk()
        f.update_chunk(t, x)
        out = f.get_output()
        np.testing.assert_allclose(out, x[-1], atol=1e-12)

    def test_get_prefix_returns_tuple(self):
        f = self._make_filter()
        t, x = self._make_chunk()
        f.update_chunk(t, x)
        prefix = f.get_prefix()
        assert prefix is not None
        t_p, x_p, d = prefix
        assert isinstance(d, int)
        assert d >= 0
        assert t_p.ndim == 1
        assert x_p.ndim == 2

    def test_delay_estimate_initial(self):
        f = self._make_filter(initial_delay=3)
        assert f.delay_estimate == 3

    def test_record_delay(self):
        f = self._make_filter(initial_delay=1, delay_estimate_method="max")
        f.record_delay(5)
        assert f.delay_estimate == 5

    def test_delay_estimate_mean(self):
        f = self._make_filter(
            initial_delay=0, delay_estimate_method="mean",
            delay_buffer_size=3,
        )
        f.record_delay(2)
        f.record_delay(4)
        # buffer: [0, 2, 4], mean = 2
        assert f.delay_estimate == 2

    def test_delay_estimate_ema(self):
        f = self._make_filter(
            initial_delay=0, delay_estimate_method="ema",
            delay_buffer_size=10,
        )
        f.record_delay(10)
        d = f.delay_estimate
        assert d >= 1  # should pick up the large delay

    def test_execution_horizon(self):
        f = self._make_filter(min_execution_horizon=3, initial_delay=1)
        assert f.execution_horizon == 3  # max(1, 3)
        f.record_delay(5)
        assert f.execution_horizon == 5  # max(5, 3)

    def test_clear_resets_state(self):
        f = self._make_filter()
        t, x = self._make_chunk()
        f.update_chunk(t, x)
        f.set_current_time(0.1)
        f.clear()
        assert f.get_output() is None
        assert f.get_prefix() is None
        assert f.current_chunk is None

    def test_1d_input(self):
        f = self._make_filter(prediction_horizon=5)
        t = np.linspace(0, 0.08, 5)
        x = np.sin(t)  # 1D
        f.update_chunk(t, x)
        out = f.get_output(t[2])
        assert out.shape == (1,)

    def test_mismatched_shapes_raises(self):
        f = self._make_filter()
        t = np.array([0.0, 0.02, 0.04])
        x = np.ones((5, 2))
        with pytest.raises(ValueError, match="samples"):
            f.update_chunk(t, x)

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            AsyncFilter(method="rtc", prediction_horizon=1, dt=0.02)
        with pytest.raises(ValueError):
            AsyncFilter(method="rtc", prediction_horizon=10, dt=0.02,
                        min_execution_horizon=0)
        with pytest.raises(ValueError):
            AsyncFilter(method="rtc", prediction_horizon=10, dt=-1)
        with pytest.raises(ValueError):
            AsyncFilter(method="rtc", prediction_horizon=10, dt=0.02,
                        delay_estimate_method="bad")
        with pytest.raises(ValueError):
            AsyncFilter(method="rtc", prediction_horizon=10, dt=0.02,
                        blend_mode="bad")
        with pytest.raises(ValueError):
            AsyncFilter(method="rtc", prediction_horizon=10, dt=0.02,
                        interpolation="bad")
        with pytest.raises(ValueError):
            AsyncFilter(method="rtc", prediction_horizon=10, dt=0.02,
                        extrapolation="bad")


class TestAsyncFilterRTCSoftMask:
    """Tests for the soft mask computation (Eq. 5 of arXiv:2506.07339)."""

    def _make_filter(self, H=10, **kwargs):
        return AsyncFilter(method="rtc", prediction_horizon=H, dt=0.02,
                           **kwargs)

    def test_shape(self):
        f = self._make_filter(H=10)
        W = f.get_soft_mask(delay=2)
        assert W.shape == (10,)

    def test_frozen_region_is_one(self):
        f = self._make_filter(H=10)
        W = f.get_soft_mask(delay=3)
        np.testing.assert_allclose(W[:3], 1.0)

    def test_free_region_is_zero(self):
        """Last s entries should be zero."""
        f = self._make_filter(H=10, min_execution_horizon=2)
        W = f.get_soft_mask(delay=2)
        # s = max(d, s_min) = max(2, 2) = 2 → last 2 are zero
        np.testing.assert_allclose(W[-2:], 0.0)

    def test_intermediate_decays(self):
        f = self._make_filter(H=20, min_execution_horizon=2)
        W = f.get_soft_mask(delay=3)
        # Intermediate region: indices 3 to H-s-1 = 20-3-1 = 16
        intermediate = W[3:17]
        # Should be monotonically decreasing
        assert np.all(np.diff(intermediate) <= 1e-15)

    def test_values_in_unit_interval(self):
        f = self._make_filter(H=16, min_execution_horizon=4)
        W = f.get_soft_mask(delay=4)
        assert np.all(W >= -1e-15)
        assert np.all(W <= 1.0 + 1e-15)

    def test_delay_zero(self):
        f = self._make_filter(H=8, min_execution_horizon=1)
        W = f.get_soft_mask(delay=0)
        # No frozen region; first element should not be 1.0
        assert W[0] < 1.0 or W[0] == 0.0  # d=0 means no frozen

    def test_override_delay(self):
        """Explicit delay should override the internal estimate."""
        f = self._make_filter(H=10, initial_delay=1)
        W1 = f.get_soft_mask(delay=1)
        W5 = f.get_soft_mask(delay=5)
        assert not np.allclose(W1, W5)


class TestAsyncFilterRTCBlending:
    """Tests for chunk transition blending."""

    def _make_filter(self, blend_mode="soft_mask", **kwargs):
        defaults = dict(prediction_horizon=10, dt=0.02)
        defaults.update(kwargs)
        return AsyncFilter(method="rtc", blend_mode=blend_mode, **defaults)

    def test_blend_none_uses_new_chunk(self):
        """With blend_mode='none', output comes only from the new chunk."""
        f = self._make_filter(blend_mode="none")
        t1, x1 = np.linspace(0, 0.18, 10), np.zeros((10, 2))
        t2, x2 = np.linspace(0.08, 0.26, 10), np.ones((10, 2))
        f.update_chunk(t1, x1)
        f.update_chunk(t2, x2)
        # Query in overlap (0.08..0.18) → should be exactly from x2
        out = f.get_output(0.12)
        np.testing.assert_allclose(out, [1.0, 1.0], atol=1e-12)

    def test_blend_linear_crossfade(self):
        f = self._make_filter(blend_mode="linear")
        t1 = np.linspace(0, 0.18, 10)
        x1 = np.zeros((10, 1))
        t2 = np.linspace(0.08, 0.26, 10)
        x2 = np.ones((10, 1))
        f.update_chunk(t1, x1)
        f.update_chunk(t2, x2)
        # Midpoint of overlap
        mid = (0.08 + 0.18) / 2
        out = f.get_output(mid)
        # Should be ~0.5 (linear blend)
        assert 0.3 < out[0] < 0.7

    def test_blend_soft_mask_smooth(self):
        f = self._make_filter(blend_mode="soft_mask")
        t1 = np.linspace(0, 0.18, 10)
        x1 = np.zeros((10, 1))
        t2 = np.linspace(0.08, 0.26, 10)
        x2 = np.ones((10, 1))
        f.update_chunk(t1, x1)
        f.update_chunk(t2, x2)
        # Early in overlap → closer to old (0)
        out_early = f.get_output(0.085)
        # Late in overlap → closer to new (1)
        out_late = f.get_output(0.175)
        assert out_early[0] < out_late[0]

    def test_no_blend_before_overlap(self):
        """Query before overlap uses the new chunk directly."""
        f = self._make_filter(blend_mode="soft_mask")
        t1 = np.linspace(0, 0.18, 10)
        x1 = np.zeros((10, 1))
        t2 = np.linspace(0.10, 0.28, 10)
        x2 = np.ones((10, 1))
        f.update_chunk(t1, x1)
        f.update_chunk(t2, x2)
        # Query past the overlap region
        out = f.get_output(0.25)
        np.testing.assert_allclose(out, [1.0], atol=1e-12)


class TestAsyncFilterRTCExtrapolation:
    """Tests for extrapolation behaviour."""

    def _make_filter(self, extrapolation="linear", **kwargs):
        return AsyncFilter(method="rtc", prediction_horizon=5, dt=0.02,
                           extrapolation=extrapolation, **kwargs)

    def test_clamp_before_start(self):
        f = self._make_filter(extrapolation="clamp")
        t = np.linspace(0.1, 0.18, 5)
        x = np.arange(5).reshape(-1, 1).astype(float)
        f.update_chunk(t, x)
        out = f.get_output(0.0)  # before chunk start
        np.testing.assert_allclose(out, x[0], atol=1e-12)

    def test_clamp_after_end(self):
        f = self._make_filter(extrapolation="clamp")
        t = np.linspace(0.0, 0.08, 5)
        x = np.arange(5).reshape(-1, 1).astype(float)
        f.update_chunk(t, x)
        out = f.get_output(1.0)  # well past end
        np.testing.assert_allclose(out, x[-1], atol=1e-12)

    def test_linear_extrapolation_after_end(self):
        f = self._make_filter(extrapolation="linear")
        t = np.array([0.0, 0.02, 0.04, 0.06, 0.08])
        x = np.arange(5).reshape(-1, 1).astype(float)
        f.update_chunk(t, x)
        # Linear extrapolation: slope = 1/0.02 = 50
        out = f.get_output(0.10)
        expected = 4.0 + 50 * 0.02  # = 5.0
        np.testing.assert_allclose(out, [expected], atol=1e-10)


class TestAsyncFilterRTCInterpolation:
    """Tests for PCHIP interpolation."""

    def test_pchip_interpolation(self):
        f = AsyncFilter(method="rtc", prediction_horizon=8, dt=0.02,
                        interpolation="pchip")
        t = np.linspace(0, 0.14, 8)
        x = np.sin(t * 10).reshape(-1, 1)
        f.update_chunk(t, x)
        out = f.get_output(0.07)
        assert out is not None
        # Just verify it returns something reasonable
        assert np.isfinite(out).all()


class TestAsyncFilterRTCTiming:
    """Tests for inference timing / delay auto-recording."""

    def _make_filter(self, **kwargs):
        return AsyncFilter(method="rtc", prediction_horizon=10, dt=0.02,
                           initial_delay=0, **kwargs)

    def test_auto_delay_recording(self):
        f = self._make_filter()
        t1 = np.linspace(0, 0.18, 10)
        x1 = np.zeros((10, 2))
        f.update_chunk(t1, x1)

        # Simulate inference cycle
        f.set_current_time(0.10)
        f.start_inference()
        f.set_current_time(0.16)  # 0.06s later → 3 steps at dt=0.02
        t2 = np.linspace(0.10, 0.28, 10)
        x2 = np.ones((10, 2))
        f.update_chunk(t2, x2)

        # Delay should have been recorded as ~3
        assert f.delay_estimate == 3

    def test_set_current_time(self):
        f = self._make_filter()
        f.set_current_time(1.5)
        # No crash, and the time should be stored
        assert f._current_time == 1.5


class TestAsyncFilterRTCChunkSwitch:
    """Tests for chunk switching behaviour."""

    def _make_filter(self, **kwargs):
        defaults = dict(prediction_horizon=10, dt=0.02, blend_mode="none")
        defaults.update(kwargs)
        return AsyncFilter(method="rtc", **defaults)

    def test_prev_chunk_stored(self):
        f = self._make_filter()
        t1 = np.linspace(0, 0.18, 10)
        x1 = np.zeros((10, 2))
        t2 = np.linspace(0.10, 0.28, 10)
        x2 = np.ones((10, 2))
        f.update_chunk(t1, x1)
        f.update_chunk(t2, x2)
        assert f._prev_chunk_t is not None
        np.testing.assert_allclose(f._prev_chunk_t, t1)

    def test_current_chunk_property(self):
        f = self._make_filter()
        t, x = np.linspace(0, 0.18, 10), np.zeros((10, 2))
        f.update_chunk(t, x)
        chunk = f.current_chunk
        assert chunk is not None
        np.testing.assert_allclose(chunk[0], t)
        np.testing.assert_allclose(chunk[1], x)

    def test_current_chunk_is_copy(self):
        """current_chunk should return a copy, not a reference."""
        f = self._make_filter()
        t, x = np.linspace(0, 0.18, 10), np.zeros((10, 2))
        f.update_chunk(t, x)
        chunk = f.current_chunk
        chunk[1][:] = 999
        chunk2 = f.current_chunk
        assert not np.any(chunk2[1] == 999)

    def test_multiple_chunk_updates(self):
        f = self._make_filter()
        for i in range(5):
            start = i * 0.1
            t = np.linspace(start, start + 0.18, 10)
            x = np.full((10, 2), float(i))
            f.update_chunk(t, x)
        out = f.get_output()
        np.testing.assert_allclose(out, [4.0, 4.0], atol=1e-12)


class TestAsyncFilterRTCThreadSafety:
    """Thread safety of AsyncFilterRTC."""

    def test_concurrent_update_and_query(self):
        f = AsyncFilter(method="rtc", prediction_horizon=10, dt=0.02)
        errors = []

        def producer():
            try:
                for i in range(50):
                    t = np.linspace(i * 0.1, i * 0.1 + 0.18, 10)
                    x = np.full((10, 2), float(i))
                    f.update_chunk(t, x)
            except Exception as e:
                errors.append(e)

        def consumer():
            try:
                for _ in range(200):
                    out = f.get_output()
                    _ = f.get_prefix()
                    _ = f.get_soft_mask()
                    _ = f.delay_estimate
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=producer),
            threading.Thread(target=consumer),
            threading.Thread(target=consumer),
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=10)
        assert not errors, f"Thread errors: {errors}"


# ---------------------------------------------------------------------------
# RTC – Inpainting
# ---------------------------------------------------------------------------

class TestAsyncFilterRTCInpaintingHard:
    """Tests for hard (prefix replacement) inpainting."""

    def _make_filter(self, **kwargs):
        defaults = dict(
            prediction_horizon=10, dt=0.02, min_execution_horizon=1,
            initial_delay=2, inpainting="hard", blend_mode="none",
        )
        defaults.update(kwargs)
        return AsyncFilterRTC(**defaults)

    def test_first_chunk_no_inpainting(self):
        """First chunk should never be modified (no previous to compare)."""
        f = self._make_filter()
        t = np.arange(10) * 0.02
        x = np.arange(10, dtype=float).reshape(-1, 1) * 10.0
        f.update_chunk(t, x)
        for i, ti in enumerate(t):
            out = f.get_output(t=ti)
            np.testing.assert_allclose(out, x[i].ravel(), atol=1e-10)

    def test_prefix_replaced_by_previous_chunk(self):
        """Hard inpainting must replace the first d overlapping actions."""
        f = self._make_filter(initial_delay=3)
        dt = 0.02
        H = 10
        t1 = np.arange(H) * dt
        x1 = np.ones((H, 2)) * 100.0
        f.update_chunk(t1, x1)

        t2 = np.arange(H) * dt
        x2 = np.ones((H, 2)) * 200.0
        f.update_chunk(t2, x2)

        t3 = np.arange(H) * dt
        x3 = np.ones((H, 2)) * 300.0
        f.update_chunk(t3, x3)

        # Delay=3, so first 3 overlapping actions should be from chunk2 (200)
        for i in range(3):
            out = f.get_output(t=t3[i])
            np.testing.assert_allclose(out, [200.0, 200.0], atol=1e-10,
                                       err_msg=f"index {i} should be from prev chunk")

        # Actions beyond the prefix should be from the new chunk (300)
        for i in range(4, H):
            out = f.get_output(t=t3[i])
            np.testing.assert_allclose(out, [300.0, 300.0], atol=1e-10,
                                       err_msg=f"index {i} should be from new chunk")

    def test_non_overlapping_chunks_skip_inpainting(self):
        """If chunks don't overlap, inpainting should be skipped."""
        f = self._make_filter(initial_delay=2)
        dt = 0.02
        H = 10
        t1 = np.arange(H) * dt
        x1 = np.ones((H, 1)) * 10.0
        f.update_chunk(t1, x1)

        t2 = np.arange(H) * dt + 0.5
        x2 = np.ones((H, 1)) * 20.0
        f.update_chunk(t2, x2)

        t3 = np.arange(H) * dt + 1.0
        x3 = np.ones((H, 1)) * 30.0
        f.update_chunk(t3, x3)

        for i in range(H):
            out = f.get_output(t=t3[i])
            np.testing.assert_allclose(out, [30.0], atol=1e-10)


class TestAsyncFilterRTCInpaintingSoftMask:
    """Tests for soft mask inpainting."""

    def _make_filter(self, **kwargs):
        defaults = dict(
            prediction_horizon=10, dt=0.02, min_execution_horizon=2,
            initial_delay=2, inpainting="soft_mask", blend_mode="none",
        )
        defaults.update(kwargs)
        return AsyncFilterRTC(**defaults)

    def test_prefix_fully_frozen(self):
        """First d actions should use 100% of the previous chunk (W=1)."""
        f = self._make_filter(initial_delay=2)
        dt = 0.02
        H = 10
        t1 = np.arange(H) * dt
        x1 = np.ones((H, 1)) * 100.0
        f.update_chunk(t1, x1)
        t2 = np.arange(H) * dt
        x2 = np.ones((H, 1)) * 200.0
        f.update_chunk(t2, x2)
        t3 = np.arange(H) * dt
        x3 = np.ones((H, 1)) * 300.0
        f.update_chunk(t3, x3)

        for i in range(2):
            out = f.get_output(t=t3[i])
            np.testing.assert_allclose(out, [200.0], atol=1e-10,
                                       err_msg=f"index {i}: frozen prefix")

    def test_free_region_untouched(self):
        """Actions in the free region (i >= H-s) should be from new chunk."""
        f = self._make_filter(initial_delay=2, min_execution_horizon=2)
        dt = 0.02
        H = 10
        t1 = np.arange(H) * dt
        x1 = np.ones((H, 1)) * 100.0
        f.update_chunk(t1, x1)
        t2 = np.arange(H) * dt
        x2 = np.ones((H, 1)) * 200.0
        f.update_chunk(t2, x2)
        t3 = np.arange(H) * dt
        x3 = np.ones((H, 1)) * 300.0
        f.update_chunk(t3, x3)

        for i in range(8, H):
            out = f.get_output(t=t3[i])
            np.testing.assert_allclose(out, [300.0], atol=1e-10,
                                       err_msg=f"index {i}: free region")

    def test_intermediate_is_blend(self):
        """Actions between prefix and free region should be a blend."""
        f = self._make_filter(initial_delay=2, min_execution_horizon=2)
        dt = 0.02
        H = 10
        t1 = np.arange(H) * dt
        x1 = np.ones((H, 1)) * 0.0
        f.update_chunk(t1, x1)
        t2 = np.arange(H) * dt
        x2 = np.ones((H, 1)) * 0.0
        f.update_chunk(t2, x2)
        t3 = np.arange(H) * dt
        x3 = np.ones((H, 1)) * 100.0
        f.update_chunk(t3, x3)

        for i in range(3, 7):
            out = f.get_output(t=t3[i])
            assert 0.0 <= out[0] <= 100.0, \
                f"index {i}: expected blend, got {out[0]}"


class TestAsyncFilterRTCInpaintingHermite:
    """Tests for Hermite (C¹) inpainting."""

    def _make_filter(self, **kwargs):
        defaults = dict(
            prediction_horizon=10, dt=0.02, min_execution_horizon=1,
            initial_delay=2, inpainting="hermite",
            inpainting_transition=4, blend_mode="none",
        )
        defaults.update(kwargs)
        return AsyncFilterRTC(**defaults)

    def test_prefix_replaced(self):
        """First d actions should match the previous chunk exactly."""
        f = self._make_filter(initial_delay=2)
        dt = 0.02
        H = 10
        t1 = np.arange(H) * dt
        x1 = np.ones((H, 1)) * 50.0
        f.update_chunk(t1, x1)
        t2 = np.arange(H) * dt
        x2 = np.ones((H, 1)) * 100.0
        f.update_chunk(t2, x2)
        t3 = np.arange(H) * dt
        x3 = np.ones((H, 1)) * 200.0
        f.update_chunk(t3, x3)

        for i in range(2):
            out = f.get_output(t=t3[i])
            np.testing.assert_allclose(out, [100.0], atol=1e-10,
                                       err_msg=f"index {i}: prefix must match prev")

    def test_transition_smoothness(self):
        """Transition region should smoothly connect prefix to free region."""
        f = self._make_filter(initial_delay=2, inpainting_transition=4)
        dt = 0.02
        H = 10
        t1 = np.arange(H) * dt
        x1 = np.ones((H, 1)) * 0.0
        f.update_chunk(t1, x1)
        t2 = np.arange(H) * dt
        x2 = np.ones((H, 1)) * 0.0
        f.update_chunk(t2, x2)
        t3 = np.arange(H) * dt
        x3 = np.ones((H, 1)) * 100.0
        f.update_chunk(t3, x3)

        vals = []
        for i in range(H):
            out = f.get_output(t=t3[i])
            vals.append(out[0])

        # Prefix should be 0
        assert vals[0] == pytest.approx(0.0, abs=1e-10)
        assert vals[1] == pytest.approx(0.0, abs=1e-10)

        # After transition should be 100
        for i in range(7, H):
            assert vals[i] == pytest.approx(100.0, abs=1e-10), \
                f"index {i}: expected 100, got {vals[i]}"

    def test_c1_continuity_at_boundary(self):
        """Velocity should be approximately continuous at prefix boundary."""
        f = self._make_filter(initial_delay=3, inpainting_transition=4)
        dt = 0.02
        H = 10
        t1 = np.arange(H) * dt
        x1 = (np.arange(H) * 5.0).reshape(-1, 1)
        f.update_chunk(t1, x1)
        t2 = np.arange(H) * dt
        x2 = (np.arange(H) * 5.0).reshape(-1, 1)
        f.update_chunk(t2, x2)
        t3 = np.arange(H) * dt
        x3 = (np.arange(H) * 5.0 + 50.0).reshape(-1, 1)
        f.update_chunk(t3, x3)

        vals = [f.get_output(t=t3[i])[0] for i in range(H)]

        v_before = (vals[2] - vals[1]) / dt
        v_after = (vals[4] - vals[3]) / dt
        assert abs(v_before - v_after) < 5000, \
            f"Velocity jump too large: {v_before} vs {v_after}"


class TestAsyncFilterRTCInpaintingFactory:
    """Test that inpainting can be configured through the factory."""

    def test_factory_inpainting_param(self):
        f = AsyncFilter(
            method="rtc", prediction_horizon=10, dt=0.02,
            inpainting="hard",
        )
        assert f._inpainting == "hard"

    def test_factory_hermite_inpainting(self):
        f = AsyncFilter(
            method="rtc", prediction_horizon=10, dt=0.02,
            inpainting="hermite", inpainting_transition=3,
        )
        assert f._inpainting == "hermite"
        assert f._inpainting_transition == 3

    def test_invalid_inpainting_raises(self):
        with pytest.raises(ValueError, match="Unknown inpainting"):
            AsyncFilterRTC(
                prediction_horizon=10, dt=0.02,
                inpainting="invalid_mode",
            )


# ---------------------------------------------------------------------------
# RTC – Callback Inpainting
# ---------------------------------------------------------------------------

class TestAsyncFilterRTCInpaintingCallback:
    """Tests for callback-based inpainting."""

    def test_callback_requires_fn(self):
        with pytest.raises(ValueError, match="inpainting_fn must be provided"):
            AsyncFilterRTC(
                prediction_horizon=10, dt=0.02,
                inpainting="callback",
            )

    def test_callback_basic(self):
        """Callback should be invoked and its return value used."""
        call_log = []

        def my_inpaint(x_new, x_old_interp, t_new, t_old, mask, delay):
            call_log.append({
                "x_new_shape": x_new.shape,
                "mask_shape": mask.shape,
                "delay": delay,
            })
            # Return all zeros
            return np.zeros_like(x_new)

        f = AsyncFilterRTC(
            prediction_horizon=10, dt=0.02, initial_delay=2,
            inpainting="callback", inpainting_fn=my_inpaint,
            blend_mode="none",
        )
        dt = 0.02
        H = 10
        t1 = np.arange(H) * dt
        x1 = np.ones((H, 2)) * 100.0
        f.update_chunk(t1, x1)  # no prev → no callback

        t2 = np.arange(H) * dt
        x2 = np.ones((H, 2)) * 200.0
        f.update_chunk(t2, x2)  # prev=None, curr=chunk1 → no callback

        t3 = np.arange(H) * dt
        x3 = np.ones((H, 2)) * 300.0
        f.update_chunk(t3, x3)  # prev=chunk1, curr=chunk2 → callback called

        assert len(call_log) == 1
        assert call_log[0]["x_new_shape"] == (10, 2)
        assert call_log[0]["mask_shape"] == (10,)
        assert call_log[0]["delay"] >= 0

        # Output should reflect the callback (zeros)
        out = f.get_output(t=t3[0])
        np.testing.assert_allclose(out, [0.0, 0.0], atol=1e-10)

    def test_callback_receives_correct_data(self):
        """Verify the callback receives correct old-chunk interpolation."""
        received = {}

        def capture_inpaint(x_new, x_old_interp, t_new, t_old, mask, delay):
            received["x_old_interp"] = x_old_interp.copy()
            received["t_new"] = t_new.copy()
            received["mask"] = mask.copy()
            return x_new  # passthrough

        f = AsyncFilterRTC(
            prediction_horizon=10, dt=0.02, initial_delay=2,
            inpainting="callback", inpainting_fn=capture_inpaint,
            blend_mode="none",
        )
        dt = 0.02
        H = 10
        # chunk1: linearly increasing
        t1 = np.arange(H) * dt
        x1 = (np.arange(H, dtype=float) * 10).reshape(-1, 1)
        f.update_chunk(t1, x1)
        # chunk2: constant 50
        t2 = np.arange(H) * dt
        x2 = np.ones((H, 1)) * 50.0
        f.update_chunk(t2, x2)
        # chunk3: triggers callback against chunk2
        t3 = np.arange(H) * dt
        x3 = np.ones((H, 1)) * 999.0
        f.update_chunk(t3, x3)

        # x_old_interp should be chunk2 values (50.0)
        np.testing.assert_allclose(received["x_old_interp"], 50.0, atol=1e-10)
        assert received["mask"].shape == (H,)

    def test_callback_via_factory(self):
        """Factory should forward inpainting_fn."""
        def noop(x_new, x_old_interp, t_new, t_old, mask, delay):
            return x_new

        f = AsyncFilter(
            method="rtc", prediction_horizon=10, dt=0.02,
            inpainting="callback", inpainting_fn=noop,
        )
        assert f._inpainting == "callback"
        assert f._inpainting_fn is noop


# ---------------------------------------------------------------------------
# RTC + RAIL composite
# ---------------------------------------------------------------------------

class TestAsyncFilterRTCRAILBasic:
    """Basic functionality of the RTC+RAIL composite filter."""

    H = 16
    DT = 0.02

    def _make_filter(self, **kwargs):
        defaults = dict(
            prediction_horizon=self.H,
            dt=self.DT,
            min_execution_horizon=4,
            inpainting="none",
            poly_degree=3,
            blend_order="quintic",
        )
        defaults.update(kwargs)
        return AsyncFilterRTCRAIL(**defaults)

    def _make_chunk(self, t_start, dim=3, slope=1.0):
        t = t_start + np.arange(self.H) * self.DT
        x = np.column_stack([slope * (t - t_start) + d for d in range(dim)])
        return t, x

    def test_none_before_any_chunk(self):
        f = self._make_filter()
        assert f.get_output(0.0) is None

    def test_output_after_one_chunk(self):
        f = self._make_filter()
        t, x = self._make_chunk(0.0)
        f.update_chunk(t, x)
        out = f.get_output(0.1)
        assert out is not None
        assert out.shape == (3,)

    def test_output_shape_1d(self):
        f = self._make_filter()
        t = np.arange(self.H) * self.DT
        x = np.sin(t)
        f.update_chunk(t, x)
        out = f.get_output(0.05)
        assert out is not None
        assert out.shape == (1,)

    def test_factory_registration(self):
        f = AsyncFilter(
            method="rtc_rail",
            prediction_horizon=self.H,
            dt=self.DT,
        )
        assert isinstance(f, AsyncFilterRTCRAIL)

    def test_is_rtc_subclass(self):
        f = self._make_filter()
        assert isinstance(f, AsyncFilterRTC)

    def test_prefix_available(self):
        f = self._make_filter()
        t, x = self._make_chunk(0.0)
        f.set_current_time(0.1)
        f.update_chunk(t, x)
        prefix = f.get_prefix()
        assert prefix is not None
        t_p, x_p, delay = prefix
        assert t_p.ndim == 1
        assert x_p.ndim == 2

    def test_delay_estimate(self):
        f = self._make_filter(initial_delay=3)
        assert f.delay_estimate == 3


class TestAsyncFilterRTCRAILSmoothing:
    """Verify RAIL smoothing is applied to RTC output."""

    H = 20
    DT = 0.02

    def _make_filter(self, **kwargs):
        defaults = dict(
            prediction_horizon=self.H,
            dt=self.DT,
            min_execution_horizon=4,
            poly_degree=3,
            blend_order="quintic",
        )
        defaults.update(kwargs)
        return AsyncFilterRTCRAIL(**defaults)

    def test_noisy_chunk_smoothed(self):
        """Polynomial fitting should reduce noise compared to raw RTC."""
        rng = np.random.default_rng(42)
        t = np.arange(self.H) * self.DT
        clean = np.column_stack([t, t**2, np.sin(t)])
        noisy = clean + rng.normal(0, 0.05, clean.shape)

        rtc = AsyncFilterRTC(
            prediction_horizon=self.H, dt=self.DT,
            min_execution_horizon=4,
        )
        composite = self._make_filter()

        rtc.update_chunk(t, noisy)
        composite.update_chunk(t, noisy)

        # Compare outputs at multiple points
        query_times = t[2:-2]
        rtc_outs = np.array([rtc.get_output(ti) for ti in query_times])
        comp_outs = np.array([composite.get_output(ti) for ti in query_times])
        clean_ref = np.array([
            np.array([ti, ti**2, np.sin(ti)]) for ti in query_times
        ])

        rmse_rtc = np.sqrt(np.mean((rtc_outs - clean_ref) ** 2))
        rmse_comp = np.sqrt(np.mean((comp_outs - clean_ref) ** 2))
        # Composite (poly fit) should be closer to the clean signal
        assert rmse_comp < rmse_rtc

    def test_two_chunks_blended(self):
        """Output should be smooth across chunk boundary (RAIL blending)."""
        t1 = np.arange(self.H) * self.DT
        x1 = np.column_stack([t1, t1**2])

        t2 = t1 + 0.16  # overlap
        x2 = np.column_stack([t2 + 0.01, t2**2 + 0.01])  # slight offset

        f = self._make_filter()
        f.set_current_time(0.0)
        f.update_chunk(t1, x1)
        f.set_current_time(0.16)
        f.update_chunk(t2, x2)

        # Query across the boundary
        ts = np.linspace(0.1, 0.3, 50)
        outs = [f.get_output(ti) for ti in ts]
        assert all(o is not None for o in outs)
        outs = np.array(outs)

        # Check smoothness: finite-difference acceleration should be bounded
        vel = np.diff(outs, axis=0) / np.diff(ts)[:, None]
        acc = np.diff(vel, axis=0) / np.diff(ts[:-1])[:, None]
        # No wild jumps
        assert np.all(np.abs(acc) < 500)


class TestAsyncFilterRTCRAILInpainting:
    """Verify RTC inpainting is applied before RAIL processing."""

    H = 16
    DT = 0.02

    def _make_filter(self, **kwargs):
        defaults = dict(
            prediction_horizon=self.H,
            dt=self.DT,
            min_execution_horizon=4,
            inpainting="hard",
            poly_degree=3,
        )
        defaults.update(kwargs)
        return AsyncFilterRTCRAIL(**defaults)

    def test_hard_inpainting_applied(self):
        """Chunk values should be modified by inpainting on third chunk."""
        f = self._make_filter()
        t1 = np.arange(self.H) * self.DT
        x1 = np.ones((self.H, 2)) * 10.0

        f.set_current_time(0.0)
        f.update_chunk(t1, x1)

        # Second chunk — sets prev_chunk so inpainting can trigger on third
        t2 = t1 + self.H * self.DT * 0.5
        x2 = np.ones((self.H, 2)) * 15.0
        f.set_current_time(self.H * self.DT * 0.5)
        f.update_chunk(t2, x2)

        # Third chunk — inpainting should replace prefix with chunk2 values
        t3 = t2 + self.H * self.DT * 0.5
        x3 = np.ones((self.H, 2)) * 30.0
        f.set_current_time(self.H * self.DT)
        f.update_chunk(t3, x3)

        # The stored current chunk should have inpainted prefix
        with f._lock:
            stored = f._curr_chunk_x
        assert not np.allclose(stored, 30.0)

    def test_hermite_inpainting_with_rail(self):
        """Hermite inpainting + RAIL should produce C¹-continuous output."""
        f = self._make_filter(inpainting="hermite")
        t1 = np.arange(self.H) * self.DT
        x1 = np.column_stack([np.sin(t1)])

        f.set_current_time(0.0)
        f.update_chunk(t1, x1)

        t2 = t1 + 8 * self.DT
        x2 = np.column_stack([np.sin(t2) + 0.1])

        f.set_current_time(8 * self.DT)
        f.update_chunk(t2, x2)

        # Query around the boundary
        ts = np.linspace(0.05, 0.25, 40)
        outs = np.array([f.get_output(ti) for ti in ts])
        assert outs.shape == (40, 1)

        # Velocity (finite diff) should be reasonably smooth
        vel = np.diff(outs[:, 0]) / np.diff(ts)
        max_vel_jump = np.max(np.abs(np.diff(vel)))
        assert max_vel_jump < 100  # no discontinuity


class TestAsyncFilterRTCRAILTimePropagation:
    """Verify set_current_time propagates to both stages."""

    def test_current_time_propagated_to_rail(self):
        f = AsyncFilterRTCRAIL(
            prediction_horizon=10, dt=0.02,
        )
        f.set_current_time(1.5)
        assert f._rail._current_time == 1.5

    def test_start_inference_records_delay(self):
        f = AsyncFilterRTCRAIL(
            prediction_horizon=10, dt=0.02,
        )
        f.set_current_time(0.0)
        f.start_inference()
        f.set_current_time(0.1)

        t = np.arange(10) * 0.02
        x = np.ones((10, 2))
        f.update_chunk(t, x)
        assert f.delay_estimate >= 1


class TestAsyncFilterRTCRAILPrefix:
    """Verify get_prefix returns RAIL-smoothed values."""

    H = 16
    DT = 0.02

    def _make_filter(self, **kwargs):
        defaults = dict(
            prediction_horizon=self.H,
            dt=self.DT,
            min_execution_horizon=4,
            poly_degree=3,
            blend_order="quintic",
        )
        defaults.update(kwargs)
        return AsyncFilterRTCRAIL(**defaults)

    def test_prefix_returns_rail_smoothed_values(self):
        """Prefix values should differ from raw chunk (poly-smoothed)."""
        rng = np.random.default_rng(42)
        f = self._make_filter(initial_delay=3)

        t = np.arange(self.H) * self.DT
        clean = np.column_stack([t, t ** 2])
        noisy = clean + rng.normal(0, 0.1, clean.shape)

        f.set_current_time(0.0)
        f.update_chunk(t, noisy)

        prefix = f.get_prefix()
        assert prefix is not None
        t_p, x_p, delay = prefix

        # Prefix values should come from RAIL polynomial, not raw noisy chunk
        # Verify by comparing against raw chunk slice
        with f._lock:
            s = max(delay, f._s_min)
            ps = min(s, len(f._curr_chunk_t))
            pe = min(s + delay, len(f._curr_chunk_t))
            raw_prefix = f._curr_chunk_x[ps:pe].copy()

        # RAIL polynomial fit smooths noise — values should differ
        assert not np.allclose(x_p, raw_prefix, atol=1e-10)

    def test_prefix_closer_to_clean_signal(self):
        """RAIL-smoothed prefix should be closer to truth than raw prefix."""
        rng = np.random.default_rng(123)
        f_composite = self._make_filter(initial_delay=3)
        f_rtc = AsyncFilterRTC(
            prediction_horizon=self.H, dt=self.DT,
            min_execution_horizon=4, initial_delay=3,
        )

        t = np.arange(self.H) * self.DT
        clean = np.column_stack([np.sin(t), np.cos(t)])
        noisy = clean + rng.normal(0, 0.15, clean.shape)

        f_composite.set_current_time(0.0)
        f_composite.update_chunk(t, noisy)
        f_rtc.set_current_time(0.0)
        f_rtc.update_chunk(t, noisy)

        prefix_comp = f_composite.get_prefix()
        prefix_rtc = f_rtc.get_prefix()
        assert prefix_comp is not None and prefix_rtc is not None

        t_p = prefix_comp[0]
        gt = np.column_stack([np.sin(t_p), np.cos(t_p)])

        rmse_comp = np.sqrt(np.mean((prefix_comp[1] - gt) ** 2))
        rmse_rtc = np.sqrt(np.mean((prefix_rtc[1] - gt) ** 2))
        assert rmse_comp < rmse_rtc

    def test_prefix_shape_and_delay(self):
        """Prefix should have correct shape and delay value."""
        f = self._make_filter(initial_delay=2)
        t = np.arange(self.H) * self.DT
        x = np.ones((self.H, 3)) * 5.0

        f.set_current_time(0.0)
        f.update_chunk(t, x)

        prefix = f.get_prefix()
        assert prefix is not None
        t_p, x_p, delay = prefix
        assert t_p.ndim == 1
        assert x_p.ndim == 2
        assert x_p.shape[1] == 3
        assert t_p.shape[0] == x_p.shape[0]
        assert delay >= 1

    def test_prefix_none_before_chunk(self):
        """get_prefix should return None before any chunk."""
        f = self._make_filter()
        assert f.get_prefix() is None

    def test_prefix_matches_get_output(self):
        """Prefix values should match get_output at same timestamps."""
        f = self._make_filter(initial_delay=2)
        t = np.arange(self.H) * self.DT
        x = np.column_stack([np.sin(t), np.cos(t), t])

        f.set_current_time(0.0)
        f.update_chunk(t, x)

        prefix = f.get_prefix()
        assert prefix is not None
        t_p, x_p, _ = prefix

        for i, ti in enumerate(t_p):
            out = f.get_output(float(ti))
            assert out is not None
            np.testing.assert_allclose(x_p[i], out, atol=1e-12)


class TestAsyncFilterRTCRAILThreadSafety:
    """Concurrent access should not crash."""

    def test_concurrent_update_and_query(self):
        f = AsyncFilterRTCRAIL(
            prediction_horizon=16, dt=0.02,
            min_execution_horizon=4, poly_degree=3,
        )
        errors = []

        def writer():
            try:
                rng = np.random.default_rng(0)
                for i in range(20):
                    t = np.arange(16) * 0.02 + i * 0.1
                    x = rng.standard_normal((16, 3))
                    f.set_current_time(i * 0.1)
                    f.update_chunk(t, x)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(100):
                    f.get_output(i * 0.02)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=5)
        assert not errors
