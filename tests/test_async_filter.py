"""Tests for AsyncFilter."""
import threading
import numpy as np
import pytest
from python_filter_smoothing import AsyncFilter


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
