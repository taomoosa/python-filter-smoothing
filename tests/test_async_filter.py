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

    def test_converges_to_constant(self):
        f = AsyncFilter(method="ema", alpha=0.5)
        for i in range(60):
            f.update(float(i), 1.0)
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

    def test_spline_sine(self):
        f = AsyncFilter(method="spline", buffer_size=50)
        t = np.linspace(0, 2 * np.pi, 30)
        x = np.sin(t)
        for ti, xi in zip(t, x):
            f.update(float(ti), float(xi))
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
        # t=None → evaluate at latest t=2.0 → value 4.0
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
