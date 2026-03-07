"""Tests for OnlineFilter."""
import numpy as np
import pytest
from python_filter_smoothing import OnlineFilter


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class TestOnlineFilterEMA:
    def test_converges_to_constant(self):
        f = OnlineFilter(method="ema", alpha=0.5)
        val = None
        for i in range(60):
            val = f.update(float(i), 1.0)
        np.testing.assert_allclose(val.item(), 1.0, atol=0.01)

    def test_high_alpha_tracks_faster(self):
        # Start both filters from 0.0, then feed 1.0 repeatedly.
        # Higher alpha should reach 1.0 faster.
        f_fast = OnlineFilter(method="ema", alpha=0.9)
        f_slow = OnlineFilter(method="ema", alpha=0.1)
        # Seed both at 0.0
        f_fast.update(0.0, 0.0)
        f_slow.update(0.0, 0.0)
        for i in range(1, 16):
            v_fast = f_fast.update(float(i), 1.0)
            v_slow = f_slow.update(float(i), 1.0)
        # Fast filter should be closer to 1.0 after 15 more steps
        assert abs(v_fast.item() - 1.0) < abs(v_slow.item() - 1.0)

    def test_vector_input(self):
        f = OnlineFilter(method="ema", alpha=0.5)
        val = f.update(0.0, [1.0, 2.0, 3.0])
        assert val.shape == (3,)

    def test_get_value_none_before_update(self):
        f = OnlineFilter(method="ema")
        assert f.get_value() is None

    def test_get_value_after_update(self):
        f = OnlineFilter(method="ema")
        f.update(0.0, 1.0)
        assert f.get_value() is not None

    def test_reset_clears_state(self):
        f = OnlineFilter(method="ema", alpha=0.5)
        f.update(0.0, 5.0)
        f.reset()
        assert f.get_value() is None


# ---------------------------------------------------------------------------
# Moving average
# ---------------------------------------------------------------------------

class TestOnlineFilterMovingAverage:
    def test_full_window_average(self):
        f = OnlineFilter(method="moving_average", window=3)
        f.update(0, 1.0)
        f.update(1, 2.0)
        val = f.update(2, 3.0)
        np.testing.assert_allclose(val.item(), 2.0)

    def test_window_slides_out_old_values(self):
        f = OnlineFilter(method="moving_average", window=2)
        f.update(0, 0.0)
        f.update(1, 0.0)
        val = f.update(2, 10.0)
        # Buffer now holds [0.0, 10.0] → mean = 5.0
        np.testing.assert_allclose(val.item(), 5.0)

    def test_vector_input(self):
        f = OnlineFilter(method="moving_average", window=2)
        f.update(0, [0.0, 0.0])
        val = f.update(1, [2.0, 4.0])
        np.testing.assert_allclose(val, [1.0, 2.0])

    def test_reset(self):
        f = OnlineFilter(method="moving_average", window=3)
        f.update(0, 5.0)
        f.reset()
        assert f.get_value() is None


# ---------------------------------------------------------------------------
# IIR low-pass
# ---------------------------------------------------------------------------

class TestOnlineFilterLowpass:
    def test_passes_dc(self):
        """Low-pass filter should pass DC (zero-frequency) signals."""
        f = OnlineFilter(method="lowpass", cutoff_freq=0.1, sample_rate=1.0)
        vals = [f.update(i, 1.0) for i in range(100)]
        np.testing.assert_allclose(vals[-1].item(), 1.0, atol=0.05)

    def test_reduces_noise_variance(self):
        rng = np.random.RandomState(0)
        f = OnlineFilter(method="lowpass", cutoff_freq=0.05, sample_rate=1.0)
        noise_vals, filtered_vals = [], []
        for i in range(200):
            noisy = 1.0 + rng.randn() * 0.5
            noise_vals.append(noisy)
            filtered_vals.append(f.update(i, noisy).item())
        # Filter should reduce variance after the transient period
        assert np.var(filtered_vals[50:]) < np.var(noise_vals[50:])

    def test_vector_input(self):
        f = OnlineFilter(method="lowpass", cutoff_freq=0.1, sample_rate=1.0)
        for i in range(10):
            val = f.update(i, [1.0, 2.0])
        assert val.shape == (2,)


# ---------------------------------------------------------------------------
# Invalid method
# ---------------------------------------------------------------------------

class TestOnlineFilterInvalid:
    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            OnlineFilter(method="unknown_method")
