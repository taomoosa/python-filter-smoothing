"""Tests for OnlineFilter."""
import numpy as np
import pytest
from python_filter_smoothing import OnlineFilter


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class TestOnlineFilterEMA:
    def test_converges_to_constant(self, rerun_log_scalar):
        f = OnlineFilter(method="ema", alpha=0.5)
        val = None
        for i in range(60):
            val = f.update(float(i), 1.0)
            rerun_log_scalar("online/ema/converge/input", float(i), 1.0)
            rerun_log_scalar("online/ema/converge/output", float(i), val)
        np.testing.assert_allclose(val.item(), 1.0, atol=0.01)

    def test_high_alpha_tracks_faster(self, rerun_log_scalar):
        f_fast = OnlineFilter(method="ema", alpha=0.9)
        f_slow = OnlineFilter(method="ema", alpha=0.1)
        f_fast.update(0.0, 0.0)
        f_slow.update(0.0, 0.0)
        rerun_log_scalar("online/ema/alpha_cmp/input", 0.0, 0.0)
        for i in range(1, 16):
            v_fast = f_fast.update(float(i), 1.0)
            v_slow = f_slow.update(float(i), 1.0)
            rerun_log_scalar("online/ema/alpha_cmp/input", float(i), 1.0)
            rerun_log_scalar("online/ema/alpha_cmp/fast_alpha09", float(i), v_fast)
            rerun_log_scalar("online/ema/alpha_cmp/slow_alpha01", float(i), v_slow)
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
    def test_full_window_average(self, rerun_log_scalar):
        f = OnlineFilter(method="moving_average", window=3)
        inputs = [1.0, 2.0, 3.0]
        for i, xi in enumerate(inputs):
            val = f.update(i, xi)
            rerun_log_scalar("online/mavg/input", float(i), xi)
            rerun_log_scalar("online/mavg/output", float(i), val)
        np.testing.assert_allclose(val.item(), 2.0)

    def test_window_slides_out_old_values(self):
        f = OnlineFilter(method="moving_average", window=2)
        f.update(0, 0.0)
        f.update(1, 0.0)
        val = f.update(2, 10.0)
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
    def test_passes_dc(self, rerun_log_scalar):
        """Low-pass filter should pass DC (zero-frequency) signals."""
        f = OnlineFilter(method="lowpass", cutoff_freq=0.1, sample_rate=1.0)
        for i in range(100):
            val = f.update(i, 1.0)
            rerun_log_scalar("online/lowpass/dc/input", float(i), 1.0)
            rerun_log_scalar("online/lowpass/dc/output", float(i), val)
        np.testing.assert_allclose(val.item(), 1.0, atol=0.05)

    def test_reduces_noise_variance(self, rerun_log_scalar):
        rng = np.random.RandomState(0)
        f = OnlineFilter(method="lowpass", cutoff_freq=0.05, sample_rate=1.0)
        noise_vals, filtered_vals = [], []
        for i in range(200):
            noisy = 1.0 + rng.randn() * 0.5
            filt_val = f.update(i, noisy)
            noise_vals.append(noisy)
            filtered_vals.append(filt_val.item())
            rerun_log_scalar("online/lowpass/noise/input", float(i), noisy)
            rerun_log_scalar("online/lowpass/noise/output", float(i), filt_val)
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


# ---------------------------------------------------------------------------
# One Euro filter
# ---------------------------------------------------------------------------

class TestOnlineFilterOneEuro:
    def test_converges_to_constant(self, rerun_log_scalar):
        f = OnlineFilter(method="one_euro", min_cutoff=1.0, beta=0.0)
        val = None
        for i in range(100):
            val = f.update(float(i), 1.0)
            rerun_log_scalar("online/one_euro/converge/input", float(i), 1.0)
            rerun_log_scalar("online/one_euro/converge/output", float(i), val)
        np.testing.assert_allclose(val.item(), 1.0, atol=0.01)

    def test_tracks_slow_signal(self, rerun_log_scalar):
        f = OnlineFilter(method="one_euro", min_cutoff=1.0, beta=0.007)
        n = 200
        outputs = []
        inputs = []
        for i in range(n):
            t = float(i)
            x = np.sin(2 * np.pi * t / 100.0)
            val = f.update(t, x)
            inputs.append(x)
            outputs.append(val.item())
            rerun_log_scalar("online/one_euro/slow/input", t, x)
            rerun_log_scalar("online/one_euro/slow/output", t, val)
        corr = np.corrcoef(inputs[10:], outputs[10:])[0, 1]
        assert corr > 0.95

    def test_smooths_noise(self, rerun_log_scalar):
        rng = np.random.RandomState(42)
        f = OnlineFilter(method="one_euro", min_cutoff=1.0, beta=0.0)
        input_vals, output_vals = [], []
        for i in range(200):
            x = 1.0 + rng.randn() * 0.3
            val = f.update(float(i), x)
            input_vals.append(x)
            output_vals.append(val.item())
            rerun_log_scalar("online/one_euro/noise/input", float(i), x)
            rerun_log_scalar("online/one_euro/noise/output", float(i), val)
        assert np.var(output_vals[20:]) < np.var(input_vals[20:])

    def test_vector_input(self):
        f = OnlineFilter(method="one_euro", min_cutoff=1.0, beta=0.0)
        for i in range(10):
            val = f.update(float(i), [1.0, 2.0, 3.0])
        assert val.shape == (3,)

    def test_reset_clears_state(self):
        f = OnlineFilter(method="one_euro", min_cutoff=1.0, beta=0.0)
        f.update(0.0, 1.0)
        f.reset()
        assert f.get_value() is None

    def test_high_beta_fast_tracking(self):
        f_low = OnlineFilter(method="one_euro", min_cutoff=1.0, beta=0.0)
        f_high = OnlineFilter(method="one_euro", min_cutoff=1.0, beta=1.0)
        # Warm up at 0.0
        for i in range(20):
            f_low.update(float(i), 0.0)
            f_high.update(float(i), 0.0)
        # Step to 1.0
        for i in range(20, 30):
            v_low = f_low.update(float(i), 1.0)
            v_high = f_high.update(float(i), 1.0)
        assert abs(v_high.item() - 1.0) < abs(v_low.item() - 1.0)


# ---------------------------------------------------------------------------
# FIR
# ---------------------------------------------------------------------------

class TestOnlineFilterFIR:
    def test_passes_dc(self, rerun_log_scalar):
        f = OnlineFilter(
            method="fir", numtaps=31, cutoff_freq=5.0, sample_rate=100.0,
        )
        for i in range(100):
            val = f.update(float(i), 1.0)
            rerun_log_scalar("online/fir/dc/input", float(i), 1.0)
            rerun_log_scalar("online/fir/dc/output", float(i), val)
        np.testing.assert_allclose(val.item(), 1.0, atol=0.05)

    def test_reduces_noise(self, rerun_log_scalar):
        rng = np.random.RandomState(42)
        f = OnlineFilter(
            method="fir", numtaps=31, cutoff_freq=5.0, sample_rate=100.0,
        )
        input_vals, output_vals = [], []
        for i in range(200):
            x = 1.0 + rng.randn() * 0.5
            val = f.update(float(i) / 100.0, x)
            input_vals.append(x)
            output_vals.append(val.item())
            rerun_log_scalar("online/fir/noise/input", float(i), x)
            rerun_log_scalar("online/fir/noise/output", float(i), val)
        assert np.var(output_vals[50:]) < np.var(input_vals[50:])

    def test_vector_input(self):
        f = OnlineFilter(
            method="fir", numtaps=15, cutoff_freq=5.0, sample_rate=100.0,
        )
        for i in range(20):
            val = f.update(float(i), [1.0, 2.0, 3.0])
        assert val.shape == (3,)

    def test_reset(self):
        f = OnlineFilter(
            method="fir", numtaps=15, cutoff_freq=5.0, sample_rate=100.0,
        )
        f.update(0.0, 1.0)
        f.reset()
        assert f.get_value() is None


# ---------------------------------------------------------------------------
# IIR (general)
# ---------------------------------------------------------------------------

class TestOnlineFilterIIR:
    def test_butterworth_passes_dc(self, rerun_log_scalar):
        f = OnlineFilter(
            method="iir", cutoff_freq=5.0, sample_rate=100.0,
            iir_type="butterworth",
        )
        for i in range(200):
            val = f.update(float(i), 1.0)
            rerun_log_scalar("online/iir/dc/input", float(i), 1.0)
            rerun_log_scalar("online/iir/dc/output", float(i), val)
        np.testing.assert_allclose(val.item(), 1.0, atol=0.05)

    def test_chebyshev1(self):
        f = OnlineFilter(
            method="iir", cutoff_freq=5.0, sample_rate=100.0,
            iir_type="chebyshev1", rp=1.0,
        )
        for i in range(50):
            val = f.update(float(i), 1.0)
        assert val.shape == (1,)

    def test_chebyshev2(self):
        f = OnlineFilter(
            method="iir", cutoff_freq=5.0, sample_rate=100.0,
            iir_type="chebyshev2", rs=40.0,
        )
        for i in range(50):
            val = f.update(float(i), 1.0)
        assert val.shape == (1,)

    def test_elliptic(self):
        f = OnlineFilter(
            method="iir", cutoff_freq=5.0, sample_rate=100.0,
            iir_type="elliptic", rp=1.0, rs=40.0,
        )
        for i in range(50):
            val = f.update(float(i), 1.0)
        assert val.shape == (1,)

    def test_bessel(self):
        f = OnlineFilter(
            method="iir", cutoff_freq=5.0, sample_rate=100.0,
            iir_type="bessel",
        )
        for i in range(50):
            val = f.update(float(i), 1.0)
        assert val.shape == (1,)

    def test_reduces_noise(self, rerun_log_scalar):
        rng = np.random.RandomState(42)
        f = OnlineFilter(
            method="iir", cutoff_freq=5.0, sample_rate=100.0,
            iir_type="butterworth", order=4,
        )
        input_vals, output_vals = [], []
        for i in range(200):
            x = 1.0 + rng.randn() * 0.5
            val = f.update(float(i) / 100.0, x)
            input_vals.append(x)
            output_vals.append(val.item())
            rerun_log_scalar("online/iir/noise/input", float(i), x)
            rerun_log_scalar("online/iir/noise/output", float(i), val)
        assert np.var(output_vals[50:]) < np.var(input_vals[50:])

    def test_vector_input(self):
        f = OnlineFilter(
            method="iir", cutoff_freq=5.0, sample_rate=100.0,
        )
        for i in range(20):
            val = f.update(float(i), [1.0, 2.0, 3.0])
        assert val.shape == (3,)

    def test_reset(self):
        f = OnlineFilter(
            method="iir", cutoff_freq=5.0, sample_rate=100.0,
        )
        f.update(0.0, 1.0)
        f.reset()
        assert f.get_value() is None


# ---------------------------------------------------------------------------
# Kalman
# ---------------------------------------------------------------------------

class TestOnlineFilterKalman:
    def test_converges_to_constant(self, rerun_log_scalar):
        f = OnlineFilter(
            method="kalman", process_noise=0.01, measurement_noise=0.1,
        )
        for i in range(100):
            val = f.update(float(i), 1.0)
            rerun_log_scalar("online/kalman/const/input", float(i), 1.0)
            rerun_log_scalar("online/kalman/const/output", float(i), val)
        np.testing.assert_allclose(val.item(), 1.0, atol=0.05)

    def test_reduces_noise(self, rerun_log_scalar):
        rng = np.random.RandomState(42)
        f = OnlineFilter(
            method="kalman", process_noise=0.01, measurement_noise=0.25,
        )
        input_vals, output_vals = [], []
        for i in range(200):
            x = np.sin(float(i) / 20.0) + rng.randn() * 0.5
            val = f.update(float(i), x)
            input_vals.append(x)
            output_vals.append(val.item())
            rerun_log_scalar("online/kalman/noise/input", float(i), x)
            rerun_log_scalar("online/kalman/noise/output", float(i), val)
        assert np.var(output_vals[20:]) < np.var(input_vals[20:])

    def test_position_velocity_model(self):
        f = OnlineFilter(
            method="kalman", state_model="position_velocity",
            process_noise=0.1, measurement_noise=0.1, dt=0.01,
        )
        for i in range(50):
            val = f.update(float(i) * 0.01, float(i) * 0.01)
        assert val.shape == (1,)

    def test_vector_input(self):
        f = OnlineFilter(
            method="kalman", process_noise=0.01, measurement_noise=0.1,
        )
        for i in range(20):
            val = f.update(float(i), [1.0, 2.0, 3.0])
        assert val.shape == (3,)

    def test_custom_matrices(self):
        F = np.eye(2)
        H = np.eye(2)
        Q = np.eye(2) * 0.01
        R = np.eye(2) * 0.1
        f = OnlineFilter(method="kalman", F=F, H=H, Q=Q, R=R)
        for i in range(20):
            val = f.update(float(i), [1.0, 2.0])
        assert val.shape == (2,)

    def test_reset(self):
        f = OnlineFilter(
            method="kalman", process_noise=0.01, measurement_noise=0.1,
        )
        f.update(0.0, 1.0)
        f.reset()
        assert f.get_value() is None
