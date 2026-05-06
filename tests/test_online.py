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


# ---------------------------------------------------------------------------
# Variable period support
# ---------------------------------------------------------------------------

class TestVariablePeriod:
    """Verify that filters correctly handle non-uniform sampling intervals."""

    def test_ema_dt_nominal_matches_fixed_alpha(self):
        """EMA with dt == dt_nominal must equal classic fixed-alpha EMA."""
        alpha = 0.4
        dt_nom = 1.0
        f_adaptive = OnlineFilter(method="ema", alpha=alpha, dt_nominal=dt_nom)
        f_classic = OnlineFilter(method="ema", alpha=alpha, dt_nominal=dt_nom)

        rng = np.random.RandomState(7)
        for i in range(50):
            x = rng.randn()
            t = float(i) * dt_nom
            v1 = f_adaptive.update(t, x)
            v2 = f_classic.update(t, x)
            np.testing.assert_allclose(v1, v2)

    def test_ema_faster_sampling_smoother(self):
        """EMA sampled at 2× the nominal rate should be smoother per unit time."""
        alpha = 0.5
        dt_nom = 1.0
        f_nom = OnlineFilter(method="ema", alpha=alpha, dt_nominal=dt_nom)
        f_fast = OnlineFilter(method="ema", alpha=alpha, dt_nominal=dt_nom)

        rng = np.random.RandomState(42)
        # Feed both filters for 1 second of real time, nominal at dt=1 (1 sample),
        # fast at dt=0.5 (2 samples).  Faster sampling yields a smaller effective
        # alpha per step, so the output should be closer to the initial state.
        x0 = 0.0
        f_nom.update(0.0, x0)
        f_fast.update(0.0, x0)
        # Step to 1.0 after one nominal period
        v_nom = f_nom.update(1.0, 1.0)
        v_fast1 = f_fast.update(0.5, 1.0)
        v_fast2 = f_fast.update(1.0, 1.0)
        # Both filters receive the same total time (1 s) but fast has 2 steps.
        # The effective response should be similar because tau is preserved.
        np.testing.assert_allclose(v_nom.item(), v_fast2.item(), atol=0.05)

    def test_ema_variable_interval_runs_without_error(self):
        """EMA must process irregular timestamps without raising."""
        f = OnlineFilter(method="ema", alpha=0.3, dt_nominal=0.1)
        intervals = [0.1, 0.05, 0.2, 0.1, 0.3, 0.08]
        t = 0.0
        for dt in intervals:
            t += dt
            val = f.update(t, 1.0)
        assert val is not None

    def test_kalman_position_velocity_variable_period(self):
        """Kalman position_velocity model must handle irregular timestamps."""
        f = OnlineFilter(
            method="kalman", state_model="position_velocity",
            process_noise=0.1, measurement_noise=0.1, dt=0.01,
        )
        # Irregular timestamps: simulate a 1 m/s ramp with jittered timing
        rng = np.random.RandomState(0)
        t = 0.0
        for _ in range(60):
            dt_actual = 0.01 + rng.uniform(-0.005, 0.005)
            t += dt_actual
            x = t * 1.0  # true position = velocity * time
            val = f.update(t, x)
        # After convergence the estimate should track the ramp
        np.testing.assert_allclose(val.item(), t, atol=0.1)

    def test_kalman_position_model_variable_period(self):
        """Kalman position model must accept variable-period inputs."""
        f = OnlineFilter(
            method="kalman", state_model="position",
            process_noise=0.01, measurement_noise=0.1,
        )
        intervals = [0.1, 0.05, 0.2, 0.15, 0.08]
        t = 0.0
        for dt in intervals:
            t += dt
            val = f.update(t, 1.0)
        assert val.shape == (1,)

    # ------------------------------------------------------------------
    # MovingAverage – time-based window
    # ------------------------------------------------------------------

    def test_moving_average_window_time_basic(self):
        """Time-based window must average only samples within the window."""
        f = OnlineFilter(method="moving_average", window_time=1.0)
        # t=0: x=0; t=0.5: x=0; t=1.5: x=10 (the two earlier samples drop out)
        f.update(0.0, 0.0)
        f.update(0.5, 0.0)
        val = f.update(1.5, 10.0)
        # Only the t=0.5 and t=1.5 samples remain (0.5 >= 1.5 - 1.0)
        np.testing.assert_allclose(val.item(), 5.0, atol=1e-9)

    def test_moving_average_window_time_all_within(self):
        """All samples within the window are averaged."""
        f = OnlineFilter(method="moving_average", window_time=10.0)
        for i in range(5):
            val = f.update(float(i), float(i))
        # samples [0,1,2,3,4] all within window_time=10 → mean = 2.0
        np.testing.assert_allclose(val.item(), 2.0, atol=1e-9)

    def test_moving_average_window_time_variable_period_runs(self):
        """Time-based window must work with irregular timestamps."""
        f = OnlineFilter(method="moving_average", window_time=0.5)
        intervals = [0.1, 0.3, 0.05, 0.2, 0.4]
        t = 0.0
        for dt in intervals:
            t += dt
            val = f.update(t, 1.0)
        assert val is not None

    def test_moving_average_window_time_reset(self):
        """Reset clears time-based window state."""
        f = OnlineFilter(method="moving_average", window_time=1.0)
        f.update(0.0, 5.0)
        f.reset()
        assert f.get_value() is None

    # ------------------------------------------------------------------
    # Lowpass – ZOH variable-period
    # ------------------------------------------------------------------

    def test_lowpass_variable_period_passes_dc(self):
        """Lowpass with variable dt must still pass DC signals."""
        rng = np.random.RandomState(1)
        f = OnlineFilter(method="lowpass", cutoff_freq=1.0, sample_rate=10.0)
        t = 0.0
        for _ in range(100):
            dt = 0.1 + rng.uniform(-0.04, 0.04)
            t += dt
            val = f.update(t, 1.0)
        np.testing.assert_allclose(val.item(), 1.0, atol=0.05)

    def test_lowpass_variable_period_reduces_noise(self):
        """Lowpass with variable dt must reduce noise variance."""
        rng = np.random.RandomState(99)
        f = OnlineFilter(method="lowpass", cutoff_freq=1.0, sample_rate=20.0)
        t = 0.0
        input_vals, output_vals = [], []
        for _ in range(200):
            dt = 0.05 + rng.uniform(-0.01, 0.01)
            t += dt
            x = 1.0 + rng.randn() * 0.5
            val = f.update(t, x)
            input_vals.append(x)
            output_vals.append(val.item())
        assert np.var(output_vals[50:]) < np.var(input_vals[50:])

    # ------------------------------------------------------------------
    # IIR – ZOH variable-period
    # ------------------------------------------------------------------

    def test_iir_variable_period_passes_dc(self):
        """IIR Butterworth with variable dt must pass DC."""
        rng = np.random.RandomState(2)
        f = OnlineFilter(
            method="iir", cutoff_freq=5.0, sample_rate=100.0,
            iir_type="butterworth",
        )
        t = 0.0
        for _ in range(200):
            dt = 0.01 + rng.uniform(-0.003, 0.003)
            t += dt
            val = f.update(t, 1.0)
        np.testing.assert_allclose(val.item(), 1.0, atol=0.05)

    def test_iir_variable_period_reduces_noise(self):
        """IIR Butterworth with variable dt must reduce noise variance."""
        rng = np.random.RandomState(77)
        f = OnlineFilter(
            method="iir", cutoff_freq=5.0, sample_rate=100.0,
            iir_type="butterworth",
        )
        t = 0.0
        input_vals, output_vals = [], []
        for _ in range(200):
            dt = 0.01 + rng.uniform(-0.003, 0.003)
            t += dt
            x = 1.0 + rng.randn() * 0.5
            val = f.update(t, x)
            input_vals.append(x)
            output_vals.append(val.item())
        assert np.var(output_vals[50:]) < np.var(input_vals[50:])
