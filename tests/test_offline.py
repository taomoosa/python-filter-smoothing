"""Tests for OfflineFilter."""
import numpy as np
import pytest
from python_filter_smoothing import OfflineFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sine(n: int = 100, dims: int = 1) -> tuple:
    t = np.linspace(0, 2 * np.pi, n)
    x = np.sin(t)
    if dims > 1:
        x = np.column_stack([np.sin(t) * (d + 1) for d in range(dims)])
    return t, x


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestOfflineFilterInit:
    def test_scalar_input_shape(self):
        t, x = make_sine()
        f = OfflineFilter(t, x)
        assert f.n_samples == 100
        assert f.n_dims == 1

    def test_vector_input_shape(self):
        t, x = make_sine(dims=3)
        f = OfflineFilter(t, x)
        assert f.n_dims == 3

    def test_sort_by_time(self):
        t = np.array([3.0, 1.0, 2.0])
        x = np.array([30.0, 10.0, 20.0])
        f = OfflineFilter(t, x)
        np.testing.assert_array_equal(f.t, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(f.x[:, 0], [10.0, 20.0, 30.0])

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            OfflineFilter(np.arange(5), np.arange(6))


# ---------------------------------------------------------------------------
# linear_interpolate
# ---------------------------------------------------------------------------

class TestLinearInterpolate:
    def test_at_known_points(self):
        t = np.array([0.0, 1.0, 2.0])
        x = np.array([0.0, 1.0, 2.0])
        f = OfflineFilter(t, x)
        np.testing.assert_allclose(f.linear_interpolate(t), x)

    def test_midpoints(self, rerun_log):
        t = np.array([0.0, 1.0, 2.0])
        x = np.array([0.0, 2.0, 4.0])
        f = OfflineFilter(t, x)
        t_q = np.array([0.5, 1.5])
        result = f.linear_interpolate(t_q)
        rerun_log("offline/linear/input", t, x)
        rerun_log("offline/linear/output", t_q, result)
        np.testing.assert_allclose(result, [1.0, 3.0])

    def test_vector_data(self):
        t = np.array([0.0, 1.0])
        x = np.array([[0.0, 0.0], [1.0, 2.0]])
        f = OfflineFilter(t, x)
        result = f.linear_interpolate([0.5])
        np.testing.assert_allclose(result, [[0.5, 1.0]])

    def test_output_shape_scalar(self):
        t, x = make_sine(n=50)
        f = OfflineFilter(t, x)
        result = f.linear_interpolate(t)
        assert result.shape == (50,)

    def test_output_shape_vector(self):
        t, x = make_sine(n=50, dims=3)
        f = OfflineFilter(t, x)
        result = f.linear_interpolate(t)
        assert result.shape == (50, 3)


# ---------------------------------------------------------------------------
# lowpass_filter
# ---------------------------------------------------------------------------

class TestLowpassFilter:
    def test_reduces_noise(self, rerun_log):
        t, x_clean = make_sine(n=200)
        rng = np.random.RandomState(0)
        x_noisy = x_clean + rng.randn(200) * 0.5
        f = OfflineFilter(t, x_noisy)
        sr = 200 / (2 * np.pi)
        x_filtered = f.lowpass_filter(cutoff_freq=0.3 * sr, sample_rate=sr)
        rerun_log("offline/lowpass/clean", t, x_clean)
        rerun_log("offline/lowpass/noisy", t, x_noisy)
        rerun_log("offline/lowpass/filtered", t, x_filtered)
        err_noisy = np.mean((x_noisy - x_clean) ** 2)
        err_filtered = np.mean((x_filtered - x_clean) ** 2)
        assert err_filtered < err_noisy

    def test_invalid_cutoff_raises(self):
        t, x = make_sine()
        f = OfflineFilter(t, x)
        with pytest.raises(ValueError):
            f.lowpass_filter(cutoff_freq=100.0, sample_rate=1.0)

    def test_output_shape(self):
        t, x = make_sine(n=200)
        f = OfflineFilter(t, x)
        sr = 200 / (2 * np.pi)
        result = f.lowpass_filter(cutoff_freq=0.4 * sr, sample_rate=sr)
        assert result.shape == (200,)


# ---------------------------------------------------------------------------
# polynomial_fit
# ---------------------------------------------------------------------------

class TestPolynomialFit:
    def test_perfect_linear(self, rerun_log):
        t = np.linspace(0, 1, 10)
        x = 2.0 * t + 1.0
        f = OfflineFilter(t, x)
        result = f.polynomial_fit(degree=1)
        rerun_log("offline/polyfit/input", t, x)
        rerun_log("offline/polyfit/output", t, result)
        np.testing.assert_allclose(result, x, atol=1e-10)

    def test_query_times(self, rerun_log):
        t = np.linspace(0, 1, 50)
        x = t ** 2
        f = OfflineFilter(t, x)
        t_q = np.array([0.0, 0.5, 1.0])
        result = f.polynomial_fit(degree=2, t_query=t_q)
        rerun_log("offline/polyfit_q/input", t, x)
        rerun_log("offline/polyfit_q/output", t_q, result)
        np.testing.assert_allclose(result, t_q ** 2, atol=1e-8)

    def test_default_query_is_original_t(self):
        t, x = make_sine(n=50)
        f = OfflineFilter(t, x)
        result = f.polynomial_fit(degree=5)
        assert result.shape == (50,)


# ---------------------------------------------------------------------------
# spline_interpolate
# ---------------------------------------------------------------------------

class TestSplineInterpolate:
    def test_at_known_points(self):
        t = np.linspace(0, 2 * np.pi, 20)
        x = np.sin(t)
        f = OfflineFilter(t, x)
        result = f.spline_interpolate(t)
        np.testing.assert_allclose(result, x, atol=1e-10)

    def test_cubic_smooth(self, rerun_log):
        t = np.linspace(0, 2 * np.pi, 20)
        x = np.sin(t)
        f = OfflineFilter(t, x)
        t_dense = np.linspace(0, 2 * np.pi, 100)
        result = f.spline_interpolate(t_dense)
        rerun_log("offline/spline/input", t, x)
        rerun_log("offline/spline/output", t_dense, result)
        rerun_log("offline/spline/truth", t_dense, np.sin(t_dense))
        np.testing.assert_allclose(result, np.sin(t_dense), atol=0.01)

    def test_linear_kind(self):
        t = np.array([0.0, 1.0, 2.0])
        x = np.array([0.0, 1.0, 2.0])
        f = OfflineFilter(t, x)
        result = f.spline_interpolate([0.5, 1.5], kind="linear")
        np.testing.assert_allclose(result, [0.5, 1.5])


# ---------------------------------------------------------------------------
# savgol_filter
# ---------------------------------------------------------------------------

class TestSavgolFilter:
    def test_preserves_polynomial(self):
        t = np.linspace(0, 5, 50)
        x = t ** 2
        f = OfflineFilter(t, x)
        result = f.savgol_filter(window_length=11, polyorder=2)
        np.testing.assert_allclose(result, x, atol=0.01)

    def test_reduces_noise(self, rerun_log):
        t, x_clean = make_sine(n=200)
        rng = np.random.RandomState(42)
        x_noisy = x_clean + rng.randn(200) * 0.5
        f = OfflineFilter(t, x_noisy)
        x_filtered = f.savgol_filter(window_length=11, polyorder=3)
        rerun_log("offline/savgol/clean", t, x_clean)
        rerun_log("offline/savgol/noisy", t, x_noisy)
        rerun_log("offline/savgol/filtered", t, x_filtered)
        err_noisy = np.mean((x_noisy - x_clean) ** 2)
        err_filtered = np.mean((x_filtered - x_clean) ** 2)
        assert err_filtered < err_noisy

    def test_output_shape_scalar(self):
        t, x = make_sine(n=100)
        f = OfflineFilter(t, x)
        result = f.savgol_filter(window_length=11, polyorder=3)
        assert result.shape == (100,)

    def test_output_shape_vector(self):
        t, x = make_sine(n=100, dims=3)
        f = OfflineFilter(t, x)
        result = f.savgol_filter(window_length=11, polyorder=3)
        assert result.shape == (100, 3)

    def test_custom_query_times(self):
        t, x = make_sine(n=100)
        f = OfflineFilter(t, x)
        x_smooth = f.savgol_filter(window_length=11, polyorder=3)
        f2 = OfflineFilter(t, x_smooth)
        t_q = np.linspace(t[0], t[-1], 5)
        result = f2.linear_interpolate(t_q)
        assert result.shape == (5,)

    def test_small_dataset(self):
        t = np.linspace(0, 1, 5)
        x = t ** 2
        f = OfflineFilter(t, x)
        result = f.savgol_filter(window_length=5, polyorder=2)
        np.testing.assert_allclose(result, x, atol=0.01)


# ---------------------------------------------------------------------------
# gaussian_filter
# ---------------------------------------------------------------------------

class TestGaussianFilter:
    def test_smooths_step_function(self):
        n = 100
        t = np.linspace(0, 1, n)
        x = np.zeros(n)
        x[n // 2 :] = 1.0
        f = OfflineFilter(t, x)
        result = f.gaussian_filter(sigma=3)
        mid = n // 2
        assert 0.1 < result[mid] < 0.9

    def test_reduces_noise(self, rerun_log):
        t, x_clean = make_sine(n=200)
        rng = np.random.RandomState(42)
        x_noisy = x_clean + rng.randn(200) * 0.5
        f = OfflineFilter(t, x_noisy)
        x_filtered = f.gaussian_filter(sigma=3)
        rerun_log("offline/gaussian/clean", t, x_clean)
        rerun_log("offline/gaussian/noisy", t, x_noisy)
        rerun_log("offline/gaussian/filtered", t, x_filtered)
        err_noisy = np.mean((x_noisy - x_clean) ** 2)
        err_filtered = np.mean((x_filtered - x_clean) ** 2)
        assert err_filtered < err_noisy

    def test_output_shape_scalar(self):
        t, x = make_sine(n=100)
        f = OfflineFilter(t, x)
        result = f.gaussian_filter(sigma=2)
        assert result.shape == (100,)

    def test_output_shape_vector(self):
        t, x = make_sine(n=100, dims=3)
        f = OfflineFilter(t, x)
        result = f.gaussian_filter(sigma=2)
        assert result.shape == (100, 3)


# ---------------------------------------------------------------------------
# median_filter
# ---------------------------------------------------------------------------

class TestMedianFilter:
    def test_removes_spike(self, rerun_log):
        t = np.linspace(0, 1, 50)
        x_clean = np.sin(2 * np.pi * t)
        x_spiked = x_clean.copy()
        spike_idx = 25
        x_spiked[spike_idx] = 100.0
        f = OfflineFilter(t, x_spiked)
        x_filtered = f.median_filter(kernel_size=5)
        rerun_log("offline/median/clean", t, x_clean)
        rerun_log("offline/median/spiked", t, x_spiked)
        rerun_log("offline/median/filtered", t, x_filtered)
        np.testing.assert_allclose(
            x_filtered[spike_idx], x_clean[spike_idx], atol=0.3
        )

    def test_output_shape_scalar(self):
        t, x = make_sine(n=100)
        f = OfflineFilter(t, x)
        result = f.median_filter(kernel_size=5)
        assert result.shape == (100,)

    def test_output_shape_vector(self):
        t, x = make_sine(n=100, dims=3)
        f = OfflineFilter(t, x)
        result = f.median_filter(kernel_size=5)
        assert result.shape == (100, 3)


# ---------------------------------------------------------------------------
# moving_average
# ---------------------------------------------------------------------------

class TestMovingAverage:
    def test_constant_signal_unchanged(self):
        t = np.linspace(0, 1, 50)
        x = np.full(50, 5.0)
        f = OfflineFilter(t, x)
        result = f.moving_average(window_size=7)
        # Interior samples (away from edge effects) should be exact
        np.testing.assert_allclose(result[3:-3], 5.0, atol=1e-10)

    def test_reduces_noise(self, rerun_log):
        t, x_clean = make_sine(n=200)
        rng = np.random.RandomState(42)
        x_noisy = x_clean + rng.randn(200) * 0.5
        f = OfflineFilter(t, x_noisy)
        x_filtered = f.moving_average(window_size=11)
        rerun_log("offline/mavg/clean", t, x_clean)
        rerun_log("offline/mavg/noisy", t, x_noisy)
        rerun_log("offline/mavg/filtered", t, x_filtered)
        err_noisy = np.mean((x_noisy - x_clean) ** 2)
        err_filtered = np.mean((x_filtered - x_clean) ** 2)
        assert err_filtered < err_noisy

    def test_output_shape_scalar(self):
        t, x = make_sine(n=100)
        f = OfflineFilter(t, x)
        result = f.moving_average(window_size=5)
        assert result.shape == (100,)

    def test_output_shape_vector(self):
        t, x = make_sine(n=100, dims=3)
        f = OfflineFilter(t, x)
        result = f.moving_average(window_size=5)
        assert result.shape == (100, 3)


# ---------------------------------------------------------------------------
# fir_filter
# ---------------------------------------------------------------------------

class TestFIRFilter:
    def test_reduces_noise(self, rerun_log):
        t, x_clean = make_sine(n=200)
        rng = np.random.RandomState(42)
        x_noisy = x_clean + rng.randn(200) * 0.5
        f = OfflineFilter(t, x_noisy)
        sample_rate = 200 / (2 * np.pi)
        x_filtered = f.fir_filter(numtaps=31, cutoff_freq=3.0, sample_rate=sample_rate)
        rerun_log("offline/fir/clean", t, x_clean)
        rerun_log("offline/fir/noisy", t, x_noisy)
        rerun_log("offline/fir/filtered", t, x_filtered)
        err_noisy = np.mean((x_noisy - x_clean) ** 2)
        err_filtered = np.mean((x_filtered - x_clean) ** 2)
        assert err_filtered < err_noisy

    def test_output_shape_scalar(self):
        t, x = make_sine(n=100)
        f = OfflineFilter(t, x)
        result = f.fir_filter(numtaps=15, cutoff_freq=5.0, sample_rate=100.0)
        assert result.shape == (100,)

    def test_output_shape_vector(self):
        t, x = make_sine(n=100, dims=3)
        f = OfflineFilter(t, x)
        result = f.fir_filter(numtaps=15, cutoff_freq=5.0, sample_rate=100.0)
        assert result.shape == (100, 3)

    def test_highpass(self):
        t = np.linspace(0, 1, 200)
        x_low = np.sin(2 * np.pi * 2 * t)  # 2 Hz
        x_high = np.sin(2 * np.pi * 40 * t)  # 40 Hz
        x = x_low + x_high
        f = OfflineFilter(t, x)
        result = f.fir_filter(
            numtaps=51, cutoff_freq=20.0, sample_rate=200.0, pass_zero=False,
        )
        # Should attenuate the 2 Hz and keep the 40 Hz
        # Check middle section (avoid edge effects)
        mid = slice(50, 150)
        corr_high = np.corrcoef(result[mid], x_high[mid])[0, 1]
        assert abs(corr_high) > 0.5


# ---------------------------------------------------------------------------
# iir_filter
# ---------------------------------------------------------------------------

class TestIIRFilter:
    def test_butterworth_reduces_noise(self, rerun_log):
        t, x_clean = make_sine(n=200)
        rng = np.random.RandomState(42)
        x_noisy = x_clean + rng.randn(200) * 0.5
        f = OfflineFilter(t, x_noisy)
        sr = 200 / (2 * np.pi)
        x_filt = f.iir_filter(cutoff_freq=3.0, sample_rate=sr, iir_type="butterworth")
        rerun_log("offline/iir_butter/clean", t, x_clean)
        rerun_log("offline/iir_butter/noisy", t, x_noisy)
        rerun_log("offline/iir_butter/filtered", t, x_filt)
        assert np.mean((x_filt - x_clean) ** 2) < np.mean((x_noisy - x_clean) ** 2)

    def test_chebyshev1(self):
        t, x = make_sine(n=200)
        rng = np.random.RandomState(42)
        x_noisy = x + rng.randn(200) * 0.3
        f = OfflineFilter(t, x_noisy)
        sr = 200 / (2 * np.pi)
        result = f.iir_filter(
            cutoff_freq=3.0, sample_rate=sr, iir_type="chebyshev1", rp=1.0,
        )
        assert result.shape == (200,)
        assert np.mean((result - x) ** 2) < np.mean((x_noisy - x) ** 2)

    def test_chebyshev1_requires_rp(self):
        t, x = make_sine(n=100)
        f = OfflineFilter(t, x)
        with pytest.raises(ValueError, match="rp"):
            f.iir_filter(cutoff_freq=5.0, sample_rate=100.0, iir_type="chebyshev1")

    def test_chebyshev2(self):
        t, x = make_sine(n=200)
        rng = np.random.RandomState(42)
        x_noisy = x + rng.randn(200) * 0.3
        f = OfflineFilter(t, x_noisy)
        sr = 200 / (2 * np.pi)
        result = f.iir_filter(
            cutoff_freq=3.0, sample_rate=sr, iir_type="chebyshev2", rs=40.0,
        )
        assert result.shape == (200,)

    def test_chebyshev2_requires_rs(self):
        t, x = make_sine(n=100)
        f = OfflineFilter(t, x)
        with pytest.raises(ValueError, match="rs"):
            f.iir_filter(cutoff_freq=5.0, sample_rate=100.0, iir_type="chebyshev2")

    def test_elliptic(self):
        t, x = make_sine(n=200)
        rng = np.random.RandomState(42)
        x_noisy = x + rng.randn(200) * 0.3
        f = OfflineFilter(t, x_noisy)
        sr = 200 / (2 * np.pi)
        result = f.iir_filter(
            cutoff_freq=3.0, sample_rate=sr,
            iir_type="elliptic", rp=1.0, rs=40.0,
        )
        assert result.shape == (200,)

    def test_elliptic_requires_both(self):
        t, x = make_sine(n=100)
        f = OfflineFilter(t, x)
        with pytest.raises(ValueError, match="rp.*rs"):
            f.iir_filter(cutoff_freq=5.0, sample_rate=100.0, iir_type="elliptic")

    def test_bessel(self):
        t, x = make_sine(n=200)
        rng = np.random.RandomState(42)
        x_noisy = x + rng.randn(200) * 0.3
        f = OfflineFilter(t, x_noisy)
        sr = 200 / (2 * np.pi)
        result = f.iir_filter(cutoff_freq=3.0, sample_rate=sr, iir_type="bessel")
        assert result.shape == (200,)

    def test_bandpass(self):
        t = np.linspace(0, 1, 500)
        x = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 100 * t)
        f = OfflineFilter(t, x)
        result = f.iir_filter(
            cutoff_freq=[5.0, 20.0], sample_rate=500.0, btype="bandpass",
        )
        assert result.shape == (500,)

    def test_unknown_type_raises(self):
        t, x = make_sine(n=100)
        f = OfflineFilter(t, x)
        with pytest.raises(ValueError, match="Unknown iir_type"):
            f.iir_filter(cutoff_freq=5.0, sample_rate=100.0, iir_type="unknown")

    def test_vector_output(self):
        t, x = make_sine(n=200, dims=3)
        f = OfflineFilter(t, x)
        result = f.iir_filter(cutoff_freq=5.0, sample_rate=200 / (2 * np.pi))
        assert result.shape == (200, 3)


# ---------------------------------------------------------------------------
# kalman_smooth
# ---------------------------------------------------------------------------

class TestKalmanSmooth:
    def test_position_model_reduces_noise(self, rerun_log):
        t, x_clean = make_sine(n=200)
        rng = np.random.RandomState(42)
        x_noisy = x_clean + rng.randn(200) * 0.5
        f = OfflineFilter(t, x_noisy)
        x_smooth = f.kalman_smooth(process_noise=0.01, measurement_noise=0.25)
        rerun_log("offline/kalman/clean", t, x_clean)
        rerun_log("offline/kalman/noisy", t, x_noisy)
        rerun_log("offline/kalman/smoothed", t, x_smooth)
        err_noisy = np.mean((x_noisy - x_clean) ** 2)
        err_smooth = np.mean((x_smooth - x_clean) ** 2)
        assert err_smooth < err_noisy

    def test_position_velocity_model(self):
        t = np.linspace(0, 2 * np.pi, 200)
        x_clean = np.sin(t)
        rng = np.random.RandomState(42)
        x_noisy = x_clean + rng.randn(200) * 0.3
        f = OfflineFilter(t, x_noisy)
        x_smooth = f.kalman_smooth(
            state_model="position_velocity",
            process_noise=1.0,
            measurement_noise=0.09,
        )
        assert x_smooth.shape == (200,)
        assert np.mean((x_smooth - x_clean) ** 2) < np.mean((x_noisy - x_clean) ** 2)

    def test_vector_output(self):
        t, x = make_sine(n=100, dims=3)
        rng = np.random.RandomState(42)
        x_noisy = x + rng.randn(100, 3) * 0.3
        f = OfflineFilter(t, x_noisy)
        result = f.kalman_smooth()
        assert result.shape == (100, 3)

    def test_custom_matrices(self):
        t = np.linspace(0, 1, 50)
        x = np.column_stack([t, t ** 2])
        rng = np.random.RandomState(42)
        x_noisy = x + rng.randn(50, 2) * 0.1
        f = OfflineFilter(t, x_noisy)
        F = np.eye(2)
        H = np.eye(2)
        Q = np.eye(2) * 0.01
        R = np.eye(2) * 0.01
        result = f.kalman_smooth(F=F, H=H, Q=Q, R=R)
        assert result.shape == (50, 2)

    def test_unknown_model_raises(self):
        t, x = make_sine(n=50)
        f = OfflineFilter(t, x)
        with pytest.raises(ValueError, match="Unknown state_model"):
            f.kalman_smooth(state_model="unknown")
