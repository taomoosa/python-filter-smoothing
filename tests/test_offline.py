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

    def test_midpoints(self):
        t = np.array([0.0, 1.0, 2.0])
        x = np.array([0.0, 2.0, 4.0])
        f = OfflineFilter(t, x)
        np.testing.assert_allclose(f.linear_interpolate([0.5, 1.5]), [1.0, 3.0])

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
    def test_reduces_noise(self):
        t, x_clean = make_sine(n=200)
        rng = np.random.RandomState(0)
        x_noisy = x_clean + rng.randn(200) * 0.5
        f = OfflineFilter(t, x_noisy)
        sr = 200 / (2 * np.pi)
        x_filtered = f.lowpass_filter(cutoff_freq=0.3 * sr, sample_rate=sr)
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
    def test_perfect_linear(self):
        t = np.linspace(0, 1, 10)
        x = 2.0 * t + 1.0
        f = OfflineFilter(t, x)
        result = f.polynomial_fit(degree=1)
        np.testing.assert_allclose(result, x, atol=1e-10)

    def test_query_times(self):
        t = np.linspace(0, 1, 50)
        x = t ** 2
        f = OfflineFilter(t, x)
        t_q = np.array([0.0, 0.5, 1.0])
        result = f.polynomial_fit(degree=2, t_query=t_q)
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

    def test_cubic_smooth(self):
        t = np.linspace(0, 2 * np.pi, 20)
        x = np.sin(t)
        f = OfflineFilter(t, x)
        t_dense = np.linspace(0, 2 * np.pi, 100)
        result = f.spline_interpolate(t_dense)
        np.testing.assert_allclose(result, np.sin(t_dense), atol=0.01)

    def test_linear_kind(self):
        t = np.array([0.0, 1.0, 2.0])
        x = np.array([0.0, 1.0, 2.0])
        f = OfflineFilter(t, x)
        result = f.spline_interpolate([0.5, 1.5], kind="linear")
        np.testing.assert_allclose(result, [0.5, 1.5])
