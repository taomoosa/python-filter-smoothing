"""Tests for ChunkFilter."""
import numpy as np
import pytest
from python_filter_smoothing import ChunkFilter


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------

class TestChunkFilterBasic:
    def test_single_chunk_no_timestamps(self):
        f = ChunkFilter()
        f.add_chunk([1.0, 2.0, 3.0])
        t_out, x_out = f.get_filtered()
        assert len(t_out) == 3

    def test_single_chunk_with_timestamps(self, rerun_log):
        f = ChunkFilter()
        f.add_chunk([1.0, 2.0, 3.0], t=[0.0, 0.5, 1.0])
        t_out, x_out = f.get_filtered()
        rerun_log("chunk/basic/input", t_out, x_out)
        np.testing.assert_allclose(t_out, [0.0, 0.5, 1.0])
        np.testing.assert_allclose(x_out, [1.0, 2.0, 3.0])

    def test_two_non_overlapping_chunks(self, rerun_log):
        f = ChunkFilter()
        f.add_chunk([1.0, 2.0], t=[0.0, 1.0])
        f.add_chunk([3.0, 4.0], t=[2.0, 3.0])
        t_out, x_out = f.get_filtered()
        rerun_log("chunk/two_chunks/output", t_out, x_out)
        assert len(t_out) == 4

    def test_no_data_raises(self):
        f = ChunkFilter()
        with pytest.raises(RuntimeError):
            f.get_filtered()

    def test_mismatched_t_x_raises(self):
        f = ChunkFilter()
        with pytest.raises(ValueError):
            f.add_chunk([1.0, 2.0, 3.0], t=[0.0, 1.0])

    def test_auto_timestamps_dt(self):
        f = ChunkFilter()
        f.add_chunk([1.0, 2.0, 3.0], dt=0.5)
        t_out, _ = f.get_filtered()
        np.testing.assert_allclose(t_out, [0.0, 0.5, 1.0])

    def test_second_auto_chunk_follows_first(self):
        f = ChunkFilter()
        f.add_chunk([1.0, 2.0], dt=1.0)   # t = [0.0, 1.0]
        f.add_chunk([3.0, 4.0], dt=1.0)   # t should start at 2.0
        t_out, _ = f.get_filtered()
        assert t_out[2] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Overlap strategies
# ---------------------------------------------------------------------------

class TestChunkFilterOverlap:
    def test_latest_strategy_keeps_newest(self, rerun_log):
        f = ChunkFilter(overlap_strategy="latest")
        f.add_chunk([1.0, 2.0], t=[0.0, 1.0])
        f.add_chunk([99.0, 3.0], t=[1.0, 2.0])
        t_out, x_out = f.get_filtered()
        rerun_log("chunk/overlap_latest/output", t_out, x_out)
        idx = np.where(t_out == 1.0)[0]
        np.testing.assert_allclose(x_out[idx], 99.0)

    def test_mean_strategy_averages(self, rerun_log):
        f = ChunkFilter(overlap_strategy="mean")
        f.add_chunk([0.0, 2.0], t=[0.0, 1.0])
        f.add_chunk([4.0, 3.0], t=[1.0, 2.0])
        t_out, x_out = f.get_filtered()
        rerun_log("chunk/overlap_mean/output", t_out, x_out)
        idx = np.where(t_out == 1.0)[0]
        np.testing.assert_allclose(x_out[idx], 3.0)

    def test_invalid_strategy_raises(self):
        f = ChunkFilter(overlap_strategy="unknown")
        f.add_chunk([1.0, 2.0], t=[0.0, 1.0])
        with pytest.raises(ValueError):
            f.get_filtered()


# ---------------------------------------------------------------------------
# Smoothing methods
# ---------------------------------------------------------------------------

class TestChunkFilterMethods:
    def test_spline_method(self, rerun_log):
        f = ChunkFilter(method="spline")
        t = np.linspace(0, 2 * np.pi, 30)
        x = np.sin(t)
        f.add_chunk(x, t=t)
        t_q = np.linspace(0.5, 2 * np.pi - 0.5, 10)
        _, x_out = f.get_filtered(t_query=t_q)
        rerun_log("chunk/spline/input", t, x)
        rerun_log("chunk/spline/output", t_q, x_out)
        rerun_log("chunk/spline/truth", t_q, np.sin(t_q))
        np.testing.assert_allclose(x_out, np.sin(t_q), atol=0.01)

    def test_polynomial_method(self, rerun_log):
        f = ChunkFilter(method="polynomial", degree=2)
        t = np.linspace(0, 1, 30)
        x = t ** 2
        f.add_chunk(x, t=t)
        t_q = np.array([0.0, 0.5, 1.0])
        _, x_out = f.get_filtered(t_query=t_q)
        rerun_log("chunk/polynomial/input", t, x)
        rerun_log("chunk/polynomial/output", t_q, x_out)
        np.testing.assert_allclose(x_out, t_q ** 2, atol=1e-6)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            ChunkFilter(method="unknown")

    def test_custom_query_times(self):
        f = ChunkFilter()
        f.add_chunk([0.0, 2.0], t=[0.0, 2.0])
        t_q = np.array([0.0, 1.0, 2.0])
        t_out, x_out = f.get_filtered(t_query=t_q)
        np.testing.assert_allclose(x_out, [0.0, 1.0, 2.0])


# ---------------------------------------------------------------------------
# Vector data
# ---------------------------------------------------------------------------

class TestChunkFilterVectorData:
    def test_vector_output_shape(self):
        f = ChunkFilter()
        x = np.column_stack([np.arange(5.0), np.arange(5.0) * 2])
        f.add_chunk(x, t=np.arange(5.0))
        t_out, x_out = f.get_filtered()
        assert x_out.shape == (5, 2)

    def test_vector_values_correct(self):
        f = ChunkFilter()
        x = np.column_stack([np.array([0.0, 2.0]), np.array([0.0, 4.0])])
        f.add_chunk(x, t=[0.0, 1.0])
        _, x_out = f.get_filtered(t_query=[0.5])
        np.testing.assert_allclose(x_out, [[1.0, 2.0]])


# ---------------------------------------------------------------------------
# Blend overlap strategy
# ---------------------------------------------------------------------------

class TestChunkFilterBlendOverlap:
    def test_blend_smooth_transition(self, rerun_log):
        """Overlap region should transition smoothly with no sharp jump."""
        t1 = np.linspace(0.0, 2.0, 21)
        x1 = np.linspace(0.0, 2.0, 21)

        t2 = np.linspace(1.5, 3.5, 21)
        x2 = np.linspace(2.3, 4.3, 21)

        f = ChunkFilter(overlap_strategy="blend")
        f.add_chunk(x1, t=t1)
        f.add_chunk(x2, t=t2)
        t_out, x_out = f.get_filtered()
        rerun_log("chunk/blend_smooth/output", t_out, x_out)

        # In the overlap region [1.5, 2.0], adjacent differences should be small
        overlap_mask = (t_out >= 1.5) & (t_out <= 2.0)
        overlap_vals = x_out[overlap_mask]
        diffs = np.abs(np.diff(overlap_vals))
        assert np.all(diffs < 0.5), "Blend should produce smooth transitions"

    def test_blend_vector_data(self):
        """Blend with 2D vector data should preserve shape."""
        t1 = np.linspace(0.0, 2.0, 11)
        x1 = np.column_stack([t1, t1 * 2])

        t2 = np.linspace(1.5, 3.5, 11)
        x2 = np.column_stack([t2 + 0.3, t2 * 2 + 0.3])

        f = ChunkFilter(overlap_strategy="blend")
        f.add_chunk(x1, t=t1)
        f.add_chunk(x2, t=t2)
        t_out, x_out = f.get_filtered()
        assert x_out.ndim == 2
        assert x_out.shape[1] == 2

    def test_blend_non_overlapping_chunks(self):
        """Non-overlapping chunks should work normally with blend strategy."""
        f = ChunkFilter(overlap_strategy="blend")
        f.add_chunk([1.0, 2.0], t=[0.0, 1.0])
        f.add_chunk([3.0, 4.0], t=[2.0, 3.0])
        t_out, x_out = f.get_filtered()
        assert len(t_out) == 4

    def test_blend_identical_to_mean_at_midpoint(self):
        """At the midpoint of the overlap, the blended value should be close
        to the average of the two chunks' values."""
        t1 = np.linspace(0.0, 2.0, 201)
        x1 = t1 * 1.0

        t2 = np.linspace(1.0, 3.0, 201)
        x2 = t2 * 1.0 + 0.4

        f = ChunkFilter(overlap_strategy="blend")
        f.add_chunk(x1, t=t1)
        f.add_chunk(x2, t=t2)
        t_out, x_out = f.get_filtered()

        # Midpoint of overlap [1.0, 2.0] is 1.5
        mid_idx = np.argmin(np.abs(t_out - 1.5))
        blended_val = x_out[mid_idx]

        # Expected chunk values at t=1.5
        val_chunk1 = 1.5
        val_chunk2 = 1.5 + 0.4
        expected_avg = (val_chunk1 + val_chunk2) / 2.0
        np.testing.assert_allclose(blended_val, expected_avg, atol=0.05)


# ---------------------------------------------------------------------------
# Savitzky-Golay filter
# ---------------------------------------------------------------------------

class TestChunkFilterSavgol:
    def test_preserves_polynomial(self):
        """Savgol with polyorder=2 should preserve a degree-2 polynomial."""
        t = np.linspace(0, 1, 51)
        x = 3.0 * t ** 2 - 2.0 * t + 1.0
        f = ChunkFilter(method="savgol", window_length=11, polyorder=2)
        f.add_chunk(x, t=t)
        _, x_out = f.get_filtered()
        np.testing.assert_allclose(x_out, x, atol=1e-6)

    def test_reduces_noise(self, rerun_log):
        """Filtered noisy sine should have lower MSE vs clean than raw."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 2 * np.pi, 200)
        clean = np.sin(t)
        noisy = clean + rng.normal(0, 0.3, size=len(t))

        f = ChunkFilter(method="savgol", window_length=15, polyorder=3)
        f.add_chunk(noisy, t=t)
        t_out, x_out = f.get_filtered()
        rerun_log("chunk/savgol/noisy", t, noisy)
        rerun_log("chunk/savgol/filtered", t_out, x_out)

        mse_noisy = np.mean((noisy - clean) ** 2)
        mse_filtered = np.mean((x_out - clean) ** 2)
        assert mse_filtered < mse_noisy

    def test_vector_data(self):
        """Savgol should handle 2D input and preserve shape."""
        t = np.linspace(0, 1, 51)
        x = np.column_stack([np.sin(t), np.cos(t)])
        f = ChunkFilter(method="savgol", window_length=11, polyorder=3)
        f.add_chunk(x, t=t)
        _, x_out = f.get_filtered()
        assert x_out.shape == (51, 2)


# ---------------------------------------------------------------------------
# Gaussian filter
# ---------------------------------------------------------------------------

class TestChunkFilterGaussian:
    def test_smooths_data(self):
        """Gaussian filter should produce smoother output than noisy input."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, 100)
        noisy = np.sin(2 * np.pi * t) + rng.normal(0, 0.3, size=len(t))

        f = ChunkFilter(method="gaussian", sigma=3.0)
        f.add_chunk(noisy, t=t)
        _, x_out = f.get_filtered()

        var_input = np.var(np.diff(noisy))
        var_output = np.var(np.diff(x_out))
        assert var_output < var_input

    def test_vector_data(self):
        """Gaussian filter should handle 2D input."""
        t = np.linspace(0, 1, 50)
        x = np.column_stack([np.sin(t), np.cos(t)])
        f = ChunkFilter(method="gaussian", sigma=2.0)
        f.add_chunk(x, t=t)
        _, x_out = f.get_filtered()
        assert x_out.shape == (50, 2)


# ---------------------------------------------------------------------------
# Lowpass filter
# ---------------------------------------------------------------------------

class TestChunkFilterLowpass:
    def test_reduces_noise(self):
        """Lowpass-filtered noisy sine should have lower MSE vs clean."""
        rng = np.random.default_rng(42)
        n = 200
        sample_rate = 100.0
        t = np.arange(n) / sample_rate
        clean = np.sin(2 * np.pi * 2.0 * t)
        noisy = clean + rng.normal(0, 0.3, size=n)

        f = ChunkFilter(method="lowpass", cutoff_freq=5.0,
                         sample_rate=sample_rate, order=4)
        f.add_chunk(noisy, t=t)
        _, x_out = f.get_filtered()

        mse_noisy = np.mean((noisy - clean) ** 2)
        mse_filtered = np.mean((x_out - clean) ** 2)
        assert mse_filtered < mse_noisy

    def test_vector_data(self):
        """Lowpass filter should handle 2D input."""
        t = np.linspace(0, 1, 100)
        x = np.column_stack([np.sin(t), np.cos(t)])
        f = ChunkFilter(method="lowpass", cutoff_freq=5.0, sample_rate=100.0)
        f.add_chunk(x, t=t)
        _, x_out = f.get_filtered()
        assert x_out.shape == (100, 2)


# ---------------------------------------------------------------------------
# Median filter
# ---------------------------------------------------------------------------

class TestChunkFilterMedian:
    def test_removes_spike(self):
        """Median filter should remove a single spike from clean data."""
        x = np.ones(51)
        x[25] = 100.0  # spike
        t = np.linspace(0, 1, 51)

        f = ChunkFilter(method="median", kernel_size=5)
        f.add_chunk(x, t=t)
        _, x_out = f.get_filtered()
        np.testing.assert_allclose(x_out[25], 1.0, atol=0.01)

    def test_vector_data(self):
        """Median filter should handle 2D input."""
        t = np.linspace(0, 1, 51)
        x = np.column_stack([np.sin(t), np.cos(t)])
        f = ChunkFilter(method="median", kernel_size=5)
        f.add_chunk(x, t=t)
        _, x_out = f.get_filtered()
        assert x_out.shape == (51, 2)


# ---------------------------------------------------------------------------
# Cosine blend overlap strategy
# ---------------------------------------------------------------------------

class TestChunkFilterCosineBlend:
    def test_cosine_blend_smooth_transition(self, rerun_log):
        """Cosine crossfade in overlap should give a smooth transition."""
        f = ChunkFilter(method="linear", overlap_strategy="cosine_blend")
        t1 = np.linspace(0, 5, 50)
        t2 = np.linspace(4, 10, 60)
        x1 = np.sin(t1) + 0.5  # deliberate offset
        x2 = np.sin(t2)
        f.add_chunk(x1, t=t1)
        f.add_chunk(x2, t=t2)
        t_out, x_out = f.get_filtered()
        rerun_log("chunk/cosine_blend/output", t_out, x_out)
        overlap_mask = (t_out >= 4.0) & (t_out <= 5.0)
        x_overlap = x_out[overlap_mask]
        diffs = np.abs(np.diff(x_overlap))
        assert np.max(diffs) < 0.2, f"Max diff in overlap region too large: {np.max(diffs)}"

    def test_cosine_blend_vector_data(self):
        f = ChunkFilter(method="linear", overlap_strategy="cosine_blend")
        t1 = np.linspace(0, 5, 50)
        t2 = np.linspace(4, 10, 60)
        x1 = np.column_stack([np.sin(t1), np.cos(t1)])
        x2 = np.column_stack([np.sin(t2), np.cos(t2)])
        f.add_chunk(x1, t=t1)
        f.add_chunk(x2, t=t2)
        t_out, x_out = f.get_filtered()
        assert x_out.shape[1] == 2

    def test_cosine_blend_non_overlapping(self):
        """Non-overlapping chunks should concatenate normally."""
        f = ChunkFilter(method="linear", overlap_strategy="cosine_blend")
        f.add_chunk([1.0, 2.0], t=[0.0, 1.0])
        f.add_chunk([3.0, 4.0], t=[2.0, 3.0])
        t_out, x_out = f.get_filtered()
        assert len(t_out) == 4

    def test_cosine_smoother_than_linear_blend(self):
        """Cosine blend should start/end with near-zero slope at overlap
        boundaries, whereas linear blend has a constant slope."""
        t1 = np.linspace(0, 6, 200)
        t2 = np.linspace(4, 10, 200)

        f_lin = ChunkFilter(method="linear", overlap_strategy="blend")
        f_lin.add_chunk(np.zeros(200), t=t1)
        f_lin.add_chunk(np.ones(200), t=t2)
        t_lin, x_lin = f_lin.get_filtered()

        f_cos = ChunkFilter(method="linear", overlap_strategy="cosine_blend")
        f_cos.add_chunk(np.zeros(200), t=t1)
        f_cos.add_chunk(np.ones(200), t=t2)
        t_cos, x_cos = f_cos.get_filtered()

        # At the midpoint both should be ~0.5
        mid_idx = np.argmin(np.abs(t_cos - 5.0))
        np.testing.assert_allclose(x_cos[mid_idx], 0.5, atol=0.05)

        # Cosine slope at the start of the overlap should be near 0
        # Linear slope is constant = 1/(overlap_end - overlap_start) = 0.5
        overlap_mask_cos = (t_cos >= 4.0) & (t_cos <= 4.3)
        overlap_mask_lin = (t_lin >= 4.0) & (t_lin <= 4.3)
        dx_cos = np.diff(x_cos[overlap_mask_cos])
        dx_lin = np.diff(x_lin[overlap_mask_lin])
        # Mean slope near start for cosine should be smaller than linear
        assert np.mean(np.abs(dx_cos)) < np.mean(np.abs(dx_lin))


# ---------------------------------------------------------------------------
# FIR chunk filter
# ---------------------------------------------------------------------------

class TestChunkFilterFIR:
    def test_reduces_noise(self, rerun_log):
        rng = np.random.default_rng(42)
        n = 200
        sr = 100.0
        t = np.arange(n) / sr
        clean = np.sin(2 * np.pi * 2.0 * t)
        noisy = clean + rng.normal(0, 0.3, size=n)

        f = ChunkFilter(method="fir", numtaps=31, cutoff_freq=5.0, sample_rate=sr)
        f.add_chunk(noisy, t=t)
        _, x_out = f.get_filtered()
        rerun_log("chunk/fir/noisy", t, noisy)
        rerun_log("chunk/fir/filtered", t, x_out)
        assert np.mean((x_out - clean) ** 2) < np.mean((noisy - clean) ** 2)

    def test_vector_data(self):
        t = np.linspace(0, 1, 100)
        x = np.column_stack([np.sin(t), np.cos(t)])
        f = ChunkFilter(method="fir", numtaps=15, cutoff_freq=5.0, sample_rate=100.0)
        f.add_chunk(x, t=t)
        _, x_out = f.get_filtered()
        assert x_out.shape == (100, 2)

    def test_two_chunks(self):
        t1 = np.linspace(0, 1, 100)
        t2 = np.linspace(0.9, 2, 100)
        x1 = np.sin(t1)
        x2 = np.sin(t2)
        f = ChunkFilter(method="fir", numtaps=15, cutoff_freq=5.0, sample_rate=100.0)
        f.add_chunk(x1, t=t1)
        f.add_chunk(x2, t=t2)
        t_out, x_out = f.get_filtered()
        assert len(t_out) > 100


# ---------------------------------------------------------------------------
# IIR chunk filter
# ---------------------------------------------------------------------------

class TestChunkFilterIIR:
    def test_butterworth_reduces_noise(self, rerun_log):
        rng = np.random.default_rng(42)
        n = 200
        sr = 100.0
        t = np.arange(n) / sr
        clean = np.sin(2 * np.pi * 2.0 * t)
        noisy = clean + rng.normal(0, 0.3, size=n)

        f = ChunkFilter(
            method="iir", cutoff_freq=5.0, sample_rate=sr,
            iir_type="butterworth",
        )
        f.add_chunk(noisy, t=t)
        _, x_out = f.get_filtered()
        rerun_log("chunk/iir_butter/noisy", t, noisy)
        rerun_log("chunk/iir_butter/filtered", t, x_out)
        assert np.mean((x_out - clean) ** 2) < np.mean((noisy - clean) ** 2)

    def test_chebyshev1(self):
        t = np.linspace(0, 1, 100)
        x = np.sin(t)
        f = ChunkFilter(
            method="iir", cutoff_freq=5.0, sample_rate=100.0,
            iir_type="chebyshev1", rp=1.0,
        )
        f.add_chunk(x, t=t)
        _, x_out = f.get_filtered()
        assert x_out.shape == (100,)

    def test_bessel(self):
        t = np.linspace(0, 1, 100)
        x = np.sin(t)
        f = ChunkFilter(
            method="iir", cutoff_freq=5.0, sample_rate=100.0,
            iir_type="bessel",
        )
        f.add_chunk(x, t=t)
        _, x_out = f.get_filtered()
        assert x_out.shape == (100,)

    def test_vector_data(self):
        t = np.linspace(0, 1, 100)
        x = np.column_stack([np.sin(t), np.cos(t)])
        f = ChunkFilter(
            method="iir", cutoff_freq=5.0, sample_rate=100.0,
        )
        f.add_chunk(x, t=t)
        _, x_out = f.get_filtered()
        assert x_out.shape == (100, 2)


# ---------------------------------------------------------------------------
# Kalman chunk filter
# ---------------------------------------------------------------------------

class TestChunkFilterKalman:
    def test_reduces_noise(self, rerun_log):
        rng = np.random.default_rng(42)
        t = np.linspace(0, 2 * np.pi, 200)
        clean = np.sin(t)
        noisy = clean + rng.normal(0, 0.3, size=200)

        f = ChunkFilter(
            method="kalman", process_noise=0.01, measurement_noise=0.09,
        )
        f.add_chunk(noisy, t=t)
        _, x_out = f.get_filtered()
        rerun_log("chunk/kalman/noisy", t, noisy)
        rerun_log("chunk/kalman/filtered", t, x_out)
        assert np.mean((x_out - clean) ** 2) < np.mean((noisy - clean) ** 2)

    def test_position_velocity_model(self):
        t = np.linspace(0, 1, 100)
        x = t ** 2
        rng = np.random.default_rng(42)
        x_noisy = x + rng.normal(0, 0.1, size=100)
        f = ChunkFilter(
            method="kalman", state_model="position_velocity",
            process_noise=0.1, measurement_noise=0.01,
        )
        f.add_chunk(x_noisy, t=t)
        _, x_out = f.get_filtered()
        assert x_out.shape == (100,)

    def test_vector_data(self):
        t = np.linspace(0, 1, 50)
        x = np.column_stack([np.sin(t), np.cos(t)])
        f = ChunkFilter(method="kalman")
        f.add_chunk(x, t=t)
        _, x_out = f.get_filtered()
        assert x_out.shape == (50, 2)

    def test_two_chunks(self):
        t1 = np.linspace(0, 1, 50)
        t2 = np.linspace(0.8, 2, 60)
        f = ChunkFilter(method="kalman", process_noise=0.01, measurement_noise=0.1)
        f.add_chunk(np.sin(t1), t=t1)
        f.add_chunk(np.sin(t2), t=t2)
        t_out, x_out = f.get_filtered()
        assert len(t_out) > 50
