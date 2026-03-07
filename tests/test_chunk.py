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

    def test_single_chunk_with_timestamps(self):
        f = ChunkFilter()
        f.add_chunk([1.0, 2.0, 3.0], t=[0.0, 0.5, 1.0])
        t_out, x_out = f.get_filtered()
        np.testing.assert_allclose(t_out, [0.0, 0.5, 1.0])
        np.testing.assert_allclose(x_out, [1.0, 2.0, 3.0])

    def test_two_non_overlapping_chunks(self):
        f = ChunkFilter()
        f.add_chunk([1.0, 2.0], t=[0.0, 1.0])
        f.add_chunk([3.0, 4.0], t=[2.0, 3.0])
        t_out, x_out = f.get_filtered()
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
    def test_latest_strategy_keeps_newest(self):
        f = ChunkFilter(overlap_strategy="latest")
        f.add_chunk([1.0, 2.0], t=[0.0, 1.0])
        # Second chunk overlaps at t=1.0 with a different value
        f.add_chunk([99.0, 3.0], t=[1.0, 2.0])
        t_out, x_out = f.get_filtered()
        idx = np.where(t_out == 1.0)[0]
        np.testing.assert_allclose(x_out[idx], 99.0)

    def test_mean_strategy_averages(self):
        f = ChunkFilter(overlap_strategy="mean")
        f.add_chunk([0.0, 2.0], t=[0.0, 1.0])
        f.add_chunk([4.0, 3.0], t=[1.0, 2.0])
        t_out, x_out = f.get_filtered()
        # t=1.0 should be (2.0 + 4.0) / 2 = 3.0
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
    def test_spline_method(self):
        f = ChunkFilter(method="spline")
        t = np.linspace(0, 2 * np.pi, 30)
        x = np.sin(t)
        f.add_chunk(x, t=t)
        t_q = np.linspace(0.5, 2 * np.pi - 0.5, 10)
        _, x_out = f.get_filtered(t_query=t_q)
        np.testing.assert_allclose(x_out, np.sin(t_q), atol=0.01)

    def test_polynomial_method(self):
        f = ChunkFilter(method="polynomial", degree=2)
        t = np.linspace(0, 1, 30)
        x = t ** 2
        f.add_chunk(x, t=t)
        t_q = np.array([0.0, 0.5, 1.0])
        _, x_out = f.get_filtered(t_query=t_q)
        np.testing.assert_allclose(x_out, t_q ** 2, atol=1e-6)

    def test_invalid_method_raises(self):
        f = ChunkFilter(method="unknown")
        f.add_chunk([1.0, 2.0], t=[0.0, 1.0])
        with pytest.raises(ValueError):
            f.get_filtered()

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
