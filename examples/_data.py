"""Shared data generation utilities for examples.

Generates clean and noisy XYZ vector time series, as well as chunked
versions with small inter-chunk discontinuities and optional timestamp overlap.
"""

from __future__ import annotations

import numpy as np


DIM_NAMES = ["x", "y", "z"]


def generate_xyz_signal(
    duration: float = 10.0,
    sample_rate: float = 100.0,
    noise_std: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a clean + noisy XYZ time series.

    Clean signal
    ~~~~~~~~~~~~~
    - X = sin(t)
    - Y = cos(t)
    - Z = sin(2t) + 0.5 * cos(t)

    Returns
    -------
    t : ndarray, shape (N,)
    x_clean : ndarray, shape (N, 3)
    x_noisy : ndarray, shape (N, 3)
    """
    rng = np.random.RandomState(seed)
    n = int(duration * sample_rate)
    t = np.linspace(0, duration, n, endpoint=False)

    x_clean = np.column_stack([
        np.sin(t),
        np.cos(t),
        np.sin(2 * t) + 0.5 * np.cos(t),
    ])
    x_noisy = x_clean + rng.randn(n, 3) * noise_std

    return t, x_clean, x_noisy


def generate_chunked_xyz(
    duration: float = 10.0,
    sample_rate: float = 100.0,
    noise_std: float = 0.15,
    n_chunks: int = 10,
    overlap_samples: int = 5,
    discontinuity_std: float = 0.12,
    seed: int = 42,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[np.ndarray, np.ndarray]],
]:
    """Generate chunked XYZ data with inter-chunk discontinuities and overlap.

    Each chunk covers a time segment.  Adjacent chunks overlap by
    *overlap_samples* samples and have a small random offset added to
    simulate sensor recalibration or packet reordering.

    Returns
    -------
    t_full : ndarray, shape (N,)
        Full (non-chunked) timestamp array.
    x_clean : ndarray, shape (N, 3)
        Clean reference signal.
    x_noisy : ndarray, shape (N, 3)
        Noisy signal (before chunking).
    chunks : list of (t_chunk, x_chunk)
        Each element is a (timestamps, values) pair for one chunk, including
        the discontinuity offset.  Timestamps may overlap with adjacent chunks.
    """
    rng = np.random.RandomState(seed)
    t_full, x_clean, x_noisy = generate_xyz_signal(
        duration, sample_rate, noise_std, seed,
    )
    n = len(t_full)

    chunk_size = n // n_chunks
    chunks: list[tuple[np.ndarray, np.ndarray]] = []
    offset = np.zeros(3)

    for i in range(n_chunks):
        start = max(0, i * chunk_size - overlap_samples)
        end = min(n, (i + 1) * chunk_size)
        if i == n_chunks - 1:
            end = n

        t_chunk = t_full[start:end].copy()
        x_chunk = x_noisy[start:end].copy() + offset

        # Add a small discontinuity for the *next* chunk
        if i < n_chunks - 1:
            offset = offset + rng.randn(3) * discontinuity_std

        chunks.append((t_chunk, x_chunk))

    return t_full, x_clean, x_noisy, chunks
