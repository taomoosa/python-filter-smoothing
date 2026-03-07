"""Chunk-based online time series filtering and smoothing.

Chunks of time series data are added one at a time.  Chunks may have
overlapping timestamps or imprecise / missing timing information.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import interpolate as sp_interpolate


class ChunkFilter:
    """Filter and smoother for chunk-based online time series processing.

    Chunks of time series data are added one at a time.  Each chunk may
    overlap in time with previously added chunks.  When exact timestamps
    are unknown, a chunk can be appended without timestamps and synthetic
    ones will be generated automatically.

    Parameters
    ----------
    method : str, optional
        Smoothing / interpolation method applied to the merged data.
        One of:

        - ``'linear'``     – Piecewise-linear interpolation (default).
        - ``'spline'``     – Cubic-spline interpolation.
        - ``'polynomial'`` – Least-squares polynomial fit.

    overlap_strategy : str, optional
        How to handle duplicate / overlapping timestamps.  One of:

        - ``'latest'`` – Keep the most recently added sample (default).
        - ``'mean'``   – Average over samples sharing the same timestamp.

    **kwargs
        Method-specific keyword arguments.

        For ``'polynomial'``:
            degree : int, optional
                Polynomial degree (default: ``3``).

        For ``'spline'``:
            kind : str, optional
                Spline kind passed to :func:`scipy.interpolate.interp1d`
                (default: ``'cubic'``).

    Raises
    ------
    ValueError
        If an unsupported method or overlap_strategy is given.
    """

    def __init__(
        self,
        method: str = "linear",
        overlap_strategy: str = "latest",
        **kwargs,
    ) -> None:
        self.method = method
        self.overlap_strategy = overlap_strategy
        self._kwargs = kwargs

        self._t_chunks: list[np.ndarray] = []
        self._x_chunks: list[np.ndarray] = []
        self._current_end_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_chunk(self, x, t=None, dt: float = 1.0) -> None:
        """Add a chunk of time series data.

        Parameters
        ----------
        x : array-like, shape (M,) or (M, D)
            Data values for this chunk.  ``M`` is the number of samples;
            ``D`` is the data dimensionality.
        t : array-like, shape (M,), optional
            Timestamps for this chunk.  If ``None``, timestamps are
            generated automatically with step ``dt`` starting from the
            end of the previously added chunk.
        dt : float, optional
            Time step used for two purposes:

            1. When ``t`` is ``None``, timestamps are auto-generated with
               this step size starting from the end of the previous chunk.
            2. Regardless of whether ``t`` is provided, ``dt`` is added to
               the last timestamp of this chunk to determine where the next
               auto-generated chunk will start (default: ``1.0``).

        Raises
        ------
        ValueError
            If ``t`` and ``x`` have incompatible lengths.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[:, np.newaxis]
        n = len(x)

        if t is None:
            t = np.arange(n, dtype=float) * dt + self._current_end_time
        else:
            t = np.asarray(t, dtype=float)

        if len(t) != n:
            raise ValueError(
                f"t and x must have the same number of samples, "
                f"got {len(t)} and {n}"
            )

        self._t_chunks.append(t)
        self._x_chunks.append(x)
        self._current_end_time = float(t[-1]) + dt

    def get_filtered(
        self, t_query=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return filtered data over the merged time series.

        Parameters
        ----------
        t_query : array-like, shape (M,), optional
            Query timestamps.  If ``None``, the deduplicated union of all
            chunk timestamps is used.

        Returns
        -------
        t_out : np.ndarray, shape (M,)
            Output timestamps.
        x_out : np.ndarray, shape (M,) or (M, D)
            Filtered / smoothed values.  Shape is ``(M,)`` when all
            chunks contained 1-D scalar data, otherwise ``(M, D)``.

        Raises
        ------
        RuntimeError
            If no data has been added yet.
        """
        if not self._t_chunks:
            raise RuntimeError("No data has been added. Call add_chunk first.")

        t_merged, x_merged = self._merge()
        scalar = all(c.shape[1] == 1 for c in self._x_chunks)

        t_out = (
            t_merged
            if t_query is None
            else np.atleast_1d(np.asarray(t_query, dtype=float))
        )

        x_out = self._apply_method(t_merged, x_merged, t_out)
        if scalar:
            x_out = x_out.squeeze(-1)
        return t_out, x_out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _merge(self) -> Tuple[np.ndarray, np.ndarray]:
        """Merge all accumulated chunks into a single sorted series."""
        t_all = np.concatenate(self._t_chunks)
        x_all = np.concatenate(self._x_chunks, axis=0)

        # Stable sort by time so that later chunks stay after earlier ones
        # when timestamps are equal.
        idx = np.argsort(t_all, kind="stable")
        t_all = t_all[idx]
        x_all = x_all[idx]

        if self.overlap_strategy == "latest":
            # Keep the last (most recently added) sample at each timestamp.
            # After stable sort, the last occurrence is at the highest index.
            _, last_in_reversed = np.unique(t_all[::-1], return_index=True)
            keep = np.sort(len(t_all) - 1 - last_in_reversed)
            return t_all[keep], x_all[keep]

        elif self.overlap_strategy == "mean":
            unique_t, inverse = np.unique(t_all, return_inverse=True)
            x_mean = np.zeros((len(unique_t), x_all.shape[1]))
            counts = np.zeros(len(unique_t))
            for i in range(len(x_all)):
                x_mean[inverse[i]] += x_all[i]
                counts[inverse[i]] += 1
            x_mean /= counts[:, np.newaxis]
            return unique_t, x_mean

        else:
            raise ValueError(
                f"Unknown overlap_strategy '{self.overlap_strategy}'. "
                "Choose from: 'latest', 'mean'."
            )

    def _apply_method(
        self,
        t: np.ndarray,
        x: np.ndarray,
        t_query: np.ndarray,
    ) -> np.ndarray:
        """Apply the chosen smoothing method and return shape (M, D)."""
        n_dims = x.shape[1]

        if self.method == "linear":
            return np.column_stack(
                [np.interp(t_query, t, x[:, d]) for d in range(n_dims)]
            )

        elif self.method == "spline":
            kind = self._kwargs.get("kind", "cubic")
            return np.column_stack(
                [
                    sp_interpolate.interp1d(
                        t, x[:, d], kind=kind, fill_value="extrapolate"
                    )(t_query)
                    for d in range(n_dims)
                ]
            )

        elif self.method == "polynomial":
            degree = self._kwargs.get("degree", 3)
            return np.column_stack(
                [
                    np.polyval(np.polyfit(t, x[:, d], degree), t_query)
                    for d in range(n_dims)
                ]
            )

        else:
            raise ValueError(
                f"Unknown method '{self.method}'. "
                "Choose from: 'linear', 'spline', 'polynomial'."
            )
