"""Offline time series filtering and smoothing.

The entire time series is provided at once and processed in batch.
Each data point at each timestamp is a real-valued scalar or vector.
"""
from __future__ import annotations

import numpy as np
from scipy import interpolate as sp_interpolate
from scipy import signal as sp_signal


class OfflineFilter:
    """Filter and smoother for offline (batch) time series processing.

    A complete time series is provided at once and processed offline.
    Each data point at each timestamp is a real-valued vector.

    Parameters
    ----------
    t : array-like, shape (N,)
        Timestamps of the data points.  Need not be sorted.
    x : array-like, shape (N,) or (N, D)
        Data values.  One-dimensional input is treated as scalar (D=1).

    Raises
    ------
    ValueError
        If ``t`` and ``x`` have incompatible lengths.
    """

    def __init__(self, t, x) -> None:
        self.t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[:, np.newaxis]
            self._scalar_input = True
        else:
            self._scalar_input = False
        self.x = x

        if len(self.t) != len(self.x):
            raise ValueError(
                f"t and x must have the same number of samples, "
                f"got {len(self.t)} and {len(self.x)}"
            )

        # Sort by time (stable so equal timestamps keep insertion order)
        idx = np.argsort(self.t, kind="stable")
        self.t = self.t[idx]
        self.x = self.x[idx]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        """Number of data samples."""
        return len(self.t)

    @property
    def n_dims(self) -> int:
        """Dimensionality of each data point."""
        return self.x.shape[1]

    # ------------------------------------------------------------------
    # Public filtering / smoothing methods
    # ------------------------------------------------------------------

    def linear_interpolate(self, t_query) -> np.ndarray:
        """Piecewise-linear interpolation at query timestamps.

        Parameters
        ----------
        t_query : array-like, shape (M,)
            Query timestamps.

        Returns
        -------
        np.ndarray, shape (M,) or (M, D)
            Interpolated values.  Shape is ``(M,)`` when input data was
            1-D, otherwise ``(M, D)``.
        """
        t_query = np.atleast_1d(np.asarray(t_query, dtype=float))
        result = np.column_stack(
            [np.interp(t_query, self.t, self.x[:, d]) for d in range(self.n_dims)]
        )
        return self._squeeze_output(result)

    def lowpass_filter(
        self,
        cutoff_freq: float,
        sample_rate: float,
        order: int = 4,
    ) -> np.ndarray:
        """Apply a zero-phase Butterworth low-pass filter.

        Uses :func:`scipy.signal.filtfilt` for zero-phase (forward-backward)
        filtering, which introduces no phase distortion.

        Parameters
        ----------
        cutoff_freq : float
            Cutoff frequency (same units as ``sample_rate``).
        sample_rate : float
            Sampling rate of the data.
        order : int, optional
            Order of the Butterworth filter (default: 4).

        Returns
        -------
        np.ndarray, shape (N,) or (N, D)
            Filtered data.

        Raises
        ------
        ValueError
            If the normalised cutoff frequency is not in ``(0, 1)``.
        """
        nyq = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyq
        if not (0.0 < normal_cutoff < 1.0):
            raise ValueError(
                f"Normalised cutoff frequency must be in (0, 1); "
                f"got {normal_cutoff:.4f}. "
                "Check cutoff_freq and sample_rate values."
            )
        b, a = sp_signal.butter(order, normal_cutoff, btype="low", analog=False)
        result = np.column_stack(
            [sp_signal.filtfilt(b, a, self.x[:, d]) for d in range(self.n_dims)]
        )
        return self._squeeze_output(result)

    def polynomial_fit(self, degree: int, t_query=None) -> np.ndarray:
        """Fit a least-squares polynomial and evaluate at query timestamps.

        Parameters
        ----------
        degree : int
            Degree of the fitting polynomial.
        t_query : array-like, shape (M,), optional
            Query timestamps.  Defaults to the original timestamps
            ``self.t``.

        Returns
        -------
        np.ndarray, shape (M,) or (M, D)
            Polynomial-fitted values.
        """
        if t_query is None:
            t_query = self.t
        t_query = np.atleast_1d(np.asarray(t_query, dtype=float))
        result = np.column_stack(
            [
                np.polyval(np.polyfit(self.t, self.x[:, d], degree), t_query)
                for d in range(self.n_dims)
            ]
        )
        return self._squeeze_output(result)

    def spline_interpolate(self, t_query, kind: str = "cubic") -> np.ndarray:
        """Spline interpolation at query timestamps.

        Parameters
        ----------
        t_query : array-like, shape (M,)
            Query timestamps.
        kind : str or int, optional
            Type of spline interpolation passed to
            :func:`scipy.interpolate.interp1d` (default: ``'cubic'``).

        Returns
        -------
        np.ndarray, shape (M,) or (M, D)
            Interpolated values.
        """
        t_query = np.atleast_1d(np.asarray(t_query, dtype=float))
        result = np.column_stack(
            [
                sp_interpolate.interp1d(
                    self.t,
                    self.x[:, d],
                    kind=kind,
                    fill_value="extrapolate",
                )(t_query)
                for d in range(self.n_dims)
            ]
        )
        return self._squeeze_output(result)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _squeeze_output(self, y: np.ndarray) -> np.ndarray:
        """Squeeze trailing dimension when the original input was 1-D."""
        if self._scalar_input:
            return y.squeeze(-1)
        return y
