"""Chunk-based online time series filtering and smoothing.

Chunks of time series data are added one at a time.  Chunks may have
overlapping timestamps or imprecise / missing timing information.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from scipy import interpolate as sp_interpolate
from scipy import signal as sp_signal
from scipy.ndimage import gaussian_filter1d, median_filter as _nd_median


# ======================================================================
# Abstract base class
# ======================================================================


class ChunkFilterBase(ABC):
    """Abstract base for chunk-based online time series filters.

    Subclasses only need to implement :meth:`_apply`, which receives the
    merged (deduplicated) time series and the desired query timestamps
    and returns the interpolated / smoothed values.

    Parameters
    ----------
    overlap_strategy : str, optional
        How to handle duplicate / overlapping timestamps.  One of:

        - ``'latest'``       – Keep the most recently added sample (default).
        - ``'mean'``         – Average over samples sharing the same timestamp.
        - ``'blend'``        – Linear crossfade in the overlap region.
        - ``'cosine_blend'`` – Cosine (C¹-smooth) crossfade in the overlap region.
    """

    def __init__(self, overlap_strategy: str = "latest") -> None:
        self.overlap_strategy = overlap_strategy
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

        x_out = self._apply(t_merged, x_merged, t_out)
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

        elif self.overlap_strategy == "blend":
            return self._merge_blend(t_all, x_all)

        elif self.overlap_strategy == "cosine_blend":
            return self._merge_cosine_blend(t_all, x_all)

        else:
            raise ValueError(
                f"Unknown overlap_strategy '{self.overlap_strategy}'. "
                "Choose from: 'latest', 'mean', 'blend', 'cosine_blend'."
            )

    def _merge_blend(
        self,
        t_sorted: np.ndarray,
        x_sorted: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Merge with linear crossfade in overlap regions.

        Each pair of adjacent chunks that overlap in time is blended with
        a weight that transitions linearly from 0 (old chunk) to 1 (new
        chunk) across the overlap region.
        """
        if len(self._t_chunks) <= 1:
            unique_t, idx = np.unique(t_sorted, return_index=True)
            return unique_t, x_sorted[idx]

        # Build merged series chunk by chunk
        n_dims = self._x_chunks[0].shape[1]
        t_merged = self._t_chunks[0].copy()
        x_merged = self._x_chunks[0].copy()

        for k in range(1, len(self._t_chunks)):
            t_new = self._t_chunks[k]
            x_new = self._x_chunks[k]

            # Overlap region: timestamps present in both old and new
            overlap_start = max(t_merged[0], t_new[0])
            overlap_end = min(t_merged[-1], t_new[-1])

            if overlap_start >= overlap_end:
                # No overlap – simply concatenate
                t_merged = np.concatenate([t_merged, t_new])
                x_merged = np.concatenate([x_merged, x_new], axis=0)
                idx = np.argsort(t_merged, kind="stable")
                t_merged = t_merged[idx]
                x_merged = x_merged[idx]
                # Deduplicate (keep latest)
                _, ui = np.unique(t_merged[::-1], return_index=True)
                keep = np.sort(len(t_merged) - 1 - ui)
                t_merged = t_merged[keep]
                x_merged = x_merged[keep]
                continue

            # Build a common timeline covering both chunks
            t_union = np.unique(np.concatenate([t_merged, t_new]))

            # Interpolate old and new onto the union grid
            x_old_interp = np.column_stack(
                [np.interp(t_union, t_merged, x_merged[:, d]) for d in range(n_dims)]
            )
            x_new_interp = np.column_stack(
                [np.interp(t_union, t_new, x_new[:, d]) for d in range(n_dims)]
            )

            # Blend weight: 0 at overlap_start → 1 at overlap_end
            w = np.zeros(len(t_union))
            in_overlap = (t_union >= overlap_start) & (t_union <= overlap_end)
            span = overlap_end - overlap_start
            if span > 0:
                w[in_overlap] = (t_union[in_overlap] - overlap_start) / span
            # Before overlap: weight = 0 (use old), after overlap: weight = 1 (use new)
            w[t_union > overlap_end] = 1.0

            x_blended = (1.0 - w[:, np.newaxis]) * x_old_interp + w[:, np.newaxis] * x_new_interp
            t_merged = t_union
            x_merged = x_blended

        return t_merged, x_merged

    def _merge_cosine_blend(
        self,
        t_sorted: np.ndarray,
        x_sorted: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Merge with raised-cosine crossfade in overlap regions.

        Like :meth:`_merge_blend` but uses a raised-cosine (Hann) weight
        ``w = 0.5*(1 - cos(π*u))`` instead of a linear ramp, giving a
        smoother C¹-continuous transition at overlap boundaries.
        """
        if len(self._t_chunks) <= 1:
            unique_t, idx = np.unique(t_sorted, return_index=True)
            return unique_t, x_sorted[idx]

        n_dims = self._x_chunks[0].shape[1]
        t_merged = self._t_chunks[0].copy()
        x_merged = self._x_chunks[0].copy()

        for k in range(1, len(self._t_chunks)):
            t_new = self._t_chunks[k]
            x_new = self._x_chunks[k]

            overlap_start = max(t_merged[0], t_new[0])
            overlap_end = min(t_merged[-1], t_new[-1])

            if overlap_start >= overlap_end:
                t_merged = np.concatenate([t_merged, t_new])
                x_merged = np.concatenate([x_merged, x_new], axis=0)
                idx = np.argsort(t_merged, kind="stable")
                t_merged = t_merged[idx]
                x_merged = x_merged[idx]
                _, ui = np.unique(t_merged[::-1], return_index=True)
                keep = np.sort(len(t_merged) - 1 - ui)
                t_merged = t_merged[keep]
                x_merged = x_merged[keep]
                continue

            t_union = np.unique(np.concatenate([t_merged, t_new]))

            x_old_interp = np.column_stack(
                [np.interp(t_union, t_merged, x_merged[:, d]) for d in range(n_dims)]
            )
            x_new_interp = np.column_stack(
                [np.interp(t_union, t_new, x_new[:, d]) for d in range(n_dims)]
            )

            w = np.zeros(len(t_union))
            in_overlap = (t_union >= overlap_start) & (t_union <= overlap_end)
            span = overlap_end - overlap_start
            if span > 0:
                linear_u = (t_union[in_overlap] - overlap_start) / span
                w[in_overlap] = 0.5 * (1.0 - np.cos(np.pi * linear_u))
            w[t_union > overlap_end] = 1.0

            x_blended = (
                (1.0 - w[:, np.newaxis]) * x_old_interp
                + w[:, np.newaxis] * x_new_interp
            )
            t_merged = t_union
            x_merged = x_blended

        return t_merged, x_merged

    @abstractmethod
    def _apply(
        self,
        t: np.ndarray,
        x: np.ndarray,
        t_query: np.ndarray,
    ) -> np.ndarray:
        """Apply the smoothing method and return an array of shape (M, D).

        Parameters
        ----------
        t : np.ndarray, shape (N,)
            Merged, deduplicated timestamps.
        x : np.ndarray, shape (N, D)
            Merged, deduplicated values.
        t_query : np.ndarray, shape (M,)
            Desired output timestamps.
        """


# ======================================================================
# Concrete subclasses
# ======================================================================


class ChunkFilterLinear(ChunkFilterBase):
    """Piecewise-linear interpolation chunk filter."""

    def _apply(
        self,
        t: np.ndarray,
        x: np.ndarray,
        t_query: np.ndarray,
    ) -> np.ndarray:
        n_dims = x.shape[1]
        return np.column_stack(
            [np.interp(t_query, t, x[:, d]) for d in range(n_dims)]
        )


class ChunkFilterSpline(ChunkFilterBase):
    """Spline interpolation chunk filter.

    Parameters
    ----------
    overlap_strategy : str, optional
        Overlap handling strategy (default: ``'latest'``).
    kind : str, optional
        Spline kind passed to :func:`scipy.interpolate.interp1d`
        (default: ``'cubic'``).
    """

    def __init__(
        self,
        overlap_strategy: str = "latest",
        kind: str = "cubic",
    ) -> None:
        super().__init__(overlap_strategy=overlap_strategy)
        self.kind = kind

    def _apply(
        self,
        t: np.ndarray,
        x: np.ndarray,
        t_query: np.ndarray,
    ) -> np.ndarray:
        n_dims = x.shape[1]
        return np.column_stack(
            [
                sp_interpolate.interp1d(
                    t, x[:, d], kind=self.kind, fill_value="extrapolate"
                )(t_query)
                for d in range(n_dims)
            ]
        )


class ChunkFilterPolynomial(ChunkFilterBase):
    """Least-squares polynomial fit chunk filter.

    Parameters
    ----------
    overlap_strategy : str, optional
        Overlap handling strategy (default: ``'latest'``).
    degree : int, optional
        Polynomial degree (default: ``3``).
    """

    def __init__(
        self,
        overlap_strategy: str = "latest",
        degree: int = 3,
    ) -> None:
        super().__init__(overlap_strategy=overlap_strategy)
        self.degree = degree

    def _apply(
        self,
        t: np.ndarray,
        x: np.ndarray,
        t_query: np.ndarray,
    ) -> np.ndarray:
        n_dims = x.shape[1]
        degree = min(self.degree, len(t) - 1)
        return np.column_stack(
            [
                np.polyval(np.polyfit(t, x[:, d], degree), t_query)
                for d in range(n_dims)
            ]
        )


class ChunkFilterSavgol(ChunkFilterBase):
    """Savitzky-Golay smoothing chunk filter.

    Applies a Savitzky-Golay filter to the merged data, then interpolates
    the result at the query timestamps.

    Parameters
    ----------
    overlap_strategy : str, optional
        Overlap handling strategy (default: ``'latest'``).
    window_length : int, optional
        Filter window length (must be odd, default: ``11``).
    polyorder : int, optional
        Polynomial order for the local fit (default: ``3``).
    """

    def __init__(
        self,
        overlap_strategy: str = "latest",
        window_length: int = 11,
        polyorder: int = 3,
    ) -> None:
        super().__init__(overlap_strategy=overlap_strategy)
        self.window_length = window_length
        self.polyorder = polyorder

    def _apply(
        self,
        t: np.ndarray,
        x: np.ndarray,
        t_query: np.ndarray,
    ) -> np.ndarray:
        n_dims = x.shape[1]
        wl = min(self.window_length, len(t))
        if wl % 2 == 0:
            wl = max(wl - 1, 1)
        po = min(self.polyorder, wl - 1)
        smoothed = np.column_stack(
            [sp_signal.savgol_filter(x[:, d], wl, po) for d in range(n_dims)]
        )
        return np.column_stack(
            [np.interp(t_query, t, smoothed[:, d]) for d in range(n_dims)]
        )


class ChunkFilterGaussian(ChunkFilterBase):
    """Gaussian smoothing chunk filter.

    Applies a 1-D Gaussian convolution to the merged data, then
    interpolates the result at the query timestamps.

    Parameters
    ----------
    overlap_strategy : str, optional
        Overlap handling strategy (default: ``'latest'``).
    sigma : float, optional
        Standard deviation of the Gaussian kernel in samples
        (default: ``3.0``).
    """

    def __init__(
        self,
        overlap_strategy: str = "latest",
        sigma: float = 3.0,
    ) -> None:
        super().__init__(overlap_strategy=overlap_strategy)
        self.sigma = sigma

    def _apply(
        self,
        t: np.ndarray,
        x: np.ndarray,
        t_query: np.ndarray,
    ) -> np.ndarray:
        n_dims = x.shape[1]
        smoothed = np.column_stack(
            [gaussian_filter1d(x[:, d], self.sigma) for d in range(n_dims)]
        )
        return np.column_stack(
            [np.interp(t_query, t, smoothed[:, d]) for d in range(n_dims)]
        )


class ChunkFilterLowpass(ChunkFilterBase):
    """Zero-phase Butterworth low-pass chunk filter.

    Applies a forward-backward (zero-phase) Butterworth low-pass filter
    to the merged data, then interpolates at the query timestamps.

    Parameters
    ----------
    overlap_strategy : str, optional
        Overlap handling strategy (default: ``'latest'``).
    cutoff_freq : float, optional
        Cutoff frequency in the same units as ``sample_rate``
        (default: ``5.0``).
    sample_rate : float, optional
        Sampling rate (default: ``100.0``).
    order : int, optional
        Filter order (default: ``4``).
    """

    def __init__(
        self,
        overlap_strategy: str = "latest",
        cutoff_freq: float = 5.0,
        sample_rate: float = 100.0,
        order: int = 4,
    ) -> None:
        super().__init__(overlap_strategy=overlap_strategy)
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        self.order = order
        nyq = 0.5 * float(sample_rate)
        wn = float(cutoff_freq) / nyq
        if not (0.0 < wn < 1.0):
            raise ValueError(
                f"Normalised cutoff frequency must be in (0, 1); got {wn:.4f}. "
                "Check cutoff_freq and sample_rate values."
            )

    def _apply(
        self,
        t: np.ndarray,
        x: np.ndarray,
        t_query: np.ndarray,
    ) -> np.ndarray:
        n_dims = x.shape[1]
        nyq = 0.5 * self.sample_rate
        wn = self.cutoff_freq / nyq
        b, a = sp_signal.butter(self.order, wn, btype="low")
        smoothed = np.column_stack(
            [sp_signal.filtfilt(b, a, x[:, d]) for d in range(n_dims)]
        )
        return np.column_stack(
            [np.interp(t_query, t, smoothed[:, d]) for d in range(n_dims)]
        )


class ChunkFilterMedian(ChunkFilterBase):
    """Median filter chunk filter.

    Applies a 1-D median filter to the merged data, then interpolates
    the result at the query timestamps.

    Parameters
    ----------
    overlap_strategy : str, optional
        Overlap handling strategy (default: ``'latest'``).
    kernel_size : int, optional
        Median filter kernel size (must be odd, default: ``5``).
    """

    def __init__(
        self,
        overlap_strategy: str = "latest",
        kernel_size: int = 5,
    ) -> None:
        super().__init__(overlap_strategy=overlap_strategy)
        self.kernel_size = kernel_size

    def _apply(
        self,
        t: np.ndarray,
        x: np.ndarray,
        t_query: np.ndarray,
    ) -> np.ndarray:
        n_dims = x.shape[1]
        smoothed = np.column_stack(
            [_nd_median(x[:, d], size=self.kernel_size) for d in range(n_dims)]
        )
        return np.column_stack(
            [np.interp(t_query, t, smoothed[:, d]) for d in range(n_dims)]
        )


class ChunkFilterFIR(ChunkFilterBase):
    """Zero-phase FIR chunk filter.

    Designs an FIR filter with :func:`scipy.signal.firwin` and applies
    it with zero-phase filtering on the merged data.

    Parameters
    ----------
    overlap_strategy : str, optional
        Overlap handling strategy (default: ``'latest'``).
    numtaps : int, optional
        FIR filter length (default: ``31``).
    cutoff_freq : float or list of float, optional
        Cutoff frequency in same units as ``sample_rate`` (default: ``5.0``).
    sample_rate : float, optional
        Sampling rate (default: ``100.0``).
    window : str, optional
        Window function (default: ``'hamming'``).
    pass_zero : bool or str, optional
        If ``True``, the DC component passes (default: ``True``).
    """

    def __init__(
        self,
        overlap_strategy: str = "latest",
        numtaps: int = 31,
        cutoff_freq: float | list[float] = 5.0,
        sample_rate: float = 100.0,
        window: str = "hamming",
        pass_zero: bool | str = True,
    ) -> None:
        super().__init__(overlap_strategy=overlap_strategy)
        self._b = sp_signal.firwin(
            numtaps, cutoff_freq, fs=sample_rate,
            window=window, pass_zero=pass_zero,
        )

    def _apply(
        self,
        t: np.ndarray,
        x: np.ndarray,
        t_query: np.ndarray,
    ) -> np.ndarray:
        n_dims = x.shape[1]
        smoothed = np.column_stack(
            [sp_signal.filtfilt(self._b, [1.0], x[:, d]) for d in range(n_dims)]
        )
        return np.column_stack(
            [np.interp(t_query, t, smoothed[:, d]) for d in range(n_dims)]
        )


class ChunkFilterIIR(ChunkFilterBase):
    """Zero-phase IIR chunk filter with selectable filter family.

    Supports Butterworth, Chebyshev Type I/II, Elliptic, and Bessel
    filters.  Applied with :func:`scipy.signal.sosfiltfilt` for
    zero-phase filtering on the merged data.

    Parameters
    ----------
    overlap_strategy : str, optional
        Overlap handling strategy (default: ``'latest'``).
    cutoff_freq : float or list of float, optional
        Cutoff frequency (default: ``5.0``).
    sample_rate : float, optional
        Sampling rate (default: ``100.0``).
    order : int, optional
        Filter order (default: ``4``).
    iir_type : str, optional
        IIR family: ``'butterworth'``, ``'chebyshev1'``, ``'chebyshev2'``,
        ``'elliptic'``, ``'bessel'`` (default: ``'butterworth'``).
    btype : str, optional
        Band type: ``'low'``, ``'high'``, ``'bandpass'``, ``'bandstop'``
        (default: ``'low'``).
    rp : float, optional
        Max passband ripple (dB) for Chebyshev I / Elliptic.
    rs : float, optional
        Min stopband attenuation (dB) for Chebyshev II / Elliptic.
    """

    def __init__(
        self,
        overlap_strategy: str = "latest",
        cutoff_freq: float | list[float] = 5.0,
        sample_rate: float = 100.0,
        order: int = 4,
        iir_type: str = "butterworth",
        btype: str = "low",
        rp: float | None = None,
        rs: float | None = None,
    ) -> None:
        super().__init__(overlap_strategy=overlap_strategy)
        from python_filter_smoothing.offline import _design_iir_sos

        self._sos = _design_iir_sos(
            cutoff_freq, sample_rate, order, iir_type, btype, rp, rs,
        )

    def _apply(
        self,
        t: np.ndarray,
        x: np.ndarray,
        t_query: np.ndarray,
    ) -> np.ndarray:
        n_dims = x.shape[1]
        smoothed = np.column_stack(
            [sp_signal.sosfiltfilt(self._sos, x[:, d]) for d in range(n_dims)]
        )
        return np.column_stack(
            [np.interp(t_query, t, smoothed[:, d]) for d in range(n_dims)]
        )


class ChunkFilterKalman(ChunkFilterBase):
    """Kalman smoother (RTS) chunk filter.

    Applies a forward Kalman filter + backward RTS smoother to the
    merged chunk data for optimal offline estimation.

    Parameters
    ----------
    overlap_strategy : str, optional
        Overlap handling strategy (default: ``'latest'``).
    process_noise : float, optional
        Process noise variance for built-in models (default: ``0.01``).
    measurement_noise : float, optional
        Measurement noise variance for built-in models (default: ``0.1``).
    state_model : str, optional
        ``'position'`` or ``'position_velocity'`` (default: ``'position'``).
    dt : float, optional
        Time step for velocity model.  If ``None``, estimated from data.
    F, H, Q, R : ndarray, optional
        Custom Kalman filter matrices.
    """

    def __init__(
        self,
        overlap_strategy: str = "latest",
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        state_model: str = "position",
        dt: float | None = None,
        F: np.ndarray | None = None,
        H: np.ndarray | None = None,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
    ) -> None:
        super().__init__(overlap_strategy=overlap_strategy)
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise
        self._state_model = state_model
        self._dt = dt
        self._custom_F = F
        self._custom_H = H
        self._custom_Q = Q
        self._custom_R = R

    def _apply(
        self,
        t: np.ndarray,
        x: np.ndarray,
        t_query: np.ndarray,
    ) -> np.ndarray:
        from python_filter_smoothing.offline import OfflineFilter

        filt = OfflineFilter(t, x)
        smoothed = filt.kalman_smooth(
            process_noise=self._process_noise,
            measurement_noise=self._measurement_noise,
            state_model=self._state_model,
            dt=self._dt,
            F=self._custom_F,
            H=self._custom_H,
            Q=self._custom_Q,
            R=self._custom_R,
        )
        if smoothed.ndim == 1:
            smoothed = smoothed[:, np.newaxis]
        n_dims = smoothed.shape[1]
        return np.column_stack(
            [np.interp(t_query, t, smoothed[:, d]) for d in range(n_dims)]
        )


# ======================================================================
# Factory function
# ======================================================================

_METHODS = {
    "linear": ChunkFilterLinear,
    "spline": ChunkFilterSpline,
    "polynomial": ChunkFilterPolynomial,
    "savgol": ChunkFilterSavgol,
    "gaussian": ChunkFilterGaussian,
    "lowpass": ChunkFilterLowpass,
    "median": ChunkFilterMedian,
    "fir": ChunkFilterFIR,
    "iir": ChunkFilterIIR,
    "kalman": ChunkFilterKalman,
}


def ChunkFilter(
    method: str = "linear",
    overlap_strategy: str = "latest",
    **kwargs,
) -> ChunkFilterBase:
    """Create a chunk filter for the given interpolation method.

    This factory function provides backward-compatible construction.

    Parameters
    ----------
    method : str, optional
        Smoothing / interpolation method.  One of ``'linear'``,
        ``'spline'``, ``'polynomial'``, ``'savgol'``, ``'gaussian'``,
        ``'lowpass'``, ``'median'``, ``'fir'``, ``'iir'``, ``'kalman'``
        (default: ``'linear'``).
    overlap_strategy : str, optional
        How to handle overlapping timestamps (default: ``'latest'``).
    **kwargs
        Extra keyword arguments forwarded to the subclass constructor
        (e.g. ``kind`` for spline, ``degree`` for polynomial).

    Returns
    -------
    ChunkFilterBase
        An instance of the appropriate subclass.

    Raises
    ------
    ValueError
        If *method* is not one of the supported methods.
    """
    cls = _METHODS.get(method)
    if cls is None:
        available = ", ".join(f"'{m}'" for m in _METHODS)
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {available}."
        )
    return cls(overlap_strategy=overlap_strategy, **kwargs)
