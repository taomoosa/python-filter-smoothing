"""Online (sample-by-sample) time series filtering and smoothing.

Data points are provided one at a time in chronological order.
Internal state is maintained between calls so filtering is causal.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

import numpy as np
from scipy import signal as sp_signal

__all__ = [
    "OnlineFilterBase",
    "OnlineFilterEMA",
    "OnlineFilterMovingAverage",
    "OnlineFilterLowpass",
    "OnlineFilterOneEuro",
    "OnlineFilterFIR",
    "OnlineFilterIIR",
    "OnlineFilterKalman",
    "OnlineFilter",
]

_AVAILABLE_METHODS = (
    "ema", "moving_average", "lowpass", "one_euro", "fir", "iir", "kalman",
)


# ======================================================================
# Base class
# ======================================================================


class OnlineFilterBase(ABC):
    """Abstract base for online (sample-by-sample) time series filters.

    Subclasses implement :meth:`_init_impl`, :meth:`_update_impl`, and
    :meth:`_reset_impl`.  All common bookkeeping (array conversion,
    first-sample initialisation, state copying) lives here.
    """

    def __init__(self, **kwargs) -> None:  # noqa: ARG002
        self._dim: Optional[int] = None
        self._state: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, t: float, x) -> np.ndarray:  # noqa: ARG002
        """Process a new sample and return the filtered value.

        Parameters
        ----------
        t : float
            Timestamp of the new sample (used for record-keeping).
        x : scalar or array-like
            Observed value.  Scalar and 1-D inputs are both accepted.

        Returns
        -------
        np.ndarray
            Filtered value with the same shape as the input ``x``.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        if self._dim is None:
            self._dim = x.size
            self._state = x.copy()
            self._init_impl(x)
        else:
            self._update_impl(x)
        return self._state.copy()

    def get_value(self) -> Optional[np.ndarray]:
        """Return the most recently computed filtered value.

        Returns
        -------
        np.ndarray or None
            Current filtered state, or ``None`` if :meth:`update` has not
            been called yet.
        """
        return self._state.copy() if self._state is not None else None

    def reset(self) -> None:
        """Reset the filter, clearing all history and internal state."""
        self._dim = None
        self._state = None
        self._reset_impl()

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _init_impl(self, x: np.ndarray) -> None:
        """Initialise method-specific state on the first sample."""

    @abstractmethod
    def _update_impl(self, x: np.ndarray) -> None:
        """Update the filter with a new sample (called after the first)."""

    @abstractmethod
    def _reset_impl(self) -> None:
        """Reset method-specific state."""


# ======================================================================
# Subclasses
# ======================================================================


class OnlineFilterEMA(OnlineFilterBase):
    """Exponential moving average filter.

    Parameters
    ----------
    alpha : float, optional
        Smoothing factor in ``(0, 1]``.  Higher values weight recent
        samples more heavily (default: ``0.3``).
    """

    def __init__(self, alpha: float = 0.3) -> None:
        super().__init__()
        self._alpha = float(alpha)

    def _init_impl(self, x: np.ndarray) -> None:  # noqa: ARG002
        pass  # state already set to first sample by base class

    def _update_impl(self, x: np.ndarray) -> None:
        self._state = self._alpha * x + (1.0 - self._alpha) * self._state

    def _reset_impl(self) -> None:
        pass


class OnlineFilterMovingAverage(OnlineFilterBase):
    """Simple sliding-window average filter.

    Parameters
    ----------
    window : int, optional
        Window length in samples (default: ``10``).
    """

    def __init__(self, window: int = 10) -> None:
        super().__init__()
        self._window_size = int(window)
        self._buffer: Optional[deque] = None

    def _init_impl(self, x: np.ndarray) -> None:
        self._buffer = deque(maxlen=self._window_size)
        self._buffer.append(x.copy())

    def _update_impl(self, x: np.ndarray) -> None:
        self._buffer.append(x.copy())
        self._state = np.mean(np.stack(list(self._buffer)), axis=0)

    def _reset_impl(self) -> None:
        self._buffer = None


class OnlineFilterLowpass(OnlineFilterBase):
    """Causal IIR Butterworth low-pass filter.

    Parameters
    ----------
    cutoff_freq : float, optional
        Cutoff frequency in the same units as ``sample_rate``
        (default: ``0.1``).  Internally normalised to the Nyquist
        frequency as ``cutoff_freq / (0.5 * sample_rate)``.
    sample_rate : float, optional
        Sampling rate used to normalise ``cutoff_freq``
        (default: ``1.0``).
    order : int, optional
        Filter order (default: ``2``).
    """

    def __init__(
        self,
        cutoff_freq: float = 0.1,
        sample_rate: float = 1.0,
        order: int = 2,
    ) -> None:
        super().__init__()
        nyq = 0.5 * float(sample_rate)
        wn = float(cutoff_freq) / nyq
        if not (0.0 < wn < 1.0):
            raise ValueError(
                f"Normalised cutoff frequency must be in (0, 1); got {wn:.4f}. "
                "Check cutoff_freq and sample_rate values."
            )
        self._b, self._a = sp_signal.butter(int(order), wn, btype="low")
        self._zi: Optional[np.ndarray] = None

    def _init_impl(self, x: np.ndarray) -> None:
        n_states = max(len(self._b), len(self._a)) - 1
        self._zi = np.zeros((n_states, self._dim))
        zi_1d = sp_signal.lfilter_zi(self._b, self._a)
        for d in range(self._dim):
            self._zi[:, d] = zi_1d * x[d]

    def _update_impl(self, x: np.ndarray) -> None:
        for d in range(self._dim):
            y, zi_new = sp_signal.lfilter(
                self._b, self._a, [x[d]], zi=self._zi[:, d]
            )
            self._state[d] = y[0]
            self._zi[:, d] = zi_new

    def _reset_impl(self) -> None:
        self._zi = None


class OnlineFilterOneEuro(OnlineFilterBase):
    """Adaptive low-pass filter (1€ filter, Casiez et al. CHI 2012).

    Dynamically adjusts the cutoff frequency based on the estimated speed
    of the signal: slow changes are smoothed aggressively while fast
    movements are tracked with low latency.

    Parameters
    ----------
    min_cutoff : float, optional
        Minimum cutoff frequency (Hz) applied when the signal is nearly
        stationary.  Lower values give more smoothing (default: ``1.0``).
    beta : float, optional
        Speed coefficient that controls how much the cutoff increases
        when the signal moves fast.  Higher values reduce lag during
        fast movements (default: ``0.0``).
    d_cutoff : float, optional
        Cutoff frequency (Hz) used to smooth the derivative estimate
        (default: ``1.0``).
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ) -> None:
        super().__init__()
        self._min_cutoff = float(min_cutoff)
        self._beta = float(beta)
        self._d_cutoff = float(d_cutoff)
        self._prev_t: float | None = None
        self._dx_state: np.ndarray | None = None

    @staticmethod
    def _smoothing_factor(te: float, cutoff: float) -> float:
        r = 2.0 * np.pi * cutoff * te
        return r / (r + 1.0)

    def _init_impl(self, x: np.ndarray) -> None:
        self._dx_state = np.zeros_like(x)

    def _update_impl(self, x: np.ndarray) -> None:
        # This is called from update() which receives t but base class
        # doesn't forward it.  We override update() instead.
        pass

    def update(self, t: float, x) -> np.ndarray:
        """Process a new sample and return the filtered value."""
        x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        if self._dim is None:
            self._dim = x.size
            self._state = x.copy()
            self._init_impl(x)
            self._prev_t = float(t)
            return self._state.copy()

        te = float(t) - self._prev_t
        if te <= 0.0:
            te = 1e-9

        # Smoothed derivative
        alpha_d = self._smoothing_factor(te, self._d_cutoff)
        dx = (x - self._state) / te
        self._dx_state = alpha_d * dx + (1.0 - alpha_d) * self._dx_state

        # Adaptive cutoff
        cutoff = self._min_cutoff + self._beta * np.abs(self._dx_state)

        # Per-dimension alpha (vector cutoff)
        alpha = np.array(
            [self._smoothing_factor(te, float(c)) for c in cutoff]
        )
        self._state = alpha * x + (1.0 - alpha) * self._state
        self._prev_t = float(t)
        return self._state.copy()

    def _reset_impl(self) -> None:
        self._prev_t = None
        self._dx_state = None


# ======================================================================
# FIR filter
# ======================================================================


class OnlineFilterFIR(OnlineFilterBase):
    """Causal FIR filter (sliding-window convolution).

    Designs an FIR filter with :func:`scipy.signal.firwin` and applies
    it causally by maintaining a buffer of the last ``numtaps`` samples.

    Parameters
    ----------
    numtaps : int
        Length of the FIR filter (number of coefficients).
    cutoff_freq : float or list of float
        Cutoff frequency (or frequencies) in the same units as
        ``sample_rate``.
    sample_rate : float
        Sampling rate of the data.
    window : str, optional
        Window function (default: ``'hamming'``).
    pass_zero : bool or str, optional
        If ``True``, the DC component passes through (default: ``True``).
    """

    def __init__(
        self,
        numtaps: int = 31,
        cutoff_freq: float = 5.0,
        sample_rate: float = 100.0,
        window: str = "hamming",
        pass_zero: bool | str = True,
    ) -> None:
        super().__init__()
        self._numtaps = int(numtaps)
        self._b = sp_signal.firwin(
            self._numtaps, cutoff_freq, fs=sample_rate,
            window=window, pass_zero=pass_zero,
        )
        self._buffer: deque | None = None

    def _init_impl(self, x: np.ndarray) -> None:
        self._buffer = deque(maxlen=self._numtaps)
        # Pre-fill with first sample to avoid startup transient
        for _ in range(self._numtaps):
            self._buffer.append(x.copy())

    def _update_impl(self, x: np.ndarray) -> None:
        self._buffer.append(x.copy())
        buf = np.array(list(self._buffer))  # (numtaps, D)
        # Convolution: y = sum(b[k] * x[n-k])
        for d in range(self._dim):
            self._state[d] = np.dot(self._b, buf[:, d])

    def _reset_impl(self) -> None:
        self._buffer = None


# ======================================================================
# General IIR filter (Butterworth, Chebyshev, Elliptic, Bessel)
# ======================================================================


class OnlineFilterIIR(OnlineFilterBase):
    """Causal IIR filter with selectable filter family.

    Supports Butterworth, Chebyshev Type I/II, Elliptic, and Bessel
    filters in SOS form for numerical stability.  Processes one sample
    at a time using :func:`scipy.signal.sosfilt` with maintained state.

    Parameters
    ----------
    cutoff_freq : float or list of float
        Cutoff frequency (or frequencies for band filters) in the same
        units as ``sample_rate``.
    sample_rate : float
        Sampling rate of the data.
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
        cutoff_freq: float | list[float] = 5.0,
        sample_rate: float = 100.0,
        order: int = 4,
        iir_type: str = "butterworth",
        btype: str = "low",
        rp: float | None = None,
        rs: float | None = None,
    ) -> None:
        super().__init__()
        from .offline import _design_iir_sos

        self._sos = _design_iir_sos(
            cutoff_freq, sample_rate, order, iir_type, btype, rp, rs,
        )
        self._zi: np.ndarray | None = None

    def _init_impl(self, x: np.ndarray) -> None:
        n_sections = self._sos.shape[0]
        # zi shape: (n_sections, 2) per dimension
        zi_1d = sp_signal.sosfilt_zi(self._sos)  # (n_sections, 2)
        self._zi = np.zeros((n_sections, 2, self._dim))
        for d in range(self._dim):
            self._zi[:, :, d] = zi_1d * x[d]

    def _update_impl(self, x: np.ndarray) -> None:
        for d in range(self._dim):
            y, zi_new = sp_signal.sosfilt(
                self._sos, [x[d]], zi=self._zi[:, :, d],
            )
            self._state[d] = y[0]
            self._zi[:, :, d] = zi_new

    def _reset_impl(self) -> None:
        self._zi = None


# ======================================================================
# Kalman filter (online, causal)
# ======================================================================


class OnlineFilterKalman(OnlineFilterBase):
    """Online Kalman filter for causal state estimation.

    Processes one measurement at a time, maintaining the state estimate
    and covariance.  Two built-in state models are provided; for full
    control, pass custom ``F``, ``H``, ``Q``, ``R`` matrices.

    Parameters
    ----------
    process_noise : float, optional
        Scalar process noise variance for built-in models (default: ``0.01``).
    measurement_noise : float, optional
        Scalar measurement noise variance for built-in models (default: ``0.1``).
    state_model : str, optional
        ``'position'`` (random walk) or ``'position_velocity'``
        (constant-velocity) (default: ``'position'``).
    dt : float, optional
        Time step for ``'position_velocity'`` model (default: ``0.01``).
    F : ndarray, optional
        Custom state transition matrix.
    H : ndarray, optional
        Custom observation matrix.
    Q : ndarray, optional
        Custom process noise covariance.
    R : ndarray, optional
        Custom measurement noise covariance.
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        state_model: str = "position",
        dt: float = 0.01,
        F: np.ndarray | None = None,
        H: np.ndarray | None = None,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise
        self._state_model = state_model
        self._dt = dt
        # Custom matrices (resolved at first update when dim is known)
        self._custom_F = F
        self._custom_H = H
        self._custom_Q = Q
        self._custom_R = R
        # Kalman state
        self._x_kal: np.ndarray | None = None
        self._P: np.ndarray | None = None
        self._F_mat: np.ndarray | None = None
        self._H_mat: np.ndarray | None = None
        self._Q_mat: np.ndarray | None = None
        self._R_mat: np.ndarray | None = None
        self._S_dim: int | None = None

    def _init_impl(self, x: np.ndarray) -> None:
        D = self._dim
        if (
            self._custom_F is not None
            and self._custom_H is not None
            and self._custom_Q is not None
            and self._custom_R is not None
        ):
            self._F_mat = np.asarray(self._custom_F, dtype=float)
            self._H_mat = np.asarray(self._custom_H, dtype=float)
            self._Q_mat = np.asarray(self._custom_Q, dtype=float)
            self._R_mat = np.asarray(self._custom_R, dtype=float)
            self._S_dim = self._F_mat.shape[0]
        else:
            from .offline import _build_kalman_model

            self._F_mat, self._H_mat, self._Q_mat, self._R_mat, self._S_dim = (
                _build_kalman_model(
                    D, self._dt, self._state_model,
                    self._process_noise, self._measurement_noise,
                )
            )

        S = self._S_dim
        self._x_kal = np.zeros(S)
        self._x_kal[:D] = x
        self._P = np.eye(S) * self._measurement_noise * 10.0

    def _update_impl(self, x: np.ndarray) -> None:
        F, H, Q, R = self._F_mat, self._H_mat, self._Q_mat, self._R_mat

        # Predict
        x_pred = F @ self._x_kal
        P_pred = F @ self._P @ F.T + Q

        # Update
        y_innov = x - H @ x_pred
        S_innov = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S_innov)
        self._x_kal = x_pred + K @ y_innov
        self._P = (np.eye(self._S_dim) - K @ H) @ P_pred

        # Output position components
        self._state[:] = self._x_kal[: self._dim]

    def _reset_impl(self) -> None:
        self._x_kal = None
        self._P = None


# ======================================================================
# Factory function
# ======================================================================


def OnlineFilter(method: str = "ema", **kwargs) -> OnlineFilterBase:
    """Create an online filter instance for the given *method*.

    This factory function provides backward-compatible construction:

    - ``OnlineFilter("ema", alpha=0.15)``
    - ``OnlineFilter("moving_average", window=20)``
    - ``OnlineFilter("lowpass", cutoff_freq=3.0, sample_rate=100.0, order=2)``

    Parameters
    ----------
    method : str
        Filtering method.  One of ``'ema'``, ``'moving_average'``,
        ``'lowpass'``, ``'one_euro'``, ``'fir'``, ``'iir'``, ``'kalman'``.
    **kwargs
        Forwarded to the corresponding subclass constructor.

    Returns
    -------
    OnlineFilterBase
        A concrete filter instance.

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    if method == "ema":
        return OnlineFilterEMA(**kwargs)
    if method == "moving_average":
        return OnlineFilterMovingAverage(**kwargs)
    if method == "lowpass":
        return OnlineFilterLowpass(**kwargs)
    if method == "one_euro":
        return OnlineFilterOneEuro(**kwargs)
    if method == "fir":
        return OnlineFilterFIR(**kwargs)
    if method == "iir":
        return OnlineFilterIIR(**kwargs)
    if method == "kalman":
        return OnlineFilterKalman(**kwargs)
    raise ValueError(
        f"Unknown method '{method}'. "
        f"Choose from: {', '.join(repr(m) for m in _AVAILABLE_METHODS)}."
    )
