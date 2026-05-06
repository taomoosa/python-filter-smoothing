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

    Variable sampling periods are supported: :meth:`update` computes the
    elapsed time since the previous call and passes it as *dt* to
    :meth:`_update_impl`, allowing each subclass to adapt its parameters
    accordingly.
    """

    def __init__(self, **kwargs) -> None:  # noqa: ARG002
        self._dim: Optional[int] = None
        self._state: Optional[np.ndarray] = None
        self._prev_t: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, t: float, x) -> np.ndarray:
        """Process a new sample and return the filtered value.

        Parameters
        ----------
        t : float
            Timestamp of the new sample (seconds).  The elapsed time
            *dt = t - t_prev* is computed automatically and forwarded to
            :meth:`_update_impl`, enabling each subclass to handle
            variable sampling periods.
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
            self._prev_t = float(t)
            self._init_impl(x)
        else:
            dt = float(t) - self._prev_t
            if dt <= 0.0:
                dt = 1e-9
            self._update_impl(x, dt)
            self._prev_t = float(t)
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
        self._prev_t = None
        self._reset_impl()

    # ------------------------------------------------------------------
    # Abstract hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _init_impl(self, x: np.ndarray) -> None:
        """Initialise method-specific state on the first sample."""

    @abstractmethod
    def _update_impl(self, x: np.ndarray, dt: float) -> None:
        """Update the filter with a new sample (called after the first).

        Parameters
        ----------
        x : np.ndarray
            New observation.
        dt : float
            Elapsed time since the previous sample (always > 0).
        """

    @abstractmethod
    def _reset_impl(self) -> None:
        """Reset method-specific state."""


# ======================================================================
# Subclasses
# ======================================================================


class OnlineFilterEMA(OnlineFilterBase):
    """Exponential moving average filter.

    The smoothing factor is adapted to the actual sampling interval so
    that the filter's time constant is preserved when the period varies.
    Internally, ``alpha`` and ``dt_nominal`` are converted to an RC time
    constant ``τ = -dt_nominal / ln(1 - alpha)``.  At each update the
    effective alpha is recomputed as ``1 - exp(-dt / τ)``.

    When the period equals *dt_nominal* the filter behaves identically to
    a fixed-rate EMA with the given ``alpha``, ensuring backward
    compatibility.

    Parameters
    ----------
    alpha : float, optional
        Smoothing factor in ``(0, 1]`` at the nominal sampling period.
        Higher values weight recent samples more heavily (default: ``0.3``).
    dt_nominal : float, optional
        Nominal sampling interval (seconds) that ``alpha`` was designed
        for (default: ``1.0``).
    """

    def __init__(self, alpha: float = 0.3, dt_nominal: float = 1.0) -> None:
        super().__init__()
        self._alpha = float(alpha)
        self._dt_nominal = float(dt_nominal)
        if self._alpha >= 1.0:
            self._tau = 0.0
        else:
            self._tau = -float(dt_nominal) / np.log(1.0 - self._alpha)

    def _init_impl(self, x: np.ndarray) -> None:  # noqa: ARG002
        pass  # state already set to first sample by base class

    def _update_impl(self, x: np.ndarray, dt: float) -> None:
        if self._tau <= 0.0:
            self._state = x.copy()
        else:
            alpha_t = 1.0 - np.exp(-dt / self._tau)
            self._state = alpha_t * x + (1.0 - alpha_t) * self._state

    def _reset_impl(self) -> None:
        pass


class OnlineFilterMovingAverage(OnlineFilterBase):
    """Simple sliding-window average filter.

    Two windowing modes are available:

    * **Sample-based** (default): keep the last ``window`` samples.
    * **Time-based**: keep all samples within the most recent
      ``window_time`` seconds.  When ``window_time`` is given,
      ``window`` is ignored and the effective number of averaged
      samples adapts automatically to the actual sampling rate,
      making the filter correct under variable periods.

    Parameters
    ----------
    window : int, optional
        Window length in samples (default: ``10``).  Used only when
        ``window_time`` is ``None``.
    window_time : float, optional
        Time-based window length in seconds.  When provided, only
        samples with timestamps within ``[t_now - window_time, t_now]``
        are averaged (default: ``None``).
    """

    def __init__(
        self,
        window: int = 10,
        window_time: float | None = None,
    ) -> None:
        super().__init__()
        self._window_size = int(window)
        self._window_time = float(window_time) if window_time is not None else None
        self._buffer: deque | None = None
        # Time-based: list of (timestamp, value) pairs
        self._t_buffer: list | None = None

    def _init_impl(self, x: np.ndarray) -> None:
        if self._window_time is not None:
            self._t_buffer = [(self._prev_t, x.copy())]
        else:
            self._buffer = deque(maxlen=self._window_size)
            self._buffer.append(x.copy())

    def _update_impl(self, x: np.ndarray, dt: float) -> None:
        if self._window_time is not None:
            t_now = self._prev_t + dt
            self._t_buffer.append((t_now, x.copy()))
            cutoff = t_now - self._window_time
            self._t_buffer = [(t, v) for t, v in self._t_buffer if t >= cutoff]
            self._state = np.mean(
                np.stack([v for _, v in self._t_buffer]), axis=0
            )
        else:
            self._buffer.append(x.copy())
            self._state = np.mean(np.stack(list(self._buffer)), axis=0)

    def _reset_impl(self) -> None:
        self._buffer = None
        self._t_buffer = None


class OnlineFilterLowpass(OnlineFilterBase):
    """Causal Butterworth low-pass filter with variable-period support.

    Internally the filter is designed as a continuous-time analog
    Butterworth prototype and discretized at each step using the
    **zero-order hold (ZOH)** method.  When the sampling interval is
    constant the result is equivalent to the standard digital design;
    when it varies the filter adapts correctly by re-discretizing with
    the actual elapsed time.

    Discretized matrices are cached so that re-discretization only
    occurs when the elapsed time changes, keeping the per-sample cost
    to O(n²) in the filter order (matrix–vector products).

    Parameters
    ----------
    cutoff_freq : float, optional
        Cutoff frequency in Hz (same units as ``sample_rate``)
        (default: ``0.1``).
    sample_rate : float, optional
        Nominal sampling rate in Hz, used for Nyquist validation and
        for the steady-state initial conditions (default: ``1.0``).
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
        sample_rate = float(sample_rate)
        cutoff_freq = float(cutoff_freq)
        if not (0.0 < cutoff_freq < 0.5 * sample_rate):
            raise ValueError(
                f"cutoff_freq must be in (0, {0.5 * sample_rate}); "
                f"got {cutoff_freq:.4f}."
            )
        # Analog prototype in rad/s
        w0 = 2.0 * np.pi * cutoff_freq
        b_a, a_a = sp_signal.butter(int(order), w0, btype="low", analog=True)
        self._A_c, self._B_c, self._C_c, self._D_c = sp_signal.tf2ss(b_a, a_a)
        self._dt_nominal = 1.0 / sample_rate
        # ZOH cache
        self._dt_cached: float | None = None
        self._A_d: np.ndarray | None = None
        self._B_d: np.ndarray | None = None
        self._C_d: np.ndarray | None = None
        self._D_d: np.ndarray | None = None
        self._x_state: np.ndarray | None = None

    def _get_discrete(self, dt: float) -> None:
        if dt != self._dt_cached:
            res = sp_signal.cont2discrete(
                (self._A_c, self._B_c, self._C_c, self._D_c), dt, method="zoh",
            )
            self._A_d, self._B_d, self._C_d, self._D_d = (
                res[0], res[1], res[2], res[3],
            )
            self._dt_cached = dt

    def _init_impl(self, x: np.ndarray) -> None:
        self._get_discrete(self._dt_nominal)
        n = self._A_d.shape[0]
        self._x_state = np.zeros((n, self._dim))
        # Steady-state: x_ss = (I - A_d)^{-1} B_d * u
        try:
            x_ss_1d = np.linalg.solve(np.eye(n) - self._A_d, self._B_d.ravel())
            for d in range(self._dim):
                self._x_state[:, d] = x_ss_1d * x[d]
        except np.linalg.LinAlgError:
            pass  # fall back to zero initial state

    def _update_impl(self, x: np.ndarray, dt: float) -> None:
        self._get_discrete(dt)
        B_flat = self._B_d.ravel()
        D_val = self._D_d.item()
        for d in range(self._dim):
            # Integrate state first (ZOH: hold u[k] over [t_{k-1}, t_k]),
            # then read output from the updated state.  This avoids the
            # time-varying one-step lag that the reversed order introduces
            # when dt is non-uniform.
            self._x_state[:, d] = self._A_d @ self._x_state[:, d] + B_flat * x[d]
            self._state[d] = (self._C_d @ self._x_state[:, d]).item() + D_val * x[d]

    def _reset_impl(self) -> None:
        self._dt_cached = None
        self._x_state = None


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

    def _update_impl(self, x: np.ndarray, dt: float) -> None:  # noqa: ARG002
        # Variable-period logic is handled in the overridden update() below.
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

    .. note::
        FIR filters operate on a fixed sample-count window with
        pre-designed coefficients, so the frequency response is only
        correct when the sampling period is constant and equals
        ``1 / sample_rate``.  For variable-period data, consider
        :class:`OnlineFilterIIR`, :class:`OnlineFilterLowpass`, or
        :class:`OnlineFilterOneEuro` instead.

    Parameters
    ----------
    numtaps : int
        Length of the FIR filter (number of coefficients).
    cutoff_freq : float or list of float
        Cutoff frequency (or frequencies) in the same units as
        ``sample_rate``.
    sample_rate : float
        Nominal sampling rate of the data.
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

    def _update_impl(self, x: np.ndarray, dt: float) -> None:  # noqa: ARG002
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
    """Causal IIR filter with selectable filter family and variable-period support.

    Supports Butterworth, Chebyshev Type I/II, Elliptic, and Bessel
    filters.  Like :class:`OnlineFilterLowpass`, the filter is designed
    as a continuous-time analog prototype and re-discretized at each
    step via the **ZOH method**, so that variable sampling intervals are
    handled correctly.  Discretized matrices are cached and recomputed
    only when the elapsed time changes.

    Parameters
    ----------
    cutoff_freq : float or list of float
        Cutoff frequency (or frequencies for band filters) in the same
        units as ``sample_rate``.
    sample_rate : float
        Nominal sampling rate of the data, used for Nyquist validation
        and steady-state initialisation.
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
        from .offline import _design_iir_analog_ss

        self._A_c, self._B_c, self._C_c, self._D_c = _design_iir_analog_ss(
            cutoff_freq, order, iir_type, btype, rp, rs,
        )
        self._dt_nominal = 1.0 / float(sample_rate)
        # ZOH cache
        self._dt_cached: float | None = None
        self._A_d: np.ndarray | None = None
        self._B_d: np.ndarray | None = None
        self._C_d: np.ndarray | None = None
        self._D_d: np.ndarray | None = None
        self._x_state: np.ndarray | None = None

    def _get_discrete(self, dt: float) -> None:
        if dt != self._dt_cached:
            res = sp_signal.cont2discrete(
                (self._A_c, self._B_c, self._C_c, self._D_c), dt, method="zoh",
            )
            self._A_d, self._B_d, self._C_d, self._D_d = (
                res[0], res[1], res[2], res[3],
            )
            self._dt_cached = dt

    def _init_impl(self, x: np.ndarray) -> None:
        self._get_discrete(self._dt_nominal)
        n = self._A_d.shape[0]
        self._x_state = np.zeros((n, self._dim))
        try:
            x_ss_1d = np.linalg.solve(np.eye(n) - self._A_d, self._B_d.ravel())
            for d in range(self._dim):
                self._x_state[:, d] = x_ss_1d * x[d]
        except np.linalg.LinAlgError:
            pass

    def _update_impl(self, x: np.ndarray, dt: float) -> None:
        self._get_discrete(dt)
        B_flat = self._B_d.ravel()
        D_val = self._D_d.item()
        for d in range(self._dim):
            self._x_state[:, d] = self._A_d @ self._x_state[:, d] + B_flat * x[d]
            self._state[d] = (self._C_d @ self._x_state[:, d]).item() + D_val * x[d]

    def _reset_impl(self) -> None:
        self._dt_cached = None
        self._x_state = None


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

    def _update_impl(self, x: np.ndarray, dt: float) -> None:
        F, H, Q, R = self._F_mat, self._H_mat, self._Q_mat, self._R_mat

        # For the built-in position_velocity model, rebuild F and Q with
        # the actual elapsed time so that variable sampling periods are
        # handled correctly.
        if self._state_model == "position_velocity" and self._custom_F is None:
            D = self._dim
            F = self._F_mat.copy()
            F[:D, D:] = np.eye(D) * dt
            q = self._process_noise
            Q = np.zeros((self._S_dim, self._S_dim))
            Q[:D, :D] = np.eye(D) * (dt**4 / 4) * q
            Q[:D, D:] = np.eye(D) * (dt**3 / 2) * q
            Q[D:, :D] = np.eye(D) * (dt**3 / 2) * q
            Q[D:, D:] = np.eye(D) * (dt**2) * q

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
