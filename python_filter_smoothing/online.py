"""Online (sample-by-sample) time series filtering and smoothing.

Data points are provided one at a time in chronological order.
Internal state is maintained between calls so filtering is causal.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
from scipy import signal as sp_signal


class OnlineFilter:
    """Filter and smoother for online (sample-by-sample) time series processing.

    Data points are provided one at a time in chronological order.
    The filter maintains internal state between :meth:`update` calls.

    Parameters
    ----------
    method : str
        Filtering method.  One of:

        - ``'ema'``            – Exponential moving average.
        - ``'moving_average'`` – Simple sliding-window average.
        - ``'lowpass'``        – Causal IIR Butterworth low-pass filter.

    **kwargs
        Method-specific keyword arguments.

        For ``'ema'``:
            alpha : float, optional
                Smoothing factor in ``(0, 1]``.  Higher values weight recent
                samples more heavily (default: ``0.3``).

        For ``'moving_average'``:
            window : int, optional
                Window length in samples (default: ``10``).

        For ``'lowpass'``:
            cutoff_freq : float, optional
                Cutoff frequency relative to the Nyquist frequency
                (default: ``0.1``).
            sample_rate : float, optional
                Sampling rate used to normalise ``cutoff_freq``
                (default: ``1.0``).
            order : int, optional
                Filter order (default: ``2``).

    Raises
    ------
    ValueError
        If an unsupported method name is given.
    """

    def __init__(self, method: str = "ema", **kwargs) -> None:
        self.method = method
        self._dim: Optional[int] = None
        self._state: Optional[np.ndarray] = None

        if method == "ema":
            self._alpha = float(kwargs.get("alpha", 0.3))
        elif method == "moving_average":
            self._window_size = int(kwargs.get("window", 10))
            self._buffer: Optional[deque] = None
        elif method == "lowpass":
            cutoff = float(kwargs.get("cutoff_freq", 0.1))
            sr = float(kwargs.get("sample_rate", 1.0))
            order = int(kwargs.get("order", 2))
            nyq = 0.5 * sr
            wn = cutoff / nyq
            self._b, self._a = sp_signal.butter(order, wn, btype="low")
            self._zi: Optional[np.ndarray] = None
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose from: 'ema', 'moving_average', 'lowpass'."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, t: float, x) -> np.ndarray:
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
            self._init(x)

        if self.method == "ema":
            self._state = self._alpha * x + (1.0 - self._alpha) * self._state
        elif self.method == "moving_average":
            self._buffer.append(x.copy())
            self._state = np.mean(np.stack(list(self._buffer)), axis=0)
        elif self.method == "lowpass":
            for d in range(self._dim):
                y, zi_new = sp_signal.lfilter(
                    self._b, self._a, [x[d]], zi=self._zi[:, d]
                )
                self._state[d] = y[0]
                self._zi[:, d] = zi_new

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
        if self.method == "moving_average":
            self._buffer = None
        elif self.method == "lowpass":
            self._zi = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init(self, x: np.ndarray) -> None:
        """Initialise internal state on the first sample."""
        self._dim = x.size
        self._state = x.copy()

        if self.method == "moving_average":
            self._buffer = deque(maxlen=self._window_size)
            self._buffer.append(x.copy())
        elif self.method == "lowpass":
            n_states = max(len(self._b), len(self._a)) - 1
            self._zi = np.zeros((n_states, self._dim))
            # Initialise each channel to steady-state for the first sample
            zi_1d = sp_signal.lfilter_zi(self._b, self._a)
            for d in range(self._dim):
                self._zi[:, d] = zi_1d * x[d]
