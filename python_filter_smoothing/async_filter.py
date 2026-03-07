"""Asynchronous time series filtering and smoothing.

Data points (or small batches) arrive at random, irregular intervals.
The filter maintains a thread-safe circular buffer and provides filtered
output that can be queried independently of the input rate.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Optional

import numpy as np
from scipy import interpolate as sp_interpolate


class AsyncFilter:
    """Filter and smoother for asynchronous time series processing.

    Data points arrive at irregular, random intervals from one or more
    producer threads.  Filtered output can be queried at any time and at
    an independent rate from a consumer thread.

    Parameters
    ----------
    method : str, optional
        Smoothing method.  One of:

        - ``'ema'``    – Exponential moving average (default).  Provides a
                         running estimate updated with every new sample.
        - ``'linear'`` – Linear interpolation over the circular buffer.
        - ``'spline'`` – Cubic-spline interpolation over the buffer.

    buffer_size : int, optional
        Maximum number of ``(timestamp, value)`` pairs held in the
        internal circular buffer (default: ``100``).

    **kwargs
        Method-specific keyword arguments.

        For ``'ema'``:
            alpha : float, optional
                Smoothing factor in ``(0, 1]``.  Higher values weight
                recent samples more heavily (default: ``0.3``).

    Notes
    -----
    All public methods are **thread-safe**.  Multiple producer threads
    can call :meth:`update` concurrently while a consumer thread calls
    :meth:`get_output`.

    Raises
    ------
    ValueError
        If an unsupported method name is given.
    """

    def __init__(
        self,
        method: str = "ema",
        buffer_size: int = 100,
        **kwargs,
    ) -> None:
        if method not in ("ema", "linear", "spline"):
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose from: 'ema', 'linear', 'spline'."
            )

        self.method = method
        self._buffer_size = buffer_size
        self._lock = threading.Lock()

        self._t_buf: deque[float] = deque(maxlen=buffer_size)
        self._x_buf: deque[np.ndarray] = deque(maxlen=buffer_size)

        self._dim: Optional[int] = None
        self._ema_state: Optional[np.ndarray] = None

        if method == "ema":
            self._alpha = float(kwargs.get("alpha", 0.3))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, t: float, x) -> None:
        """Add a new data point (thread-safe).

        Parameters
        ----------
        t : float
            Timestamp of the new sample.
        x : scalar or array-like
            Observed value.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        with self._lock:
            if self._dim is None:
                self._dim = x.size
                if self.method == "ema":
                    self._ema_state = x.copy()

            if self.method == "ema":
                self._ema_state = (
                    self._alpha * x + (1.0 - self._alpha) * self._ema_state
                )

            self._t_buf.append(float(t))
            self._x_buf.append(x.copy())

    def get_output(self, t: Optional[float] = None) -> Optional[np.ndarray]:
        """Return the current filtered output (thread-safe).

        Parameters
        ----------
        t : float, optional
            Timestamp at which to evaluate the filter.  This parameter is
            only meaningful for interpolation-based methods (``'linear'``
            and ``'spline'``); it is ignored for ``'ema'``, which always
            returns the running estimate.  If ``t`` is ``None``, the
            timestamp of the latest buffered sample is used.

        Returns
        -------
        np.ndarray or None
            Filtered value, or ``None`` if no data has been received yet.
        """
        with self._lock:
            if self._dim is None:
                return None

            if self.method == "ema":
                return self._ema_state.copy()

            # Interpolation-based methods need at least two buffered points.
            n = len(self._t_buf)
            if n == 1:
                return self._x_buf[-1].copy()
            if n == 0:
                return None

            t_arr = np.array(self._t_buf)
            x_arr = np.stack(list(self._x_buf))
            t_query = t_arr[-1] if t is None else float(t)
            return self._interpolate(t_arr, x_arr, t_query)

    def clear(self) -> None:
        """Clear all buffered data and reset internal state (thread-safe)."""
        with self._lock:
            self._t_buf.clear()
            self._x_buf.clear()
            self._dim = None
            self._ema_state = None

    @property
    def buffer_length(self) -> int:
        """Number of samples currently in the buffer (thread-safe)."""
        with self._lock:
            return len(self._t_buf)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _interpolate(
        self,
        t: np.ndarray,
        x: np.ndarray,
        t_query: float,
    ) -> np.ndarray:
        """Evaluate the interpolated curve at a single query timestamp."""
        n_dims = x.shape[1]

        if self.method == "linear":
            return np.array(
                [float(np.interp(t_query, t, x[:, d])) for d in range(n_dims)]
            )

        # spline – fall back to linear if fewer than 4 points are buffered
        kind = "cubic" if len(t) >= 4 else "linear"
        return np.array(
            [
                float(
                    sp_interpolate.interp1d(
                        t, x[:, d], kind=kind, fill_value="extrapolate"
                    )(t_query)
                )
                for d in range(n_dims)
            ]
        )
