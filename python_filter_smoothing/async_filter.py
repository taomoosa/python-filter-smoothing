"""Asynchronous time series filtering and smoothing.

Data points (or small batches) arrive at random, irregular intervals.
The filter maintains a thread-safe circular buffer and provides filtered
output that can be queried independently of the input rate.

Chunk-aware subclasses (:class:`AsyncFilterACT`, :class:`AsyncFilterRAIL`)
accept entire action chunks via :meth:`update_chunk` and apply specialised
temporal ensembling or trajectory fusion strategies.
"""
from __future__ import annotations

import abc
import threading
from collections import deque
from typing import Callable, Optional

import numpy as np
from scipy import interpolate as sp_interpolate


# ======================================================================
# Base class
# ======================================================================

class AsyncFilterBase(abc.ABC):
    """Abstract base for asynchronous time-series filters.

    Manages a thread-safe circular buffer of ``(timestamp, value)``
    pairs and delegates the actual computation to :meth:`_compute`.

    Parameters
    ----------
    buffer_size : int, optional
        Maximum number of samples held in the buffer (default: ``100``).

    Notes
    -----
    All public methods are **thread-safe**.
    """

    def __init__(self, buffer_size: int = 100) -> None:
        self._buffer_size = buffer_size
        self._lock = threading.Lock()
        self._t_buf: deque[float] = deque(maxlen=buffer_size)
        self._x_buf: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._dim: Optional[int] = None
        # Output history for stateful filters (EMA, OneEuro, MA) that need
        # to interpolate between recorded filter states.
        self._out_t_buf: deque[float] = deque(maxlen=buffer_size)
        self._out_x_buf: deque[np.ndarray] = deque(maxlen=buffer_size)

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
            t_float = float(t)
            if self._dim is None:
                self._dim = x.size
            x_arr = x.copy()
            self._t_buf.append(t_float)
            self._x_buf.append(x_arr)
            self._on_update(t_float, x_arr)

    def update_chunk(self, t, x) -> None:
        """Add a chunk of data points at once (thread-safe).

        The default implementation calls :meth:`update` for each sample.
        Chunk-aware subclasses (e.g. :class:`AsyncFilterACT`,
        :class:`AsyncFilterRAIL`) override this to perform chunk-level
        processing such as temporal ensembling or polynomial smoothing.

        Parameters
        ----------
        t : array-like, shape (N,)
            Timestamps for each sample in the chunk.
        x : array-like, shape (N,) or (N, D)
            Observed values.
        """
        t_arr = np.asarray(t, dtype=float).ravel()
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        if x_arr.shape[0] != len(t_arr):
            raise ValueError(
                f"t has {len(t_arr)} samples but x has {x_arr.shape[0]}."
            )
        for ti, xi in zip(t_arr, x_arr):
            self.update(float(ti), xi)

    def get_output(self, t: Optional[float] = None) -> Optional[np.ndarray]:
        """Return the current filtered output (thread-safe).

        Parameters
        ----------
        t : float, optional
            Timestamp at which to evaluate the filter.  Only meaningful
            for interpolation-based methods; ignored by ``'ema'``.
            If ``None``, the latest buffered timestamp is used.

        Returns
        -------
        np.ndarray or None
            Filtered value, or ``None`` if no data has been received yet.
        """
        with self._lock:
            if self._dim is None:
                return None

            n = len(self._t_buf)
            if n == 0:
                return None
            if n == 1:
                return self._x_buf[-1].copy()

            t_arr = np.array(self._t_buf)
            x_arr = np.stack(list(self._x_buf))
            t_query = t_arr[-1] if t is None else float(t)
            return self._compute(t_arr, x_arr, t_query)

    def clear(self) -> None:
        """Clear all buffered data and reset internal state (thread-safe)."""
        with self._lock:
            self._t_buf.clear()
            self._x_buf.clear()
            self._out_t_buf.clear()
            self._out_x_buf.clear()
            self._dim = None
            self._on_clear()

    @property
    def buffer_length(self) -> int:
        """Number of samples currently in the buffer (thread-safe)."""
        with self._lock:
            return len(self._t_buf)

    # ------------------------------------------------------------------
    # Hooks (override in subclasses as needed)
    # ------------------------------------------------------------------

    def _on_update(self, t: float, x: np.ndarray) -> None:  # noqa: D401
        """Called inside the lock after buffer append."""

    def _on_clear(self) -> None:
        """Called inside the lock after buffers and dim are reset."""

    # ------------------------------------------------------------------
    # Output history helpers (for stateful filters)
    # ------------------------------------------------------------------

    def _record_output(self, t: float, y: np.ndarray) -> None:
        """Record a filter output for later interpolation (call inside lock)."""
        self._out_t_buf.append(t)
        self._out_x_buf.append(y.copy())

    def _interpolate_output(self, t_query: float) -> Optional[np.ndarray]:
        """Interpolate over recorded output history using PCHIP.

        Uses shape-preserving Hermite (PCHIP) interpolation when enough
        points are available (≥4), otherwise falls back to linear.  Query
        times outside the recorded range are clamped (hold last/first value).

        Must be called **inside** the lock.
        """
        n = len(self._out_t_buf)
        if n == 0:
            return None
        if n == 1:
            return self._out_x_buf[0].copy()

        t_arr = np.array(self._out_t_buf)
        x_arr = np.stack(list(self._out_x_buf))

        # Sort by time (stable) and deduplicate keeping last occurrence
        sort_idx = np.argsort(t_arr, kind="stable")
        t_sorted = t_arr[sort_idx]
        x_sorted = x_arr[sort_idx]
        unique_mask = np.concatenate([t_sorted[:-1] != t_sorted[1:], [True]])
        t_unique = t_sorted[unique_mask]
        x_unique = x_sorted[unique_mask]

        n_unique = len(t_unique)
        if n_unique == 1:
            return x_unique[0].copy()

        # Clamp to buffer range (hold at boundaries)
        t_clamped = float(np.clip(t_query, t_unique[0], t_unique[-1]))
        n_dims = x_unique.shape[1]

        if n_unique >= 4:
            result = np.array(
                [
                    float(
                        sp_interpolate.PchipInterpolator(
                            t_unique, x_unique[:, d]
                        )(t_clamped)
                    )
                    for d in range(n_dims)
                ]
            )
        else:
            result = np.array(
                [
                    float(np.interp(t_clamped, t_unique, x_unique[:, d]))
                    for d in range(n_dims)
                ]
            )
        return result

    # ------------------------------------------------------------------
    # Abstract computation
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _compute(
        self,
        t: np.ndarray,
        x: np.ndarray,
        t_query: float,
    ) -> np.ndarray:
        """Compute the filtered output from buffered data.

        Parameters
        ----------
        t : np.ndarray, shape (N,)
            Buffered timestamps (N >= 2).
        x : np.ndarray, shape (N, D)
            Buffered values.
        t_query : float
            Timestamp at which to evaluate.

        Returns
        -------
        np.ndarray, shape (D,)
        """


# ======================================================================
# Subclasses
# ======================================================================

class AsyncFilterEMA(AsyncFilterBase):
    """Exponential moving average filter.

    Parameters
    ----------
    buffer_size : int, optional
        Buffer capacity (default: ``100``).
    alpha : float, optional
        Smoothing factor in ``(0, 1]`` (default: ``0.3``).
    """

    def __init__(self, buffer_size: int = 100, alpha: float = 0.3) -> None:
        super().__init__(buffer_size=buffer_size)
        self._alpha = float(alpha)
        self._ema_state: Optional[np.ndarray] = None

    # -- hooks ----------------------------------------------------------

    def _on_update(self, t: float, x: np.ndarray) -> None:
        if self._ema_state is None:
            self._ema_state = x.copy()
        else:
            self._ema_state = self._alpha * x + (1.0 - self._alpha) * self._ema_state
        self._record_output(t, self._ema_state)

    def _on_clear(self) -> None:
        self._ema_state = None

    # -- output ---------------------------------------------------------

    def get_output(self, t: Optional[float] = None) -> Optional[np.ndarray]:
        """Return EMA estimate, interpolated over output history when *t* is given."""
        with self._lock:
            if self._dim is None:
                return None
            if t is None:
                return self._ema_state.copy()
            return self._interpolate_output(float(t))

    def _compute(self, t: np.ndarray, x: np.ndarray, t_query: float) -> np.ndarray:
        # Not called for EMA (get_output is overridden), but required by ABC.
        return self._ema_state.copy()  # pragma: no cover


class AsyncFilterLinear(AsyncFilterBase):
    """Linear-interpolation filter over the circular buffer."""

    def _compute(self, t: np.ndarray, x: np.ndarray, t_query: float) -> np.ndarray:
        n_dims = x.shape[1]
        return np.array(
            [float(np.interp(t_query, t, x[:, d])) for d in range(n_dims)]
        )


class AsyncFilterSpline(AsyncFilterBase):
    """Cubic-spline interpolation filter (falls back to linear with <4 points)."""

    def _compute(self, t: np.ndarray, x: np.ndarray, t_query: float) -> np.ndarray:
        n_dims = x.shape[1]

        # Sort by time (buffer order may be non-monotonic after overlapping chunks)
        order = np.argsort(t, kind="stable")
        t_s = t[order]
        x_s = x[order]

        # Deduplicate timestamps (keep last occurrence for each time)
        unique_mask = np.concatenate([t_s[:-1] != t_s[1:], [True]])
        t_u = t_s[unique_mask]
        x_u = x_s[unique_mask]

        # Clamp query to data range (extrapolation causes divergence)
        t_q = float(np.clip(t_query, t_u[0], t_u[-1]))

        kind = "cubic" if len(t_u) >= 4 else "linear"
        return np.array(
            [
                float(
                    sp_interpolate.interp1d(
                        t_u, x_u[:, d], kind=kind, fill_value="extrapolate"
                    )(t_q)
                )
                for d in range(n_dims)
            ]
        )


class AsyncFilterOneEuro(AsyncFilterBase):
    """Adaptive low-pass filter (1€ filter, Casiez et al. CHI 2012).

    Same algorithm as :class:`OnlineFilterOneEuro` but wrapped in the
    thread-safe async buffer infrastructure.

    Parameters
    ----------
    buffer_size : int, optional
        Buffer capacity (default: ``100``).
    min_cutoff : float, optional
        Minimum cutoff frequency (Hz) (default: ``1.0``).
    beta : float, optional
        Speed coefficient (default: ``0.0``).
    d_cutoff : float, optional
        Cutoff for derivative smoothing (Hz) (default: ``1.0``).
    """

    def __init__(
        self,
        buffer_size: int = 100,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ) -> None:
        super().__init__(buffer_size=buffer_size)
        self._min_cutoff = float(min_cutoff)
        self._beta = float(beta)
        self._d_cutoff = float(d_cutoff)
        self._euro_state: Optional[np.ndarray] = None
        self._dx_state: Optional[np.ndarray] = None
        self._prev_t: Optional[float] = None

    @staticmethod
    def _sf(te: float, cutoff: float) -> float:
        r = 2.0 * np.pi * cutoff * te
        return r / (r + 1.0)

    def _on_update(self, t: float, x: np.ndarray) -> None:
        if self._euro_state is None:
            self._euro_state = x.copy()
            self._dx_state = np.zeros_like(x)
            self._prev_t = t
            self._record_output(t, self._euro_state)
            return
        te = t - self._prev_t
        if te <= 0.0:
            te = 1e-9
        alpha_d = self._sf(te, self._d_cutoff)
        dx = (x - self._euro_state) / te
        self._dx_state = alpha_d * dx + (1.0 - alpha_d) * self._dx_state
        cutoff = self._min_cutoff + self._beta * np.abs(self._dx_state)
        alpha = np.array([self._sf(te, float(c)) for c in cutoff])
        self._euro_state = alpha * x + (1.0 - alpha) * self._euro_state
        self._prev_t = t
        self._record_output(t, self._euro_state)

    def _on_clear(self) -> None:
        self._euro_state = None
        self._dx_state = None
        self._prev_t = None

    def get_output(self, t: Optional[float] = None) -> Optional[np.ndarray]:
        """Return 1€ filter estimate, interpolated over output history when *t* is given."""
        with self._lock:
            if self._dim is None:
                return None
            if t is None:
                return self._euro_state.copy()
            return self._interpolate_output(float(t))

    def _compute(self, t: np.ndarray, x: np.ndarray, t_query: float) -> np.ndarray:
        return self._euro_state.copy()  # pragma: no cover


class AsyncFilterMovingAverage(AsyncFilterBase):
    """Sliding-window moving average over the circular buffer.

    Parameters
    ----------
    buffer_size : int, optional
        Buffer capacity (default: ``100``).
    window : int, optional
        Averaging window size in samples (default: ``10``).
        Uses up to ``window`` most recent buffered samples.
    """

    def __init__(self, buffer_size: int = 100, window: int = 10) -> None:
        super().__init__(buffer_size=buffer_size)
        self._window = int(window)
        self._ma_state: Optional[np.ndarray] = None

    def _on_update(self, t: float, x: np.ndarray) -> None:
        w = min(self._window, len(self._x_buf))
        recent = list(self._x_buf)[-w:]
        self._ma_state = np.mean(np.stack(recent), axis=0)
        self._record_output(t, self._ma_state)

    def _on_clear(self) -> None:
        self._ma_state = None

    def get_output(self, t: Optional[float] = None) -> Optional[np.ndarray]:
        """Return moving-average estimate, interpolated when *t* is given."""
        with self._lock:
            if self._dim is None or self._ma_state is None:
                return None
            if t is None:
                return self._ma_state.copy()
            return self._interpolate_output(float(t))

    def _compute(self, t: np.ndarray, x: np.ndarray, t_query: float) -> np.ndarray:
        # Not called when get_output is overridden, but required by ABC.
        w = min(self._window, len(x))
        return np.mean(x[-w:], axis=0)


# ======================================================================
# Chunk-aware subclasses (VLA action chunk methods)
# ======================================================================

# -- Helpers ---------------------------------------------------------------

class _PolyTrajectory:
    """Multi-dimensional polynomial trajectory fitted to chunk waypoints.

    Used internally by :class:`AsyncFilterRAIL` for intra-chunk smoothing.

    Parameters
    ----------
    extrapolation : str
        Behaviour when evaluating outside ``[t_start, t_end]``:

        * ``"clamp"`` – hold boundary value (zero-order hold).
        * ``"linear"`` – linear extrapolation from boundary value and
          velocity (first-order hold).  Default.
        * ``"poly"`` – unclamped polynomial evaluation (original
          behaviour; may diverge far from the data range).
    """

    _VALID_EXTRAP = frozenset({"clamp", "linear", "poly"})

    __slots__ = ("t_start", "t_end", "_polys", "_extrapolation")

    def __init__(
        self,
        t_start: float,
        t_end: float,
        polys: list[np.poly1d],
        extrapolation: str = "linear",
    ) -> None:
        self.t_start = t_start
        self.t_end = t_end
        self._polys = polys
        self._extrapolation = extrapolation

    @classmethod
    def fit(
        cls,
        t: np.ndarray,
        x: np.ndarray,
        degree: int,
        extrapolation: str = "linear",
    ) -> "_PolyTrajectory":
        """Fit polynomial of given *degree* to each dimension of *x*."""
        n_dims = x.shape[1]
        deg = min(degree, len(t) - 1)
        polys = [np.poly1d(np.polyfit(t, x[:, d], deg)) for d in range(n_dims)]
        return cls(float(t[0]), float(t[-1]), polys, extrapolation)

    def evaluate(self, t: float) -> np.ndarray:
        """Evaluate trajectory at time *t* with extrapolation handling."""
        if self._extrapolation == "poly" or self.t_start <= t <= self.t_end:
            return np.array([float(p(t)) for p in self._polys])

        if self._extrapolation == "clamp":
            t_c = max(self.t_start, min(t, self.t_end))
            return np.array([float(p(t_c)) for p in self._polys])

        # "linear": first-order hold from boundary
        if t < self.t_start:
            t_b = self.t_start
        else:
            t_b = self.t_end
        pos = np.array([float(p(t_b)) for p in self._polys])
        vel = np.array([float(p.deriv()(t_b)) for p in self._polys])
        return pos + vel * (t - t_b)

    def evaluate_deriv(self, t: float, order: int = 1) -> np.ndarray:
        """Evaluate *order*-th derivative of trajectory at time *t*."""
        derivs = []
        for p in self._polys:
            dp = p
            for _ in range(order):
                dp = dp.deriv()
            derivs.append(dp)

        if self._extrapolation == "poly" or self.t_start <= t <= self.t_end:
            return np.array([float(dp(t)) for dp in derivs])

        if self._extrapolation == "clamp":
            t_c = max(self.t_start, min(t, self.t_end))
            if order == 0:
                return self.evaluate(t)
            # Derivatives of a constant are zero
            return np.zeros(len(self._polys))

        # "linear": velocity is constant, higher derivs are zero
        if order == 0:
            return self.evaluate(t)
        if order == 1:
            t_b = self.t_start if t < self.t_start else self.t_end
            return np.array([float(p.deriv()(t_b)) for p in self._polys])
        return np.zeros(len(self._polys))


class _CubicBlend:
    """Single cubic polynomial blend ensuring C¹ continuity.

    Uses position and velocity boundary conditions only (no acceleration),
    avoiding the acceleration-cascade amplification problem that afflicts
    C² quintic blends under noisy conditions.
    """

    __slots__ = ("t_start", "t_end", "_polys")

    def __init__(
        self,
        traj_old: _PolyTrajectory,
        traj_new: _PolyTrajectory,
        t_start: float,
        t_end: float,
        *,
        start_state: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.t_start = t_start
        self.t_end = t_end

        if start_state is not None:
            p0, v0, _a0 = start_state
        else:
            p0 = traj_old.evaluate(t_start)
            v0 = traj_old.evaluate_deriv(t_start, 1)

        p1 = traj_new.evaluate(t_end)
        v1 = traj_new.evaluate_deriv(t_end, 1)

        xi = np.array([t_start, t_end])
        n_dims = len(p0)
        self._polys: list[sp_interpolate.BPoly] = []
        for d in range(n_dims):
            yi = np.array([[p0[d], v0[d]], [p1[d], v1[d]]])
            self._polys.append(sp_interpolate.BPoly.from_derivatives(xi, yi))

    def evaluate(self, t: float) -> np.ndarray:
        return np.array([float(p(t)) for p in self._polys])

    def evaluate_deriv(self, t: float, order: int = 1) -> np.ndarray:
        return np.array([float(p.derivative(order)(t)) for p in self._polys])


class _DualCubicBlend:
    """Dual cubic blend (C¹ variant of dual-quintic).

    Splits the blend into two cubic halves at the midpoint.
    Midpoint conditions: averaged position & velocity (no acceleration).
    """

    __slots__ = ("t_start", "t_end", "_t_mid", "_left_polys", "_right_polys")

    def __init__(
        self,
        traj_old: _PolyTrajectory,
        traj_new: _PolyTrajectory,
        t_start: float,
        t_end: float,
        *,
        start_state: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.t_start = t_start
        self.t_end = t_end
        self._t_mid = 0.5 * (t_start + t_end)

        if start_state is not None:
            p0, v0, _a0 = start_state
        else:
            p0 = traj_old.evaluate(t_start)
            v0 = traj_old.evaluate_deriv(t_start, 1)

        p1 = traj_new.evaluate(t_end)
        v1 = traj_new.evaluate_deriv(t_end, 1)

        p_mid = 0.5 * (p0 + p1)
        v_mid = 0.5 * (v0 + v1)

        n_dims = len(p0)
        self._left_polys: list[sp_interpolate.BPoly] = []
        self._right_polys: list[sp_interpolate.BPoly] = []
        for d in range(n_dims):
            xi_l = np.array([t_start, self._t_mid])
            yi_l = np.array([[p0[d], v0[d]], [p_mid[d], v_mid[d]]])
            self._left_polys.append(
                sp_interpolate.BPoly.from_derivatives(xi_l, yi_l)
            )
            xi_r = np.array([self._t_mid, t_end])
            yi_r = np.array([[p_mid[d], v_mid[d]], [p1[d], v1[d]]])
            self._right_polys.append(
                sp_interpolate.BPoly.from_derivatives(xi_r, yi_r)
            )

    def evaluate(self, t: float) -> np.ndarray:
        if t < self._t_mid:
            return np.array([float(p(t)) for p in self._left_polys])
        return np.array([float(p(t)) for p in self._right_polys])

    def evaluate_deriv(self, t: float, order: int = 1) -> np.ndarray:
        if t < self._t_mid:
            return np.array(
                [float(p.derivative(order)(t)) for p in self._left_polys]
            )
        return np.array(
            [float(p.derivative(order)(t)) for p in self._right_polys]
        )


class _QuinticBlend:
    """Single quintic polynomial blend ensuring C² continuity.

    Constructs a quintic polynomial per dimension using
    ``scipy.interpolate.BPoly.from_derivatives`` so that position, velocity,
    and acceleration match the old trajectory at the start of the blend
    region and the new trajectory at the end.

    Used internally by :class:`AsyncFilterRAIL` for inter-chunk fusion.
    """

    __slots__ = ("t_start", "t_end", "_polys")

    def __init__(
        self,
        traj_old: _PolyTrajectory,
        traj_new: _PolyTrajectory,
        t_start: float,
        t_end: float,
        *,
        start_state: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.t_start = t_start
        self.t_end = t_end

        # Boundary conditions at blend start
        if start_state is not None:
            p0, v0, a0 = start_state
        else:
            p0 = traj_old.evaluate(t_start)
            v0 = traj_old.evaluate_deriv(t_start, 1)
            a0 = traj_old.evaluate_deriv(t_start, 2)

        # Boundary conditions at blend end (always from new trajectory)
        p1 = traj_new.evaluate(t_end)
        v1 = traj_new.evaluate_deriv(t_end, 1)
        a1 = traj_new.evaluate_deriv(t_end, 2)

        xi = np.array([t_start, t_end])
        n_dims = len(p0)
        self._polys: list[sp_interpolate.BPoly] = []
        for d in range(n_dims):
            yi = np.array([[p0[d], v0[d], a0[d]], [p1[d], v1[d], a1[d]]])
            self._polys.append(sp_interpolate.BPoly.from_derivatives(xi, yi))

    def evaluate(self, t: float) -> np.ndarray:
        """Evaluate the blend polynomial at time *t*."""
        return np.array([float(p(t)) for p in self._polys])

    def evaluate_deriv(self, t: float, order: int = 1) -> np.ndarray:
        """Evaluate *order*-th derivative of the blend at time *t*."""
        return np.array([float(p.derivative(order)(t)) for p in self._polys])


class _DualQuinticBlend:
    """Dual quintic blend per VLA-RAIL paper (Eq. 11-13).

    Splits the blend region into two halves at the midpoint to avoid
    overshoot caused by Runge's phenomenon in a single quintic over a
    long interval.  At the midpoint, position and velocity are the
    averages of the old and new trajectories and acceleration is zero.

    Used internally by :class:`AsyncFilterRAIL` when ``dual_quintic=True``.
    """

    __slots__ = ("t_start", "t_end", "_t_mid", "_left_polys", "_right_polys")

    def __init__(
        self,
        traj_old: _PolyTrajectory,
        traj_new: _PolyTrajectory,
        t_start: float,
        t_end: float,
        *,
        start_state: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.t_start = t_start
        self.t_end = t_end
        self._t_mid = 0.5 * (t_start + t_end)

        # Old trajectory at blend start
        if start_state is not None:
            p0, v0, a0 = start_state
        else:
            p0 = traj_old.evaluate(t_start)
            v0 = traj_old.evaluate_deriv(t_start, 1)
            a0 = traj_old.evaluate_deriv(t_start, 2)

        # New trajectory at blend end
        p1 = traj_new.evaluate(t_end)
        v1 = traj_new.evaluate_deriv(t_end, 1)
        a1 = traj_new.evaluate_deriv(t_end, 2)

        # Midpoint: averaged position & velocity, zero acceleration
        p_mid = 0.5 * (p0 + p1)
        v_mid = 0.5 * (v0 + v1)

        n_dims = len(p0)
        self._left_polys: list[sp_interpolate.BPoly] = []
        self._right_polys: list[sp_interpolate.BPoly] = []
        for d in range(n_dims):
            xi_l = np.array([t_start, self._t_mid])
            yi_l = np.array([[p0[d], v0[d], a0[d]], [p_mid[d], v_mid[d], 0.0]])
            self._left_polys.append(
                sp_interpolate.BPoly.from_derivatives(xi_l, yi_l)
            )
            xi_r = np.array([self._t_mid, t_end])
            yi_r = np.array([[p_mid[d], v_mid[d], 0.0], [p1[d], v1[d], a1[d]]])
            self._right_polys.append(
                sp_interpolate.BPoly.from_derivatives(xi_r, yi_r)
            )

    def evaluate(self, t: float) -> np.ndarray:
        """Evaluate the dual-quintic blend at time *t*."""
        if t < self._t_mid:
            return np.array([float(p(t)) for p in self._left_polys])
        return np.array([float(p(t)) for p in self._right_polys])

    def evaluate_deriv(self, t: float, order: int = 1) -> np.ndarray:
        """Evaluate *order*-th derivative of the blend at time *t*."""
        if t < self._t_mid:
            return np.array(
                [float(p.derivative(order)(t)) for p in self._left_polys]
            )
        return np.array(
            [float(p.derivative(order)(t)) for p in self._right_polys]
        )


# -- ACT temporal ensembling -----------------------------------------------

class AsyncFilterACT(AsyncFilterBase):
    """Temporal ensembling of overlapping action chunks (ACT-style).

    Stores a rolling window of action chunks and produces output by
    computing a weighted average of all chunk predictions that overlap at
    the query time.  Newer chunks receive exponentially higher weight.

    Based on Action Chunking with Transformers (Zhao et al., RSS 2023)
    and the ``weighted_average`` aggregation in LeRobot's async inference.

    Parameters
    ----------
    buffer_size : int, optional
        Per-sample buffer capacity (default: ``100``).
    k : float, optional
        Exponential decay coefficient (default: ``0.01``).  Higher values
        cause newer chunks to dominate more strongly.
    max_chunks : int, optional
        Maximum number of chunks kept in memory (default: ``10``).
    """

    def __init__(
        self,
        buffer_size: int = 100,
        k: float = 0.01,
        max_chunks: int = 10,
    ) -> None:
        super().__init__(buffer_size=buffer_size)
        self._k = float(k)
        self._max_chunks = int(max_chunks)
        self._chunks: list[tuple[np.ndarray, np.ndarray]] = []

    # -- chunk interface ----------------------------------------------------

    def update_chunk(self, t, x) -> None:
        """Add an action chunk for temporal ensembling."""
        t_arr = np.asarray(t, dtype=float).ravel()
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        if x_arr.shape[0] != len(t_arr):
            raise ValueError(
                f"t has {len(t_arr)} samples but x has {x_arr.shape[0]}."
            )
        with self._lock:
            if self._dim is None:
                self._dim = x_arr.shape[1]
            self._chunks.append((t_arr.copy(), x_arr.copy()))
            if len(self._chunks) > self._max_chunks:
                self._chunks.pop(0)
            # Also populate the base buffer for compatibility
            for ti, xi in zip(t_arr, x_arr):
                self._t_buf.append(float(ti))
                self._x_buf.append(xi.copy())

    # -- output -------------------------------------------------------------

    def get_output(self, t: Optional[float] = None) -> Optional[np.ndarray]:
        """Return temporally ensembled output at time *t*."""
        with self._lock:
            if self._dim is None or not self._chunks:
                return None
            t_query = (
                float(t)
                if t is not None
                else float(self._chunks[-1][0][-1])
            )
            return self._temporal_ensemble(t_query)

    def _temporal_ensemble(self, t_query: float) -> np.ndarray:
        """Weighted average of all chunk predictions at *t_query*."""
        n_chunks = len(self._chunks)
        values: list[np.ndarray] = []
        weights: list[float] = []
        for i, (t_c, x_c) in enumerate(self._chunks):
            if t_query < t_c[0] or t_query > t_c[-1]:
                continue
            # Interpolate this chunk's prediction at t_query
            val = np.array(
                [np.interp(t_query, t_c, x_c[:, d]) for d in range(self._dim)]
            )
            age = n_chunks - 1 - i  # 0 for newest chunk
            w = np.exp(-self._k * age)
            values.append(val)
            weights.append(w)

        if not values:
            # t_query outside all chunks: hold last chunk's nearest endpoint
            last_t, last_x = self._chunks[-1]
            if t_query >= last_t[-1]:
                return last_x[-1].copy()
            return last_x[0].copy()

        wt = np.array(weights)
        wt /= wt.sum()
        return np.average(np.stack(values), axis=0, weights=wt)

    # -- hooks / ABC --------------------------------------------------------

    def _on_clear(self) -> None:
        self._chunks.clear()

    def _compute(
        self, t: np.ndarray, x: np.ndarray, t_query: float
    ) -> np.ndarray:
        return self._temporal_ensemble(t_query)


# -- VLA-RAIL trajectory post-processing -----------------------------------

class AsyncFilterRAIL(AsyncFilterBase):
    """VLA-RAIL two-stage trajectory post-processing.

    1. **Intra-chunk smoothing** – fits a polynomial of degree *poly_degree*
       to each dimension of the incoming chunk, filtering high-frequency
       noise from VLA predictions.
    2. **Inter-chunk fusion** – constructs a polynomial blend at chunk
       boundaries, ensuring continuity between successive trajectories.

    Based on VLA-RAIL (Zhao et al., arXiv:2512.24673).

    Parameters
    ----------
    buffer_size : int, optional
        Per-sample buffer capacity (default: ``100``).
    poly_degree : int, optional
        Polynomial degree for intra-chunk smoothing (default: ``3``, i.e.
        cubic, as recommended by the VLA-RAIL paper).
    blend_duration : float or None, optional
        Duration (seconds) of the inter-chunk blend region.
        If ``None`` (default), uses 30 % of the new chunk's duration.
    dual_quintic : bool, optional
        If ``True`` (default), split the blend region into two halves at
        the midpoint (VLA-RAIL Eq. 11-13) to avoid Runge's phenomenon.
        If ``False``, use a single blend over the whole region.
    auto_align : bool, optional
        If ``True``, automatically correct new-chunk timestamps via
        temporal alignment optimisation before blending.
        Requires :meth:`set_current_time` to have been called.
        Default: ``False``.
    align_method : str, optional
        Algorithm used when ``auto_align=True``.

        * ``"direction"`` (default) – VLA-RAIL Eq. 10: maximise
          motion-direction consistency between the shifted chunk and
          the current trajectory's velocity at ``t_now``.
        * ``"least_squares"`` – minimise total squared position error
          between the current trajectory and the shifted chunk over
          their overlap region.  More robust than ``"direction"`` for
          large temporal offsets and produces smoother blends.
    align_window : float or None, optional
        Search window (seconds) for temporal alignment.  If ``None``
        (default), uses 50 % of the new chunk's duration.
    blend_start_source : str, optional
        Strategy for computing the boundary conditions at the start of
        a new blend region.

        * ``"actual_output"`` (default) – evaluate whichever curve the
          output pipeline would actually return at the blend-start time
          (may be a previous blend curve, not the raw polynomial).
          Derivatives are computed analytically from that curve.
        * ``"trajectory"`` – evaluate the old chunk's polynomial at
          the blend-start time.  This was the original behaviour but
          can suffer from position mismatches when the output was on
          a blend curve.
        * ``"output_history"`` – use finite differences of the most
          recent outputs recorded by :meth:`get_output` to estimate
          position, velocity, and acceleration.  Works even when the
          underlying curves are unavailable or unreliable, but may be
          noisy if the output sample rate is low.
    blend_order : str, optional
        Continuity order of the blend polynomial.

        * ``"cubic"`` (default) – C¹ continuity (position + velocity).
          Uses cubic polynomials.  Avoids the acceleration-cascade
          amplification problem present in quintic blends.
        * ``"quintic"`` – C² continuity (position + velocity +
          acceleration).  Original VLA-RAIL behaviour.  May suffer
          from overshoot when chunks arrive frequently under noisy
          conditions.
    acc_clamp : float or None, optional
        Maximum absolute acceleration allowed in blend boundary
        conditions.  Only effective when ``blend_order="quintic"``.
        If ``None`` (default), no clamping is applied.
        Recommended value: ``5.0``–``20.0`` depending on expected
        trajectory dynamics.
    extrapolation : str, optional
        Behaviour when evaluating a polynomial trajectory outside its
        fitted time range ``[t_start, t_end]``.

        * ``"linear"`` (default) – first-order hold: extrapolate
          linearly from boundary position and velocity.  Safe and
          produces plausible output for short extrapolation.
        * ``"clamp"`` – zero-order hold: hold the boundary value.
        * ``"poly"`` – use the raw polynomial (original behaviour).
          **Warning**: may diverge rapidly for higher-degree polynomials.
    """

    _VALID_BLEND_SOURCES = frozenset(
        {"trajectory", "actual_output", "output_history"}
    )
    _VALID_BLEND_ORDERS = frozenset({"cubic", "quintic"})
    _VALID_ALIGN_METHODS = frozenset({"direction", "least_squares"})

    def __init__(
        self,
        buffer_size: int = 100,
        poly_degree: int = 3,
        blend_duration: Optional[float] = None,
        dual_quintic: bool = True,
        auto_align: bool = False,
        align_method: str = "direction",
        align_window: Optional[float] = None,
        blend_start_source: str = "actual_output",
        blend_order: str = "cubic",
        acc_clamp: Optional[float] = None,
        extrapolation: str = "linear",
    ) -> None:
        super().__init__(buffer_size=buffer_size)
        if blend_start_source not in self._VALID_BLEND_SOURCES:
            raise ValueError(
                f"blend_start_source must be one of "
                f"{sorted(self._VALID_BLEND_SOURCES)}, "
                f"got {blend_start_source!r}."
            )
        if blend_order not in self._VALID_BLEND_ORDERS:
            raise ValueError(
                f"blend_order must be one of "
                f"{sorted(self._VALID_BLEND_ORDERS)}, "
                f"got {blend_order!r}."
            )
        if extrapolation not in _PolyTrajectory._VALID_EXTRAP:
            raise ValueError(
                f"extrapolation must be one of "
                f"{sorted(_PolyTrajectory._VALID_EXTRAP)}, "
                f"got {extrapolation!r}."
            )
        if align_method not in self._VALID_ALIGN_METHODS:
            raise ValueError(
                f"align_method must be one of "
                f"{sorted(self._VALID_ALIGN_METHODS)}, "
                f"got {align_method!r}."
            )
        self._poly_degree = int(poly_degree)
        self._blend_duration = blend_duration
        self._dual_quintic = bool(dual_quintic)
        self._auto_align = bool(auto_align)
        self._align_method = align_method
        self._align_window = align_window
        self._blend_start_source = blend_start_source
        self._blend_order = blend_order
        self._acc_clamp = float(acc_clamp) if acc_clamp is not None else None
        self._extrapolation = extrapolation
        self._current_time: Optional[float] = None
        self._prev_traj: Optional[_PolyTrajectory] = None
        self._curr_traj: Optional[_PolyTrajectory] = None
        self._blend: Optional[
            _CubicBlend | _DualCubicBlend
            | _QuinticBlend | _DualQuinticBlend
        ] = None
        # Output history for "output_history" blend-start strategy.
        self._output_history: deque[tuple[float, np.ndarray]] = deque(
            maxlen=buffer_size
        )

    # -- current-time interface ---------------------------------------------

    def set_current_time(self, t: float) -> None:
        """Update the current time used as the blend start.

        In a real-time control loop this should be called every cycle
        (or at least before each :meth:`update_chunk`) so that the blend
        region starts at the actual switch time rather than the new
        chunk's first timestamp.  Also required for ``auto_align=True``.

        This method is thread-safe.
        """
        with self._lock:
            self._current_time = float(t)

    # -- temporal alignment -------------------------------------------------

    def _compute_temporal_alignment(
        self,
        traj_new: _PolyTrajectory,
        t_now: float,
    ) -> float:
        """Find optimal time shift for new chunk (VLA-RAIL Eq. 10).

        Returns the shift to *add* to the new chunk's timestamps so that
        the motion direction of the shifted chunk at ``t_now`` is
        consistent with the current trajectory's velocity.
        """
        if self._curr_traj is None:
            return 0.0

        p_old = self._curr_traj.evaluate(t_now)
        v_old = self._curr_traj.evaluate_deriv(t_now, 1)

        chunk_dur = traj_new.t_end - traj_new.t_start
        window = (
            self._align_window
            if self._align_window is not None
            else 0.5 * chunk_dur
        )

        best_score = -float("inf")
        best_ta: float = 0.0
        n_search = 50
        default_ta = t_now - traj_new.t_start
        score_default = float(np.sum(np.sign((traj_new.evaluate(t_now) - p_old) * v_old)))

        for ta in np.linspace(0.0, window, n_search):
            t_eval = traj_new.t_start + ta
            if t_eval > traj_new.t_end:
                break
            p_new = traj_new.evaluate(t_eval)
            diff = p_new - p_old
            score = float(np.sum(np.sign(diff * v_old)))
            if score > best_score or (
                score == best_score and ta < best_ta
            ):
                best_score = score
                best_ta = ta

        if best_score <= score_default:
            best_ta = default_ta

        # Shift so that (t_new_start + best_ta) aligns with t_now
        return t_now - (traj_new.t_start + best_ta)

    def _compute_ls_alignment(
        self,
        t_raw: np.ndarray,
        x_raw: np.ndarray,
        t_now: float,
    ) -> float:
        """Find optimal time shift via least-squares overlap matching.

        For each candidate shift ``s``, a temporary polynomial is fitted
        to ``(t_raw + s, x_raw)``.  The cost is the sum of squared
        position differences between that polynomial and the current
        trajectory, evaluated at evenly-spaced points in their overlap
        region.  The shift with the lowest cost is returned.

        Uses a coarse-to-fine grid search for efficiency: a coarse pass
        (15 candidates) identifies the best basin, then a refinement
        pass (10 candidates) narrows down to the optimum.  Polynomial
        fitting and evaluation are vectorized across all dimensions
        using a Vandermonde matrix and :func:`numpy.linalg.lstsq`.
        """
        if self._curr_traj is None:
            return 0.0

        old_traj = self._curr_traj
        chunk_dur = float(t_raw[-1] - t_raw[0])
        window = (
            self._align_window
            if self._align_window is not None
            else 0.5 * chunk_dur
        )

        n_eval = 15
        deg = min(self._poly_degree, len(t_raw) - 1)
        old_polys = old_traj._polys

        def _eval_shifts(shifts: np.ndarray) -> np.ndarray:
            costs = np.full(len(shifts), np.inf)
            for i, shift in enumerate(shifts):
                t_shifted = t_raw + shift
                t_lo = max(old_traj.t_start, float(t_shifted[0]))
                t_hi = min(old_traj.t_end, float(t_shifted[-1]))
                if t_hi <= t_lo:
                    continue

                # Normalize time for numerical stability
                t_center = 0.5 * (t_shifted[0] + t_shifted[-1])
                t_half = 0.5 * (t_shifted[-1] - t_shifted[0])
                if t_half < 1e-15:
                    t_half = 1.0
                t_norm = (t_shifted - t_center) / t_half

                # Fit all dims at once via Vandermonde + lstsq
                V = np.vander(t_norm, N=deg + 1)
                try:
                    new_coeffs, _, _, _ = np.linalg.lstsq(
                        V, x_raw, rcond=None,
                    )
                except Exception:
                    continue

                t_eval = np.linspace(t_lo, t_hi, n_eval)
                t_eval_norm = (t_eval - t_center) / t_half

                # Vectorized evaluation: new trajectory
                V_eval = np.vander(t_eval_norm, N=deg + 1)
                new_vals = V_eval @ new_coeffs  # (n_eval, n_dims)

                # Vectorized evaluation: old trajectory (polyval)
                old_vals = np.column_stack(
                    [np.polyval(p.coefficients, t_eval) for p in old_polys]
                )

                diff = new_vals - old_vals
                costs[i] = float(np.sum(diff * diff))
            return costs

        # Phase 1: coarse search
        n_coarse = 15
        coarse_shifts = np.linspace(-window, window, n_coarse)
        coarse_costs = _eval_shifts(coarse_shifts)

        best_coarse = int(np.argmin(coarse_costs))
        if not np.isfinite(coarse_costs[best_coarse]):
            return 0.0

        # Phase 2: refine around the best coarse candidate
        n_refine = 10
        step = (
            (coarse_shifts[1] - coarse_shifts[0])
            if n_coarse > 1
            else window
        )
        refine_shifts = np.linspace(
            coarse_shifts[best_coarse] - step,
            coarse_shifts[best_coarse] + step,
            n_refine,
        )
        refine_costs = _eval_shifts(refine_shifts)

        all_shifts = np.concatenate([coarse_shifts, refine_shifts])
        all_costs = np.concatenate([coarse_costs, refine_costs])
        best_idx = int(np.argmin(all_costs))

        return (
            float(all_shifts[best_idx])
            if np.isfinite(all_costs[best_idx])
            else 0.0
        )

    # -- chunk interface ----------------------------------------------------

    def _get_output_state_unlocked(
        self, t_query: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (position, velocity, acceleration) from the active curve.

        Must be called with ``self._lock`` held.  Uses the same priority
        logic as :meth:`get_output` to find which curve is active, then
        evaluates its derivatives analytically.
        """
        # Determine which source curve to use
        source: _PolyTrajectory | _QuinticBlend | _DualQuinticBlend

        if (
            self._blend is not None
            and self._blend.t_start <= t_query <= self._blend.t_end
        ):
            source = self._blend
        elif (
            self._blend is not None
            and t_query < self._blend.t_start
            and self._prev_traj is not None
        ):
            source = self._prev_traj
        elif self._curr_traj is not None and t_query >= self._curr_traj.t_start:
            source = self._curr_traj
        elif self._prev_traj is not None:
            source = self._prev_traj
        else:
            source = self._curr_traj  # type: ignore[assignment]

        pos = source.evaluate(t_query)
        vel = source.evaluate_deriv(t_query, 1)
        acc = source.evaluate_deriv(t_query, 2)
        return pos, vel, acc

    def _estimate_state_from_history(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Estimate (position, velocity, acceleration) from output history.

        Uses finite differences of :meth:`get_output` values recorded in
        ``self._output_history``.  Must be called with ``self._lock`` held.
        """
        n = len(self._output_history)
        if n < 1:
            return None

        pos = self._output_history[-1][1].copy()

        if n < 2:
            return pos, np.zeros_like(pos), np.zeros_like(pos)

        t1, x1 = self._output_history[-1]
        t0, x0 = self._output_history[-2]
        dt = t1 - t0
        vel = (x1 - x0) / dt if abs(dt) > 1e-12 else np.zeros_like(pos)

        if n < 3:
            return pos, vel, np.zeros_like(pos)

        t_2, x_2 = self._output_history[-3]
        dt0 = t0 - t_2
        if abs(dt0) < 1e-12:
            return pos, vel, np.zeros_like(pos)
        vel_prev = (x0 - x_2) / dt0
        dt_avg = 0.5 * (dt + dt0)
        acc = (
            (vel - vel_prev) / dt_avg
            if abs(dt_avg) > 1e-12
            else np.zeros_like(pos)
        )
        return pos, vel, acc

    def _resolve_blend_start_state(
        self, t_bs: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Compute blend-start boundary conditions per ``blend_start_source``.

        Returns ``None`` when the default ``traj_old`` path should be used
        (and no acc_clamp is configured).
        Must be called with ``self._lock`` held.
        """
        state: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

        if self._blend_start_source == "actual_output":
            state = self._get_output_state_unlocked(t_bs)
        elif self._blend_start_source == "output_history":
            state = self._estimate_state_from_history()
        # else: "trajectory" → state stays None

        # Apply acceleration clamp if configured
        if self._acc_clamp is not None:
            if state is None:
                # Need to fetch from traj_old for clamping
                if self._prev_traj is not None:
                    p = self._prev_traj.evaluate(t_bs)
                    v = self._prev_traj.evaluate_deriv(t_bs, 1)
                    a = self._prev_traj.evaluate_deriv(t_bs, 2)
                    state = (p, v, a)
            if state is not None:
                p, v, a = state
                a = np.clip(a, -self._acc_clamp, self._acc_clamp)
                state = (p, v, a)

        return state

    def _select_blend_cls(self):
        """Return the blend class based on blend_order and dual_quintic."""
        if self._blend_order == "cubic":
            return _DualCubicBlend if self._dual_quintic else _CubicBlend
        return _DualQuinticBlend if self._dual_quintic else _QuinticBlend

    def update_chunk(self, t, x) -> None:
        """Add an action chunk with polynomial smoothing and blend fusion."""
        t_arr = np.asarray(t, dtype=float).ravel()
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        if x_arr.shape[0] != len(t_arr):
            raise ValueError(
                f"t has {len(t_arr)} samples but x has {x_arr.shape[0]}."
            )
        with self._lock:
            if self._dim is None:
                self._dim = x_arr.shape[1]

            # Stage 1: Intra-chunk smoothing (polynomial fit)
            traj = _PolyTrajectory.fit(
                t_arr, x_arr, self._poly_degree, self._extrapolation,
            )

            # Optional: temporal alignment (shift chunk timestamps)
            if self._auto_align and self._current_time is not None:
                if self._align_method == "least_squares":
                    shift = self._compute_ls_alignment(
                        t_arr, x_arr, self._current_time,
                    )
                else:
                    shift = self._compute_temporal_alignment(
                        traj, self._current_time,
                    )
                if abs(shift) > 1e-12:
                    t_shifted = t_arr + shift
                    traj = _PolyTrajectory.fit(
                        t_shifted, x_arr, self._poly_degree,
                        self._extrapolation,
                    )

            # Stage 2: Inter-chunk fusion (blend)
            if self._curr_traj is not None:
                self._prev_traj = self._curr_traj
                chunk_dur = traj.t_end - traj.t_start
                blend_dur = (
                    self._blend_duration
                    if self._blend_duration is not None
                    else 0.3 * chunk_dur
                )

                # Blend start: use current time if available (Algorithm 1),
                # otherwise fall back to new chunk's t_start.
                t_bs = (
                    self._current_time
                    if self._current_time is not None
                    else traj.t_start
                )
                t_be = t_bs + blend_dur
                # Clamp blend end to not exceed new trajectory
                t_be = min(t_be, traj.t_end)
                if t_be > t_bs:
                    BlendCls = self._select_blend_cls()
                    start_state = self._resolve_blend_start_state(t_bs)
                    self._blend = BlendCls(
                        self._prev_traj, traj, t_bs, t_be,
                        start_state=start_state,
                    )
                else:
                    self._blend = None

            self._curr_traj = traj

            # Populate base buffer for compatibility
            for ti, xi in zip(t_arr, x_arr):
                self._t_buf.append(float(ti))
                self._x_buf.append(xi.copy())

    # -- output -------------------------------------------------------------

    def get_output(self, t: Optional[float] = None) -> Optional[np.ndarray]:
        """Return the smoothed trajectory value at time *t*."""
        with self._lock:
            if self._dim is None or self._curr_traj is None:
                return None
            t_query = (
                float(t) if t is not None else self._curr_traj.t_end
            )

            # If in blend region, use quintic blend
            if (
                self._blend is not None
                and self._blend.t_start <= t_query <= self._blend.t_end
            ):
                result = self._blend.evaluate(t_query)
                self._output_history.append((t_query, result.copy()))
                return result

            # Before blend start: use previous (old) trajectory
            if (
                self._blend is not None
                and t_query < self._blend.t_start
                and self._prev_traj is not None
            ):
                result = self._prev_traj.evaluate(t_query)
                self._output_history.append((t_query, result.copy()))
                return result

            # After blend (or no blend): use current trajectory
            if t_query >= self._curr_traj.t_start:
                result = self._curr_traj.evaluate(t_query)
                self._output_history.append((t_query, result.copy()))
                return result

            # Before current trajectory: use previous if available
            if self._prev_traj is not None:
                result = self._prev_traj.evaluate(t_query)
                self._output_history.append((t_query, result.copy()))
                return result

            result = self._curr_traj.evaluate(t_query)
            self._output_history.append((t_query, result.copy()))
            return result

    # -- hooks / ABC --------------------------------------------------------

    def _on_clear(self) -> None:
        self._prev_traj = None
        self._curr_traj = None
        self._blend = None
        self._current_time = None
        self._output_history.clear()

    def _compute(
        self, t: np.ndarray, x: np.ndarray, t_query: float
    ) -> np.ndarray:
        if self._curr_traj is not None:
            return self._curr_traj.evaluate(t_query)
        return np.zeros(self._dim)  # pragma: no cover


# -- Real-Time Chunking (RTC) framework ------------------------------------


class AsyncFilterRTC(AsyncFilterBase):
    """Real-Time Chunking (RTC) asynchronous execution framework.

    Manages the asynchronous lifecycle of action chunks as described in
    Black et al. "Real-Time Execution of Action Chunking Flow Policies"
    (NeurIPS 2025, arXiv:2506.07339) and the training-time variant from
    Black et al. (arXiv:2512.05964).

    The filter provides:

    * **Prefix freezing** – :meth:`get_prefix` returns the frozen action
      prefix and current delay estimate so the caller can condition its
      model accordingly.
    * **Soft mask computation** – :meth:`get_soft_mask` returns the
      exponentially decaying mask vector :math:`\\mathbf{W}` (Eq. 5 of
      arXiv:2506.07339) for inference-time inpainting guidance.
    * **Action-space inpainting** – when *inpainting* is enabled, incoming
      chunks are automatically modified in :meth:`update_chunk` to enforce
      continuity with the frozen prefix from the previous chunk.  Three
      methods are available:

      * ``'hard'``      – replace the overlapping prefix with frozen
        actions (training-time RTC style; arXiv:2512.05964).
      * ``'soft_mask'`` – weighted blend with the previous chunk using the
        RTC exponential-decay soft mask (Eq. 5; arXiv:2506.07339).
      * ``'hermite'``   – replace prefix and apply a cubic Hermite
        transition at the boundary to guarantee :math:`C^1` continuity
        in position and velocity.
      * ``'callback'``  – delegate to a user-supplied callable
        (*inpainting_fn*), e.g. for ΠGDM-guided inpainting.
    * **Delay estimation** – a running buffer of observed delays with
      configurable aggregation (``max``, ``mean``, ``ema``).
    * **Seamless chunk hand-off** – :meth:`update_chunk` swaps in a newly
      generated chunk and updates internal bookkeeping; :meth:`get_output`
      returns the appropriate action at query time with optional blending
      across the chunk boundary.

    Parameters
    ----------
    buffer_size : int, optional
        Per-sample buffer capacity (default: ``100``).
    prediction_horizon : int
        :math:`H` – number of action steps in each chunk.
    min_execution_horizon : int, optional
        :math:`s_{\\min}` – minimum steps to execute before starting the
        next inference (default: ``1``).
    dt : float
        Controller sampling period in seconds (:math:`\\Delta t`).
    delay_buffer_size : int, optional
        Number of past delays kept for estimation (default: ``5``).
    initial_delay : int, optional
        Seed value for the delay buffer before any real measurement
        (default: ``1``).
    delay_estimate_method : str, optional
        How to aggregate past delays: ``'max'`` (default, conservative),
        ``'mean'``, or ``'ema'`` (exponential moving average with
        ``alpha = 0.3``).
    inpainting : str, optional
        Action-space inpainting applied when a new chunk is registered via
        :meth:`update_chunk`:

        * ``'none'``      – no inpainting; chunk is stored as-is (default).
        * ``'hard'``      – replace the prefix region with frozen actions
          from the previous chunk (training-time RTC, arXiv:2512.05964).
        * ``'soft_mask'`` – blend with the previous chunk using the RTC
          exponential-decay soft mask (Eq. 5, arXiv:2506.07339).
        * ``'hermite'``   – replace prefix **and** apply a cubic Hermite
          polynomial transition at the boundary for :math:`C^1` continuity.
        * ``'callback'``  – delegate to a user-supplied callable via
          *inpainting_fn* (e.g. for ΠGDM-guided inpainting).
    inpainting_fn : callable, optional
        Custom inpainting function, **required** when ``inpainting='callback'``.
        Signature::

            fn(x_new, x_old_interp, t_new, t_old, mask, delay)
                -> np.ndarray  # shape (H, D)

        where *x_new* is the raw new chunk ``(H, D)``, *x_old_interp* is the
        previous chunk interpolated at *t_new* ``(H, D)``, *t_new* and *t_old*
        are the timestamp arrays, *mask* is the soft mask ``(H,)``, and
        *delay* is the estimated delay (int).
    inpainting_transition : int, optional
        Number of action steps used for the transition region in
        ``'hermite'`` inpainting.  Ignored by other modes.  Defaults to
        ``min(4, H // 4)``.
    blend_mode : str, optional
        How to handle the transition between the old and new chunk in
        :meth:`get_output`:

        * ``'none'``      – use the new chunk directly (user handles
          continuity via inpainting).
        * ``'soft_mask'``  – blend old and new chunk values using the
          RTC soft mask weights (default).
        * ``'linear'``     – linear cross-fade over the overlap region.
    interpolation : str, optional
        Temporal interpolation within a chunk: ``'linear'`` (default) or
        ``'pchip'``.
    extrapolation : str, optional
        Behaviour when querying beyond the current chunk's time range:
        ``'linear'`` (first-order hold, default) or ``'clamp'`` (zero-order
        hold).
    """

    def __init__(
        self,
        buffer_size: int = 100,
        prediction_horizon: int = 50,
        min_execution_horizon: int = 1,
        dt: float = 0.02,
        delay_buffer_size: int = 5,
        initial_delay: int = 1,
        delay_estimate_method: str = "max",
        inpainting: str = "none",
        inpainting_fn: Optional[Callable] = None,
        inpainting_transition: Optional[int] = None,
        blend_mode: str = "soft_mask",
        interpolation: str = "linear",
        extrapolation: str = "linear",
    ) -> None:
        super().__init__(buffer_size=buffer_size)
        if prediction_horizon < 2:
            raise ValueError("prediction_horizon must be >= 2.")
        if min_execution_horizon < 1:
            raise ValueError("min_execution_horizon must be >= 1.")
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if delay_estimate_method not in ("max", "mean", "ema"):
            raise ValueError(
                f"Unknown delay_estimate_method '{delay_estimate_method}'."
            )
        if inpainting not in ("none", "hard", "soft_mask", "hermite", "callback"):
            raise ValueError(f"Unknown inpainting '{inpainting}'.")
        if inpainting == "callback" and inpainting_fn is None:
            raise ValueError(
                "inpainting_fn must be provided when inpainting='callback'."
            )
        if blend_mode not in ("none", "soft_mask", "linear"):
            raise ValueError(f"Unknown blend_mode '{blend_mode}'.")
        if interpolation not in ("linear", "pchip"):
            raise ValueError(f"Unknown interpolation '{interpolation}'.")
        if extrapolation not in ("linear", "clamp"):
            raise ValueError(f"Unknown extrapolation '{extrapolation}'.")

        self._H = int(prediction_horizon)
        self._s_min = int(min_execution_horizon)
        self._dt = float(dt)
        self._inpainting = inpainting
        self._inpainting_fn = inpainting_fn
        self._inpainting_transition = (
            int(inpainting_transition)
            if inpainting_transition is not None
            else min(4, self._H // 4)
        )
        self._blend_mode = blend_mode
        self._interpolation = interpolation
        self._extrapolation = extrapolation
        self._delay_est_method = delay_estimate_method

        # Delay estimation buffer
        self._delay_buf: deque[int] = deque(
            [max(0, int(initial_delay))], maxlen=max(1, int(delay_buffer_size))
        )

        # Chunk state
        self._curr_chunk_t: Optional[np.ndarray] = None
        self._curr_chunk_x: Optional[np.ndarray] = None
        self._prev_chunk_t: Optional[np.ndarray] = None
        self._prev_chunk_x: Optional[np.ndarray] = None

        # Timing bookkeeping
        self._chunk_switch_time: Optional[float] = None
        self._current_time: Optional[float] = None
        self._inference_start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Delay estimation
    # ------------------------------------------------------------------

    @property
    def delay_estimate(self) -> int:
        """Current estimated delay in controller timesteps (thread-safe)."""
        with self._lock:
            return self._estimate_delay()

    def _estimate_delay(self) -> int:
        """Aggregate the delay buffer (call inside lock)."""
        if not self._delay_buf:
            return 1
        if self._delay_est_method == "max":
            return int(max(self._delay_buf))
        if self._delay_est_method == "mean":
            return int(round(sum(self._delay_buf) / len(self._delay_buf)))
        # ema
        alpha = 0.3
        val = float(self._delay_buf[0])
        for d in list(self._delay_buf)[1:]:
            val = alpha * d + (1 - alpha) * val
        return max(1, int(round(val)))

    def record_delay(self, delay: int) -> None:
        """Record an observed inference delay (thread-safe).

        Parameters
        ----------
        delay : int
            Observed delay in controller timesteps.
        """
        with self._lock:
            self._delay_buf.append(max(0, int(delay)))

    # ------------------------------------------------------------------
    # Execution horizon
    # ------------------------------------------------------------------

    @property
    def execution_horizon(self) -> int:
        """Current execution horizon ``max(d, s_min)`` (thread-safe)."""
        with self._lock:
            return max(self._estimate_delay(), self._s_min)

    # ------------------------------------------------------------------
    # Current chunk accessor
    # ------------------------------------------------------------------

    @property
    def current_chunk(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Currently active ``(t, x)`` chunk, or ``None`` (thread-safe)."""
        with self._lock:
            if self._curr_chunk_t is None:
                return None
            return (self._curr_chunk_t.copy(), self._curr_chunk_x.copy())

    # ------------------------------------------------------------------
    # Soft mask
    # ------------------------------------------------------------------

    def get_soft_mask(self, delay: Optional[int] = None) -> np.ndarray:
        r"""Compute the soft mask :math:`\mathbf{W}` (Eq. 5 of arXiv:2506.07339).

        Parameters
        ----------
        delay : int, optional
            Override delay value.  If ``None``, uses the current estimate.

        Returns
        -------
        np.ndarray, shape (H,)
            Mask vector with values in ``[0, 1]``.
        """
        with self._lock:
            d = delay if delay is not None else self._estimate_delay()
        return self._compute_soft_mask(d, max(d, self._s_min))

    def _compute_soft_mask(self, d: int, s: int) -> np.ndarray:
        """Compute soft mask (no lock needed, pure function)."""
        H = self._H
        d = max(0, min(d, H - 1))
        s = max(1, min(s, H))
        W = np.zeros(H)
        denom = H - s - d + 1
        for i in range(H):
            if i < d:
                W[i] = 1.0
            elif i < H - s and denom > 0:
                c = (H - s - i) / denom
                W[i] = c * (np.exp(c) - 1.0) / (np.e - 1.0)
            # else: 0.0
        return W

    # ------------------------------------------------------------------
    # Prefix for next inference
    # ------------------------------------------------------------------

    def get_prefix(self) -> Optional[tuple[np.ndarray, np.ndarray, int]]:
        """Return the frozen action prefix for the next chunk generation.

        The prefix consists of the actions from the current chunk that will
        have been executed by the time the next chunk becomes available.

        Returns
        -------
        tuple of (t_prefix, x_prefix, delay) or None
            ``t_prefix`` – shape ``(d,)`` timestamps.
            ``x_prefix`` – shape ``(d, D)`` action values.
            ``delay``    – estimated delay in controller timesteps.
            Returns ``None`` if no chunk has been registered yet.
        """
        with self._lock:
            if self._curr_chunk_t is None:
                return None
            d = self._estimate_delay()
            s = max(d, self._s_min)
            # Prefix = first d actions counting from where we'd start
            # the next inference (after s steps have been executed).
            t_c = self._curr_chunk_t
            x_c = self._curr_chunk_x
            # The overlap region starts at index s in the current chunk
            prefix_start = min(s, len(t_c))
            prefix_end = min(s + d, len(t_c))
            if prefix_end <= prefix_start:
                # Degenerate: return the last available action
                return (
                    t_c[-1:].copy(),
                    x_c[-1:].copy(),
                    d,
                )
            return (
                t_c[prefix_start:prefix_end].copy(),
                x_c[prefix_start:prefix_end].copy(),
                d,
            )

    def start_inference(self) -> None:
        """Mark the start of an inference call (thread-safe).

        Call this right before starting model inference so that the
        observed delay can later be computed automatically when
        :meth:`update_chunk` is called.
        """
        with self._lock:
            self._inference_start_time = self._current_time

    # ------------------------------------------------------------------
    # Chunk ingestion
    # ------------------------------------------------------------------

    def update_chunk(self, t, x) -> None:  # noqa: D401
        """Register a newly generated action chunk (thread-safe).

        The previous current chunk becomes the *previous* chunk, and the
        supplied chunk becomes the *current* chunk.  If *inpainting* is
        enabled and a previous chunk exists, the new chunk is modified
        in-place before storage to enforce continuity with the frozen
        prefix.

        If :meth:`start_inference` was called beforehand, the observed
        delay is automatically recorded.

        Parameters
        ----------
        t : array-like, shape (H,)
            Timestamps for each action in the chunk.
        x : array-like, shape (H, D) or (H,)
            Action values.
        """
        t_arr = np.asarray(t, dtype=float).ravel()
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        if x_arr.shape[0] != len(t_arr):
            raise ValueError(
                f"t has {len(t_arr)} samples but x has {x_arr.shape[0]}."
            )

        with self._lock:
            if self._dim is None:
                self._dim = x_arr.shape[1]

            # Apply inpainting if enabled and a previous chunk is available
            if (
                self._inpainting != "none"
                and self._prev_chunk_t is not None
                and self._curr_chunk_t is not None
            ):
                x_arr = self._apply_inpainting(
                    t_arr, x_arr,
                    self._curr_chunk_t, self._curr_chunk_x,
                )

            # Shift chunks
            self._prev_chunk_t = self._curr_chunk_t
            self._prev_chunk_x = self._curr_chunk_x
            self._curr_chunk_t = t_arr.copy()
            self._curr_chunk_x = x_arr.copy()
            self._chunk_switch_time = self._current_time

            # Auto-record delay
            if (
                self._inference_start_time is not None
                and self._current_time is not None
            ):
                elapsed = self._current_time - self._inference_start_time
                observed_delay = max(0, int(round(elapsed / self._dt)))
                self._delay_buf.append(observed_delay)
                self._inference_start_time = None

            # Populate base buffer for compatibility
            for ti, xi in zip(t_arr, x_arr):
                self._t_buf.append(float(ti))
                self._x_buf.append(xi.copy())

    # ------------------------------------------------------------------
    # Time management
    # ------------------------------------------------------------------

    def set_current_time(self, t: float) -> None:
        """Inform the filter of the current controller time (thread-safe).

        Call this every control cycle **before** :meth:`get_output`.

        Parameters
        ----------
        t : float
            Current time in seconds.
        """
        with self._lock:
            self._current_time = float(t)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def get_output(self, t: Optional[float] = None) -> Optional[np.ndarray]:
        """Return the action at time *t* from the current chunk (thread-safe).

        If the query falls in the overlap region between the previous and
        current chunk, the output is blended according to *blend_mode*.

        Parameters
        ----------
        t : float, optional
            Query time.  Defaults to the latest chunk timestamp.

        Returns
        -------
        np.ndarray, shape (D,) or None
        """
        with self._lock:
            if self._curr_chunk_t is None or self._dim is None:
                return None

            t_c = self._curr_chunk_t
            x_c = self._curr_chunk_x
            t_query = float(t) if t is not None else float(t_c[-1])

            val_new = self._interp_chunk(t_c, x_c, t_query)

            # Blending with previous chunk
            if (
                self._blend_mode != "none"
                and self._prev_chunk_t is not None
            ):
                t_p = self._prev_chunk_t
                x_p = self._prev_chunk_x
                overlap_start = max(t_c[0], t_p[0])
                overlap_end = min(t_c[-1], t_p[-1])
                if overlap_start < overlap_end and overlap_start <= t_query <= overlap_end:
                    val_old = self._interp_chunk(t_p, x_p, t_query)
                    alpha = self._blend_weight(
                        t_query, overlap_start, overlap_end
                    )
                    return (1.0 - alpha) * val_old + alpha * val_new

            return val_new

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _interp_chunk(
        self, t_c: np.ndarray, x_c: np.ndarray, t_query: float
    ) -> np.ndarray:
        """Interpolate within a single chunk."""
        D = x_c.shape[1]

        # Extrapolation handling
        if t_query <= t_c[0]:
            if self._extrapolation == "clamp":
                return x_c[0].copy()
            # linear: first-order hold backwards
            if len(t_c) >= 2:
                dt_c = t_c[1] - t_c[0]
                if dt_c > 0:
                    slope = (x_c[1] - x_c[0]) / dt_c
                    return x_c[0] + slope * (t_query - t_c[0])
            return x_c[0].copy()
        if t_query >= t_c[-1]:
            if self._extrapolation == "clamp":
                return x_c[-1].copy()
            if len(t_c) >= 2:
                dt_c = t_c[-1] - t_c[-2]
                if dt_c > 0:
                    slope = (x_c[-1] - x_c[-2]) / dt_c
                    return x_c[-1] + slope * (t_query - t_c[-1])
            return x_c[-1].copy()

        if self._interpolation == "pchip" and len(t_c) >= 4:
            return np.array(
                [
                    float(
                        sp_interpolate.PchipInterpolator(t_c, x_c[:, d])(
                            t_query
                        )
                    )
                    for d in range(D)
                ]
            )
        # linear
        return np.array(
            [float(np.interp(t_query, t_c, x_c[:, d])) for d in range(D)]
        )

    def _blend_weight(
        self, t_query: float, t_start: float, t_end: float
    ) -> float:
        """Return blend weight for the *new* chunk (0 = all old, 1 = all new).

        Uses the soft mask schedule or linear ramp depending on blend_mode.
        """
        if t_end <= t_start:
            return 1.0
        frac = (t_query - t_start) / (t_end - t_start)
        frac = float(np.clip(frac, 0.0, 1.0))

        if self._blend_mode == "linear":
            return frac

        # soft_mask: exponential transition matching RTC Eq. 5 spirit
        if frac >= 1.0:
            return 1.0
        c = 1.0 - frac  # c goes 1 → 0 as frac goes 0 → 1
        old_weight = c * (np.exp(c) - 1.0) / (np.e - 1.0)
        return 1.0 - old_weight

    # ------------------------------------------------------------------
    # Action-space inpainting
    # ------------------------------------------------------------------

    def _apply_inpainting(
        self,
        t_new: np.ndarray,
        x_new: np.ndarray,
        t_old: np.ndarray,
        x_old: np.ndarray,
    ) -> np.ndarray:
        """Apply action-space inpainting to the incoming chunk.

        Modifies *x_new* so that it is consistent with the frozen
        prefix taken from the old (currently executing) chunk.

        Must be called **inside** the lock.

        Parameters
        ----------
        t_new, x_new : new chunk timestamps and values.
        t_old, x_old : currently executing chunk timestamps and values.

        Returns
        -------
        np.ndarray – inpainted copy of *x_new* (shape ``(N, D)``).
        """
        x_out = x_new.copy()
        N = len(t_new)
        D = x_new.shape[1]

        # Get previous chunk's values at the new chunk's timestamps
        # (interpolated where timestamps don't align exactly).
        overlap_end = min(float(t_new[-1]), float(t_old[-1]))
        overlap_start = float(t_new[0])
        if overlap_start >= overlap_end:
            return x_out  # no overlap → nothing to inpaint

        # Interpolate old chunk at new chunk timestamps
        x_old_at_new = np.column_stack(
            [np.interp(t_new, t_old, x_old[:, d]) for d in range(D)]
        )

        # Build an overlap mask: True where t_new is within the old chunk range
        in_overlap = (t_new >= t_old[0]) & (t_new <= t_old[-1])

        d = self._estimate_delay()
        s = max(d, self._s_min)

        if self._inpainting == "hard":
            self._inpaint_hard(x_out, x_old_at_new, in_overlap, d)
        elif self._inpainting == "soft_mask":
            self._inpaint_soft_mask(x_out, x_old_at_new, in_overlap, d, s)
        elif self._inpainting == "hermite":
            self._inpaint_hermite(
                t_new, x_out, x_old_at_new, t_old, x_old, in_overlap, d,
            )
        elif self._inpainting == "callback":
            mask = self._compute_soft_mask(d, s)
            x_out = np.asarray(
                self._inpainting_fn(
                    x_out, x_old_at_new, t_new, t_old, mask, d,
                ),
                dtype=float,
            )
        return x_out

    def _inpaint_hard(
        self,
        x_out: np.ndarray,
        x_old_at_new: np.ndarray,
        in_overlap: np.ndarray,
        d: int,
    ) -> None:
        """Hard prefix replacement (training-time RTC style).

        Replace the first *d* overlapping actions with frozen values
        from the previous chunk.
        """
        overlap_indices = np.where(in_overlap)[0]
        n_replace = min(d, len(overlap_indices))
        if n_replace > 0:
            idx = overlap_indices[:n_replace]
            x_out[idx] = x_old_at_new[idx]

    def _inpaint_soft_mask(
        self,
        x_out: np.ndarray,
        x_old_at_new: np.ndarray,
        in_overlap: np.ndarray,
        d: int,
        s: int,
    ) -> None:
        """Soft mask inpainting (inference-time RTC, Eq. 5).

        Blend old and new chunk values using the exponentially decaying
        soft mask across the entire overlap region.
        """
        overlap_indices = np.where(in_overlap)[0]
        n_overlap = len(overlap_indices)
        if n_overlap == 0:
            return

        W = self._compute_soft_mask(d, s)

        for k, idx in enumerate(overlap_indices):
            # Map overlap index k to soft mask index
            if k < len(W):
                w = W[k]
            else:
                w = 0.0
            x_out[idx] = w * x_old_at_new[idx] + (1.0 - w) * x_out[idx]

    def _inpaint_hermite(
        self,
        t_new: np.ndarray,
        x_out: np.ndarray,
        x_old_at_new: np.ndarray,
        t_old: np.ndarray,
        x_old: np.ndarray,
        in_overlap: np.ndarray,
        d: int,
    ) -> None:
        """Prefix replacement + cubic Hermite boundary smoothing.

        1. Replace the first *d* overlapping actions with frozen values.
        2. Apply a cubic Hermite polynomial transition over a short
           region after the prefix boundary to ensure :math:`C^1`
           continuity (matching position and velocity).
        """
        overlap_indices = np.where(in_overlap)[0]
        n_replace = min(d, len(overlap_indices))
        if n_replace == 0:
            return

        # Step 1: hard-replace prefix
        replace_idx = overlap_indices[:n_replace]
        x_out[replace_idx] = x_old_at_new[replace_idx]

        # Step 2: Hermite blend over transition region after the prefix
        boundary_idx = replace_idx[-1]  # last replaced index
        trans_len = max(1, self._inpainting_transition)
        trans_start = boundary_idx
        trans_end = min(boundary_idx + trans_len, len(t_new) - 1)
        if trans_end <= trans_start:
            return

        D = x_out.shape[1]
        t_b = t_new[trans_start]
        t_e = t_new[trans_end]
        dt_region = t_e - t_b
        if dt_region <= 0:
            return

        # Boundary conditions from the old chunk (position + velocity)
        p0 = x_old_at_new[trans_start].copy()  # position at start
        if trans_start > 0:
            dt_prev = t_new[trans_start] - t_new[trans_start - 1]
            if dt_prev > 0:
                v0 = (x_old_at_new[trans_start] - x_old_at_new[trans_start - 1]) / dt_prev
            else:
                v0 = np.zeros(D)
        else:
            # Estimate velocity from old chunk directly
            if len(t_old) >= 2:
                dt_old = t_old[-1] - t_old[-2]
                v0 = (x_old[-1] - x_old[-2]) / dt_old if dt_old > 0 else np.zeros(D)
            else:
                v0 = np.zeros(D)

        # Target: the raw (un-inpainted) new chunk values at transition end
        p1 = x_out[trans_end].copy()
        if trans_end + 1 < len(t_new):
            dt_next = t_new[trans_end + 1] - t_new[trans_end]
            if dt_next > 0:
                v1 = (x_out[trans_end + 1] - x_out[trans_end]) / dt_next
            else:
                v1 = np.zeros(D)
        elif trans_end > 0:
            dt_prev2 = t_new[trans_end] - t_new[trans_end - 1]
            if dt_prev2 > 0:
                v1 = (x_out[trans_end] - x_out[trans_end - 1]) / dt_prev2
            else:
                v1 = np.zeros(D)
        else:
            v1 = np.zeros(D)

        # Cubic Hermite interpolation for indices in (trans_start, trans_end]
        for i in range(trans_start + 1, trans_end + 1):
            s = (t_new[i] - t_b) / dt_region  # normalised [0, 1]
            # Hermite basis functions
            h00 = 2 * s**3 - 3 * s**2 + 1
            h10 = s**3 - 2 * s**2 + s
            h01 = -2 * s**3 + 3 * s**2
            h11 = s**3 - s**2
            x_out[i] = (
                h00 * p0 + h10 * dt_region * v0
                + h01 * p1 + h11 * dt_region * v1
            )

    # -- hooks / ABC --------------------------------------------------------

    def _on_clear(self) -> None:
        self._curr_chunk_t = None
        self._curr_chunk_x = None
        self._prev_chunk_t = None
        self._prev_chunk_x = None
        self._chunk_switch_time = None
        self._current_time = None
        self._inference_start_time = None
        self._delay_buf.clear()

    def _compute(
        self, t: np.ndarray, x: np.ndarray, t_query: float
    ) -> np.ndarray:
        # Delegate to chunk-aware output
        if self._curr_chunk_t is not None:
            return self._interp_chunk(
                self._curr_chunk_t, self._curr_chunk_x, t_query
            )
        # Fallback: linear interpolation on the raw buffer
        D = x.shape[1]
        return np.array(
            [float(np.interp(t_query, t, x[:, d])) for d in range(D)]
        )


# ======================================================================
# Composite: RTC + RAIL
# ======================================================================


class AsyncFilterRTCRAIL(AsyncFilterRTC):
    """RTC inpainting + RAIL polynomial smoothing composite filter.

    Combines the strengths of both filters in a two-stage pipeline:

    1. **RTC stage** – incoming action chunks are pre-processed with
       action-space inpainting (hard / soft_mask / hermite / callback) so
       the prefix region is continuous with currently-executing actions.
    2. **RAIL stage** – the inpainted chunk is polynomial-fitted
       (intra-chunk noise suppression) and blended at chunk boundaries
       with C¹ or C² continuity.

    :meth:`get_output` returns RAIL's smooth polynomial evaluation.
    All RTC functionality (:meth:`get_prefix`, :meth:`start_inference`,
    :attr:`delay_estimate`, :meth:`get_soft_mask`) remains available.

    Parameters
    ----------
    buffer_size : int, optional
        Per-sample buffer capacity (default: ``100``).

    prediction_horizon : int
        :math:`H` – action steps per chunk.
    min_execution_horizon : int, optional
        :math:`s_{\\min}` – minimum steps before next inference (default: ``1``).
    dt : float
        Controller period in seconds.
    delay_buffer_size : int, optional
        Past delays kept for estimation (default: ``5``).
    initial_delay : int, optional
        Seed value for delay buffer (default: ``1``).
    delay_estimate_method : str, optional
        ``'max'`` (default), ``'mean'``, or ``'ema'``.
    inpainting : str, optional
        RTC inpainting mode (default: ``'none'``).
    inpainting_fn : callable, optional
        Custom inpainting function for ``inpainting='callback'``.
    inpainting_transition : int, optional
        Hermite transition length.
    blend_mode : str, optional
        RTC blend mode — only used as a **fallback** when RAIL has not
        yet received a second chunk.  Default: ``'soft_mask'``.
    interpolation : str, optional
        ``'linear'`` (default) or ``'pchip'`` – RTC fallback interpolation.
    extrapolation : str, optional
        ``'linear'`` (default) or ``'clamp'`` – RTC fallback extrapolation.

    poly_degree : int, optional
        RAIL polynomial degree for intra-chunk smoothing (default: ``3``).
    blend_duration : float or None, optional
        RAIL inter-chunk blend duration in seconds.  ``None`` = 30 %
        of chunk duration.
    dual_quintic : bool, optional
        RAIL split-blend (default: ``True``).
    blend_order : str, optional
        RAIL boundary continuity: ``'cubic'`` (C¹, default) or
        ``'quintic'`` (C²).
    acc_clamp : float or None, optional
        RAIL acceleration clamping for quintic blends.
    rail_extrapolation : str, optional
        RAIL extrapolation beyond fitted range (default: ``'linear'``).
    auto_align : bool, optional
        RAIL temporal alignment (default: ``False``).
    align_method : str, optional
        RAIL alignment algorithm (default: ``'direction'``).
    align_window : float or None, optional
        RAIL alignment search window.
    blend_start_source : str, optional
        RAIL blend start conditions (default: ``'actual_output'``).
    """

    def __init__(
        self,
        buffer_size: int = 100,
        # --- RTC parameters ---
        prediction_horizon: int = 50,
        min_execution_horizon: int = 1,
        dt: float = 0.02,
        delay_buffer_size: int = 5,
        initial_delay: int = 1,
        delay_estimate_method: str = "max",
        inpainting: str = "none",
        inpainting_fn: Optional[Callable] = None,
        inpainting_transition: Optional[int] = None,
        blend_mode: str = "soft_mask",
        interpolation: str = "linear",
        extrapolation: str = "linear",
        # --- RAIL parameters ---
        poly_degree: int = 3,
        blend_duration: Optional[float] = None,
        dual_quintic: bool = True,
        blend_order: str = "cubic",
        acc_clamp: Optional[float] = None,
        rail_extrapolation: str = "linear",
        auto_align: bool = False,
        align_method: str = "direction",
        align_window: Optional[float] = None,
        blend_start_source: str = "actual_output",
    ) -> None:
        super().__init__(
            buffer_size=buffer_size,
            prediction_horizon=prediction_horizon,
            min_execution_horizon=min_execution_horizon,
            dt=dt,
            delay_buffer_size=delay_buffer_size,
            initial_delay=initial_delay,
            delay_estimate_method=delay_estimate_method,
            inpainting=inpainting,
            inpainting_fn=inpainting_fn,
            inpainting_transition=inpainting_transition,
            blend_mode=blend_mode,
            interpolation=interpolation,
            extrapolation=extrapolation,
        )
        self._rail = AsyncFilterRAIL(
            buffer_size=buffer_size,
            poly_degree=poly_degree,
            blend_duration=blend_duration,
            dual_quintic=dual_quintic,
            auto_align=auto_align,
            align_method=align_method,
            align_window=align_window,
            blend_start_source=blend_start_source,
            blend_order=blend_order,
            acc_clamp=acc_clamp,
            extrapolation=rail_extrapolation,
        )

    # ------------------------------------------------------------------
    # Chunk ingestion: RTC inpainting → RAIL polynomial fitting
    # ------------------------------------------------------------------

    def update_chunk(self, t, x) -> None:  # noqa: D401
        """Register a chunk: apply RTC inpainting, then RAIL smoothing.

        1. The RTC stage applies action-space inpainting (if enabled) so
           the chunk prefix matches currently-executing actions.
        2. The inpainted chunk is fed to RAIL for polynomial fitting and
           boundary blending.

        Parameters
        ----------
        t : array-like, shape (H,)
            Timestamps for each action in the chunk.
        x : array-like, shape (H, D) or (H,)
            Action values.
        """
        t_arr = np.asarray(t, dtype=float).ravel()
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        if x_arr.shape[0] != len(t_arr):
            raise ValueError(
                f"t has {len(t_arr)} samples but x has {x_arr.shape[0]}."
            )

        with self._lock:
            if self._dim is None:
                self._dim = x_arr.shape[1]

            # Stage 1: RTC inpainting
            if (
                self._inpainting != "none"
                and self._prev_chunk_t is not None
                and self._curr_chunk_t is not None
            ):
                x_arr = self._apply_inpainting(
                    t_arr, x_arr,
                    self._curr_chunk_t, self._curr_chunk_x,
                )

            # Shift RTC chunk state
            self._prev_chunk_t = self._curr_chunk_t
            self._prev_chunk_x = self._curr_chunk_x
            self._curr_chunk_t = t_arr.copy()
            self._curr_chunk_x = x_arr.copy()
            self._chunk_switch_time = self._current_time

            # Auto-record delay
            if (
                self._inference_start_time is not None
                and self._current_time is not None
            ):
                elapsed = self._current_time - self._inference_start_time
                observed_delay = max(0, int(round(elapsed / self._dt)))
                self._delay_buf.append(observed_delay)
                self._inference_start_time = None

            # Populate base buffer for compatibility
            for ti, xi in zip(t_arr, x_arr):
                self._t_buf.append(float(ti))
                self._x_buf.append(xi.copy())

        # Stage 2: RAIL polynomial fitting + boundary blending
        self._rail.update_chunk(t_arr, x_arr)

    # ------------------------------------------------------------------
    # Output: RAIL polynomial evaluation with RTC fallback
    # ------------------------------------------------------------------

    def get_output(self, t: Optional[float] = None) -> Optional[np.ndarray]:
        """Return the smoothed action at time *t*.

        Delegates to RAIL's polynomial evaluation for smooth output.
        Falls back to RTC's interpolation only if RAIL has not yet
        received any chunk.

        Parameters
        ----------
        t : float, optional
            Query time.  Defaults to the latest trajectory endpoint.

        Returns
        -------
        np.ndarray, shape (D,) or None
        """
        rail_out = self._rail.get_output(t)
        if rail_out is not None:
            return rail_out
        return super().get_output(t)

    # ------------------------------------------------------------------
    # Prefix: use RAIL-smoothed values for consistency
    # ------------------------------------------------------------------

    def get_prefix(self) -> Optional[tuple[np.ndarray, np.ndarray, int]]:
        """Return the frozen prefix re-evaluated through RAIL polynomials.

        Unlike the base :class:`AsyncFilterRTC` which returns the raw
        (inpainted) chunk values, this method evaluates prefix timestamps
        against RAIL's polynomial trajectory.  This ensures that the
        prefix handed to the VLA model matches what the robot actually
        executed via :meth:`get_output`.

        Falls back to the raw RTC prefix when RAIL has not yet built a
        trajectory (e.g. before the first chunk).

        Returns
        -------
        tuple of (t_prefix, x_prefix, delay) or None
            ``t_prefix`` – shape ``(d,)`` timestamps.
            ``x_prefix`` – shape ``(d, D)`` RAIL-smoothed action values.
            ``delay``    – estimated delay in controller timesteps.
            Returns ``None`` if no chunk has been registered yet.
        """
        # Step 1: Compute prefix timestamps and delay under RTC lock.
        with self._lock:
            if self._curr_chunk_t is None:
                return None
            d = self._estimate_delay()
            s = max(d, self._s_min)
            t_c = self._curr_chunk_t
            prefix_start = min(s, len(t_c))
            prefix_end = min(s + d, len(t_c))
            if prefix_end <= prefix_start:
                t_prefix = t_c[-1:].copy()
            else:
                t_prefix = t_c[prefix_start:prefix_end].copy()

        # Step 2: Re-evaluate prefix values via RAIL polynomial (its own lock).
        x_list = [self._rail.get_output(float(ti)) for ti in t_prefix]

        if any(v is None for v in x_list):
            # RAIL not ready yet — fall back to raw RTC prefix.
            return super().get_prefix()

        x_prefix = np.array(x_list)
        return (t_prefix, x_prefix, d)

    # ------------------------------------------------------------------
    # Time management: propagate to both stages
    # ------------------------------------------------------------------

    def set_current_time(self, t: float) -> None:
        """Inform both RTC and RAIL stages of current time."""
        super().set_current_time(t)
        self._rail.set_current_time(t)


# ======================================================================
# Factory
# ======================================================================

_METHOD_MAP: dict[str, type[AsyncFilterBase]] = {
    "ema": AsyncFilterEMA,
    "linear": AsyncFilterLinear,
    "spline": AsyncFilterSpline,
    "one_euro": AsyncFilterOneEuro,
    "moving_average": AsyncFilterMovingAverage,
    "act": AsyncFilterACT,
    "rail": AsyncFilterRAIL,
    "rtc": AsyncFilterRTC,
    "rtc_rail": AsyncFilterRTCRAIL,
}


def AsyncFilter(
    method: str = "ema",
    buffer_size: int = 100,
    **kwargs,
) -> AsyncFilterBase:
    """Create an asynchronous filter instance (factory function).

    Parameters
    ----------
    method : str, optional
        Smoothing method.  One of:

        - ``'ema'``            – Exponential moving average (default).
        - ``'linear'``         – Linear interpolation over the circular buffer.
        - ``'spline'``         – Cubic-spline interpolation over the buffer.
        - ``'one_euro'``       – Adaptive low-pass (1€ filter).
        - ``'moving_average'`` – Sliding-window average.
        - ``'act'``            – Temporal ensembling of overlapping action chunks.
        - ``'rail'``           – Two-stage trajectory post-processing with C² fusion.
        - ``'rtc'``            – Real-Time Chunking async execution framework.
        - ``'rtc_rail'``       – RTC inpainting + RAIL polynomial smoothing.

    buffer_size : int, optional
        Maximum buffer capacity (default: ``100``).

    **kwargs
        Method-specific arguments (e.g. ``alpha`` for ``'ema'``).

    Returns
    -------
    AsyncFilterBase

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    cls = _METHOD_MAP.get(method)
    if cls is None:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Choose from: {', '.join(repr(k) for k in _METHOD_MAP)}."
        )
    return cls(buffer_size=buffer_size, **kwargs)
