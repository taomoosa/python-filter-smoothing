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
from typing import Optional

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
    """

    __slots__ = ("t_start", "t_end", "_polys")

    def __init__(
        self, t_start: float, t_end: float, polys: list[np.poly1d]
    ) -> None:
        self.t_start = t_start
        self.t_end = t_end
        self._polys = polys

    @classmethod
    def fit(
        cls,
        t: np.ndarray,
        x: np.ndarray,
        degree: int,
    ) -> "_PolyTrajectory":
        """Fit polynomial of given *degree* to each dimension of *x*."""
        n_dims = x.shape[1]
        deg = min(degree, len(t) - 1)
        polys = [np.poly1d(np.polyfit(t, x[:, d], deg)) for d in range(n_dims)]
        return cls(float(t[0]), float(t[-1]), polys)

    def evaluate(self, t: float) -> np.ndarray:
        """Evaluate trajectory at time *t*."""
        return np.array([float(p(t)) for p in self._polys])

    def evaluate_deriv(self, t: float, order: int = 1) -> np.ndarray:
        """Evaluate *order*-th derivative of trajectory at time *t*."""
        result = np.empty(len(self._polys))
        for d, p in enumerate(self._polys):
            dp = p
            for _ in range(order):
                dp = dp.deriv()
            result[d] = float(dp(t))
        return result


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
    ) -> None:
        self.t_start = t_start
        self.t_end = t_end

        # Boundary conditions from old and new trajectories
        p0 = traj_old.evaluate(t_start)
        v0 = traj_old.evaluate_deriv(t_start, 1)
        a0 = traj_old.evaluate_deriv(t_start, 2)

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
    ) -> None:
        self.t_start = t_start
        self.t_end = t_end
        self._t_mid = 0.5 * (t_start + t_end)

        # Old trajectory at blend start
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
    2. **Inter-chunk fusion** – constructs a quintic polynomial blend at
       chunk boundaries, ensuring C² continuity (position, velocity, and
       acceleration) between successive trajectories.

    Based on VLA-RAIL (Zhao et al., arXiv:2512.24673).

    Parameters
    ----------
    buffer_size : int, optional
        Per-sample buffer capacity (default: ``100``).
    poly_degree : int, optional
        Polynomial degree for intra-chunk smoothing (default: ``3``, i.e.
        cubic, as recommended by the VLA-RAIL paper).
    blend_duration : float or None, optional
        Duration (seconds) of the inter-chunk quintic blend region.
        If ``None`` (default), uses 30 % of the new chunk's duration.
    dual_quintic : bool, optional
        If ``True`` (default), use dual-quintic spline interpolation
        (VLA-RAIL Eq. 11-13) which splits the blend region into two
        halves to avoid overshoot from Runge's phenomenon.
        If ``False``, use a single quintic blend over the whole region.
    auto_align : bool, optional
        If ``True``, automatically correct new-chunk timestamps via
        temporal alignment optimisation (VLA-RAIL Eq. 10) that maximises
        motion-direction consistency with the current trajectory.
        Requires :meth:`set_current_time` to have been called.
        Default: ``False``.
    align_window : float or None, optional
        Search window (seconds) for temporal alignment.  If ``None``
        (default), uses 50 % of the new chunk's duration.
    """

    def __init__(
        self,
        buffer_size: int = 100,
        poly_degree: int = 3,
        blend_duration: Optional[float] = None,
        dual_quintic: bool = True,
        auto_align: bool = False,
        align_window: Optional[float] = None,
    ) -> None:
        super().__init__(buffer_size=buffer_size)
        self._poly_degree = int(poly_degree)
        self._blend_duration = blend_duration
        self._dual_quintic = bool(dual_quintic)
        self._auto_align = bool(auto_align)
        self._align_window = align_window
        self._current_time: Optional[float] = None
        self._prev_traj: Optional[_PolyTrajectory] = None
        self._curr_traj: Optional[_PolyTrajectory] = None
        # Blend object: either _QuinticBlend or _DualQuinticBlend
        self._blend: Optional[_QuinticBlend | _DualQuinticBlend] = None

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

    # -- chunk interface ----------------------------------------------------

    def update_chunk(self, t, x) -> None:
        """Add an action chunk with polynomial smoothing and C² fusion."""
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
            traj = _PolyTrajectory.fit(t_arr, x_arr, self._poly_degree)

            # Optional: temporal alignment (shift chunk timestamps)
            if self._auto_align and self._current_time is not None:
                shift = self._compute_temporal_alignment(
                    traj, self._current_time
                )
                if abs(shift) > 1e-12:
                    t_shifted = t_arr + shift
                    traj = _PolyTrajectory.fit(
                        t_shifted, x_arr, self._poly_degree
                    )

            # Stage 2: Inter-chunk fusion (quintic C² blend)
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
                    BlendCls = (
                        _DualQuinticBlend
                        if self._dual_quintic
                        else _QuinticBlend
                    )
                    self._blend = BlendCls(
                        self._prev_traj, traj, t_bs, t_be
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
                return self._blend.evaluate(t_query)

            # Before blend start: use previous (old) trajectory
            if (
                self._blend is not None
                and t_query < self._blend.t_start
                and self._prev_traj is not None
            ):
                return self._prev_traj.evaluate(t_query)

            # After blend (or no blend): use current trajectory
            if t_query >= self._curr_traj.t_start:
                return self._curr_traj.evaluate(t_query)

            # Before current trajectory: use previous if available
            if self._prev_traj is not None:
                return self._prev_traj.evaluate(t_query)

            return self._curr_traj.evaluate(t_query)

    # -- hooks / ABC --------------------------------------------------------

    def _on_clear(self) -> None:
        self._prev_traj = None
        self._curr_traj = None
        self._blend = None
        self._current_time = None

    def _compute(
        self, t: np.ndarray, x: np.ndarray, t_query: float
    ) -> np.ndarray:
        if self._curr_traj is not None:
            return self._curr_traj.evaluate(t_query)
        return np.zeros(self._dim)  # pragma: no cover


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
