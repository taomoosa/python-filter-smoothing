"""Offline time series filtering and smoothing.

The entire time series is provided at once and processed in batch.
Each data point at each timestamp is a real-valued scalar or vector.
"""
from __future__ import annotations

import numpy as np
from scipy import interpolate as sp_interpolate
from scipy import signal as sp_signal
from scipy.ndimage import gaussian_filter1d, median_filter as _nd_median


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

        if len(self.t) == 0:
            raise ValueError("Cannot create filter with empty data")
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

    def savgol_filter(
        self,
        window_length: int,
        polyorder: int,
    ) -> np.ndarray:
        """Apply a Savitzky-Golay smoothing filter.

        A local polynomial regression (least-squares) is fitted within a
        sliding window.  Unlike a simple moving average, this preserves
        higher-order moments of the signal such as peak heights and widths.

        Parameters
        ----------
        window_length : int
            Length of the filter window (must be a positive odd integer).
        polyorder : int
            Order of the polynomial used to fit the samples.  Must be
            less than ``window_length``.

        Returns
        -------
        np.ndarray, shape (N,) or (N, D)
            Smoothed data.
        """
        result = np.column_stack(
            [
                sp_signal.savgol_filter(self.x[:, d], window_length, polyorder)
                for d in range(self.n_dims)
            ]
        )
        return self._squeeze_output(result)

    def gaussian_filter(self, sigma: float) -> np.ndarray:
        """Apply a 1-D Gaussian smoothing filter.

        Convolves the data with a Gaussian kernel of the given standard
        deviation.  Effective for general-purpose noise reduction.

        Parameters
        ----------
        sigma : float
            Standard deviation of the Gaussian kernel (in samples).

        Returns
        -------
        np.ndarray, shape (N,) or (N, D)
            Smoothed data.
        """
        result = np.column_stack(
            [gaussian_filter1d(self.x[:, d], sigma) for d in range(self.n_dims)]
        )
        return self._squeeze_output(result)

    def median_filter(self, kernel_size: int) -> np.ndarray:
        """Apply a 1-D median filter.

        Each output sample is the median of a sliding window centred on
        that sample.  Highly robust to impulsive / outlier noise.

        Parameters
        ----------
        kernel_size : int
            Size of the median filter window (must be a positive odd
            integer).

        Returns
        -------
        np.ndarray, shape (N,) or (N, D)
            Filtered data.
        """
        result = np.column_stack(
            [
                _nd_median(self.x[:, d], size=kernel_size)
                for d in range(self.n_dims)
            ]
        )
        return self._squeeze_output(result)

    def moving_average(self, window_size: int) -> np.ndarray:
        """Apply a centred moving-average (box-car) filter.

        Uses a uniform convolution kernel of length ``window_size``.
        Edge effects are handled by truncating the kernel at both ends
        (``mode='same'``).

        Parameters
        ----------
        window_size : int
            Number of samples in the averaging window.

        Returns
        -------
        np.ndarray, shape (N,) or (N, D)
            Smoothed data.
        """
        kernel = np.ones(window_size) / window_size
        result = np.column_stack(
            [
                np.convolve(self.x[:, d], kernel, mode="same")
                for d in range(self.n_dims)
            ]
        )
        return self._squeeze_output(result)

    # ------------------------------------------------------------------
    # FIR filter
    # ------------------------------------------------------------------

    def fir_filter(
        self,
        numtaps: int,
        cutoff_freq: float | list[float],
        sample_rate: float,
        window: str = "hamming",
        pass_zero: bool | str = True,
    ) -> np.ndarray:
        """Apply a zero-phase FIR filter.

        Designs an FIR filter with :func:`scipy.signal.firwin` and applies
        it with :func:`scipy.signal.filtfilt` for zero-phase filtering.

        Parameters
        ----------
        numtaps : int
            Length of the FIR filter (number of coefficients).
            Must be odd for zero-phase ``filtfilt`` to work well.
        cutoff_freq : float or list of float
            Cutoff frequency (or frequencies for bandpass/bandstop) in the
            same units as ``sample_rate``.
        sample_rate : float
            Sampling rate of the data.
        window : str, optional
            Window function for filter design (default: ``'hamming'``).
            See :func:`scipy.signal.get_window` for options.
        pass_zero : bool or str, optional
            If ``True``, the DC component passes (lowpass-like).
            If ``False``, the DC component is blocked (highpass-like).
            Can also be ``'bandpass'`` or ``'bandstop'`` for multi-band
            designs (default: ``True``).

        Returns
        -------
        np.ndarray, shape (N,) or (N, D)
            Filtered data.
        """
        b = sp_signal.firwin(
            numtaps, cutoff_freq, fs=sample_rate,
            window=window, pass_zero=pass_zero,
        )
        result = np.column_stack(
            [sp_signal.filtfilt(b, [1.0], self.x[:, d]) for d in range(self.n_dims)]
        )
        return self._squeeze_output(result)

    # ------------------------------------------------------------------
    # General IIR filter (Butterworth, Chebyshev, Elliptic, Bessel)
    # ------------------------------------------------------------------

    _IIR_TYPES = ("butterworth", "chebyshev1", "chebyshev2", "elliptic", "bessel")

    def iir_filter(
        self,
        cutoff_freq: float | list[float],
        sample_rate: float,
        order: int = 4,
        iir_type: str = "butterworth",
        btype: str = "low",
        rp: float | None = None,
        rs: float | None = None,
    ) -> np.ndarray:
        """Apply a zero-phase IIR filter.

        Supports multiple IIR filter families in SOS (second-order sections)
        form, applied with :func:`scipy.signal.sosfiltfilt` for zero-phase
        filtering.

        Parameters
        ----------
        cutoff_freq : float or list of float
            Cutoff frequency (or pair for band filters) in the same units
            as ``sample_rate``.
        sample_rate : float
            Sampling rate of the data.
        order : int, optional
            Filter order (default: ``4``).
        iir_type : str, optional
            IIR filter family.  One of ``'butterworth'``, ``'chebyshev1'``,
            ``'chebyshev2'``, ``'elliptic'``, ``'bessel'``
            (default: ``'butterworth'``).
        btype : str, optional
            Band type: ``'low'``, ``'high'``, ``'bandpass'``, ``'bandstop'``
            (default: ``'low'``).
        rp : float, optional
            Maximum ripple in the passband (dB).  Required for
            ``'chebyshev1'`` and ``'elliptic'``.
        rs : float, optional
            Minimum attenuation in the stopband (dB).  Required for
            ``'chebyshev2'`` and ``'elliptic'``.

        Returns
        -------
        np.ndarray, shape (N,) or (N, D)
            Filtered data.

        Raises
        ------
        ValueError
            If ``iir_type`` is unknown or required parameters are missing.
        """
        sos = _design_iir_sos(
            cutoff_freq, sample_rate, order, iir_type, btype, rp, rs,
        )
        result = np.column_stack(
            [sp_signal.sosfiltfilt(sos, self.x[:, d]) for d in range(self.n_dims)]
        )
        return self._squeeze_output(result)

    # ------------------------------------------------------------------
    # Kalman smoother (RTS)
    # ------------------------------------------------------------------

    def kalman_smooth(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        state_model: str = "position",
        dt: float | None = None,
        F: np.ndarray | None = None,
        H: np.ndarray | None = None,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply a Kalman smoother (Rauch-Tung-Striebel).

        Runs a forward Kalman filter followed by a backward RTS smoothing
        pass for optimal offline state estimation.

        Two built-in state models are provided for convenience.  For full
        control, pass custom ``F``, ``H``, ``Q``, ``R`` matrices.

        Parameters
        ----------
        process_noise : float, optional
            Scalar process noise variance used to build ``Q`` when a
            built-in model is selected (default: ``0.01``).
        measurement_noise : float, optional
            Scalar measurement noise variance used to build ``R`` when a
            built-in model is selected (default: ``0.1``).
        state_model : str, optional
            Built-in state model.  One of:

            - ``'position'``  – random-walk model (state = position).
            - ``'position_velocity'`` – constant-velocity model
              (state = [position, velocity]).

            Ignored when custom ``F``, ``H``, ``Q``, ``R`` are all
            provided (default: ``'position'``).
        dt : float, optional
            Time step for the ``'position_velocity'`` model.  If ``None``,
            estimated from the data timestamps.
        F : ndarray, optional
            State transition matrix.  Shape ``(S, S)`` where ``S`` is the
            state dimension.
        H : ndarray, optional
            Observation matrix.  Shape ``(D, S)`` where ``D`` is the
            measurement dimension.
        Q : ndarray, optional
            Process noise covariance.  Shape ``(S, S)``.
        R : ndarray, optional
            Measurement noise covariance.  Shape ``(D, D)``.
        x0 : ndarray, optional
            Initial state estimate.  Shape ``(S,)``.
        P0 : ndarray, optional
            Initial state covariance.  Shape ``(S, S)``.

        Returns
        -------
        np.ndarray, shape (N,) or (N, D)
            Smoothed data (position components of the state).
        """
        D = self.n_dims
        N = self.n_samples

        if F is not None and H is not None and Q is not None and R is not None:
            F, H, Q, R = (
                np.asarray(F, dtype=float),
                np.asarray(H, dtype=float),
                np.asarray(Q, dtype=float),
                np.asarray(R, dtype=float),
            )
            S = F.shape[0]
        else:
            if dt is None:
                dt = float(np.median(np.diff(self.t))) if N > 1 else 1.0
            F, H, Q, R, S = _build_kalman_model(
                D, dt, state_model, process_noise, measurement_noise,
            )

        # --- initial state ---
        if x0 is None:
            x0 = np.zeros(S)
            x0[:D] = self.x[0]
        else:
            x0 = np.asarray(x0, dtype=float)
        if P0 is None:
            P0 = np.eye(S) * measurement_noise * 10.0

        # --- forward Kalman filter ---
        x_pred = np.zeros((N, S))
        P_pred = np.zeros((N, S, S))
        x_filt = np.zeros((N, S))
        P_filt = np.zeros((N, S, S))

        for k in range(N):
            if k == 0:
                xp, Pp = x0, P0
            else:
                xp = F @ x_filt[k - 1]
                Pp = F @ P_filt[k - 1] @ F.T + Q
            x_pred[k] = xp
            P_pred[k] = Pp

            # update
            z = self.x[k]
            y_innov = z - H @ xp
            S_innov = H @ Pp @ H.T + R
            K = Pp @ H.T @ np.linalg.inv(S_innov)
            x_filt[k] = xp + K @ y_innov
            P_filt[k] = (np.eye(S) - K @ H) @ Pp

        # --- backward RTS smoother ---
        x_smooth = np.zeros((N, S))
        x_smooth[-1] = x_filt[-1]

        for k in range(N - 2, -1, -1):
            C = P_filt[k] @ F.T @ np.linalg.inv(P_pred[k + 1])
            x_smooth[k] = x_filt[k] + C @ (x_smooth[k + 1] - x_pred[k + 1])

        result = x_smooth[:, :D]
        return self._squeeze_output(result)

    def _squeeze_output(self, y: np.ndarray) -> np.ndarray:
        """Squeeze trailing dimension when the original input was 1-D."""
        if self._scalar_input:
            return y.squeeze(-1)
        return y


# ======================================================================
# Helper functions (module-level)
# ======================================================================


def _design_iir_sos(
    cutoff_freq: float | list[float],
    sample_rate: float,
    order: int,
    iir_type: str,
    btype: str,
    rp: float | None,
    rs: float | None,
) -> np.ndarray:
    """Design an IIR filter and return SOS coefficients.

    Shared by offline, online, and chunk IIR filters.
    """
    nyq = 0.5 * sample_rate
    if isinstance(cutoff_freq, (list, tuple)):
        wn = [f / nyq for f in cutoff_freq]
    else:
        wn = float(cutoff_freq) / nyq

    if iir_type == "butterworth":
        sos = sp_signal.butter(order, wn, btype=btype, output="sos")
    elif iir_type == "chebyshev1":
        if rp is None:
            raise ValueError("'chebyshev1' requires 'rp' (passband ripple in dB).")
        sos = sp_signal.cheby1(order, rp, wn, btype=btype, output="sos")
    elif iir_type == "chebyshev2":
        if rs is None:
            raise ValueError("'chebyshev2' requires 'rs' (stopband attenuation in dB).")
        sos = sp_signal.cheby2(order, rs, wn, btype=btype, output="sos")
    elif iir_type == "elliptic":
        if rp is None or rs is None:
            raise ValueError(
                "'elliptic' requires both 'rp' (passband ripple) and "
                "'rs' (stopband attenuation) in dB."
            )
        sos = sp_signal.ellip(order, rp, rs, wn, btype=btype, output="sos")
    elif iir_type == "bessel":
        sos = sp_signal.bessel(order, wn, btype=btype, output="sos", norm="phase")
    else:
        raise ValueError(
            f"Unknown iir_type '{iir_type}'. "
            f"Choose from: {', '.join(repr(t) for t in OfflineFilter._IIR_TYPES)}."
        )
    return sos


def _build_kalman_model(
    D: int,
    dt: float,
    state_model: str,
    process_noise: float,
    measurement_noise: float,
) -> tuple:
    """Build Kalman filter matrices for a built-in state model.

    Returns (F, H, Q, R, S) where S is the state dimension.
    """
    if state_model == "position":
        S = D
        F = np.eye(S)
        H = np.eye(D, S)
        Q = np.eye(S) * process_noise
        R = np.eye(D) * measurement_noise
    elif state_model == "position_velocity":
        S = 2 * D
        F = np.eye(S)
        F[:D, D:] = np.eye(D) * dt
        H = np.zeros((D, S))
        H[:D, :D] = np.eye(D)
        # Process noise: discrete white-noise acceleration model
        q = process_noise
        Q = np.zeros((S, S))
        Q[:D, :D] = np.eye(D) * (dt**4 / 4) * q
        Q[:D, D:] = np.eye(D) * (dt**3 / 2) * q
        Q[D:, :D] = np.eye(D) * (dt**3 / 2) * q
        Q[D:, D:] = np.eye(D) * (dt**2) * q
        R = np.eye(D) * measurement_noise
    else:
        raise ValueError(
            f"Unknown state_model '{state_model}'. "
            "Choose from: 'position', 'position_velocity'."
        )
    return F, H, Q, R, S
