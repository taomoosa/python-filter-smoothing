"""Rerun-based visualization utilities for time series data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import rerun as rr


def _ensure_rerun() -> "rr":
    """Import rerun, raising a helpful error if not installed."""
    try:
        import rerun as _rr
    except ImportError:
        raise ImportError(
            "rerun-sdk is required for visualization. "
            "Install it with: uv pip install -e '.[viz]'"
        ) from None
    return _rr


def init_recording(
    app_id: str = "python-filter-smoothing",
    *,
    spawn: bool = False,
    save_path: str | None = None,
) -> None:
    """Initialize a rerun recording.

    Parameters
    ----------
    app_id : str
        Application identifier shown in the rerun viewer.
    spawn : bool
        If True, spawn the rerun viewer automatically.
    save_path : str or None
        If set, save the recording to an .rrd file at this path.
    """
    rr = _ensure_rerun()
    rr.init(app_id, spawn=spawn)
    if save_path is not None:
        rr.save(save_path)


def log_time_series(
    entity_path: str,
    t: np.ndarray,
    x: np.ndarray,
    *,
    dim_names: list[str] | None = None,
    dim_first: bool = False,
    timeline: str = "time",
    recording: "rr.RecordingStream | None" = None,
) -> None:
    """Log a batch of time series data to rerun using columnar API.

    Parameters
    ----------
    entity_path : str
        Rerun entity path (e.g. "input/raw", "output/filtered").
    t : array-like, shape (N,)
        Timestamps.
    x : array-like, shape (N,) or (N, D)
        Values. If 2-D, each column is logged as a separate sub-entity.
    dim_names : list of str or None
        Optional names for each dimension (e.g. ``["x", "y", "z"]``).
        Defaults to ``dim_0``, ``dim_1``, … when *None*.
    dim_first : bool
        When *True* the entity hierarchy is ``{dim}/{entity_path}`` so that
        all series for the same dimension appear in one panel.  When *False*
        (default), the hierarchy is ``{entity_path}/{dim}``.
    timeline : str
        Name of the rerun timeline.
    recording : RecordingStream or None
        Optional explicit recording stream.
    """
    rr = _ensure_rerun()

    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    time_col = rr.TimeColumn(timeline, timestamp=t)

    if x.ndim == 1:
        rr.send_columns(
            entity_path,
            indexes=[time_col],
            columns=rr.Scalars.columns(scalars=x),
            recording=recording,
        )
    elif x.ndim == 2:
        names = dim_names or [f"dim_{d}" for d in range(x.shape[1])]
        for d, name in enumerate(names):
            path = f"{name}/{entity_path}" if dim_first else f"{entity_path}/{name}"
            rr.send_columns(
                path,
                indexes=[time_col],
                columns=rr.Scalars.columns(scalars=x[:, d]),
                recording=recording,
            )
    else:
        raise ValueError(f"x must be 1-D or 2-D, got {x.ndim}-D")


def log_scalar(
    entity_path: str,
    t: float,
    x: float | np.ndarray,
    *,
    dim_names: list[str] | None = None,
    dim_first: bool = False,
    timeline: str = "time",
    recording: "rr.RecordingStream | None" = None,
) -> None:
    """Log a single time-stamped scalar (or vector) to rerun.

    Useful for online / streaming filters that process one sample at a time.

    Parameters
    ----------
    entity_path : str
        Rerun entity path.
    t : float
        Timestamp.
    x : float or array-like, shape (D,)
        Value(s). If array, each element is logged as a separate sub-entity.
    dim_names : list of str or None
        Optional names for each dimension (e.g. ``["x", "y", "z"]``).
        Defaults to ``dim_0``, ``dim_1``, … when *None*.
    dim_first : bool
        When *True* the entity hierarchy is ``{dim}/{entity_path}``.
    timeline : str
        Name of the rerun timeline.
    recording : RecordingStream or None
        Optional explicit recording stream.
    """
    rr = _ensure_rerun()

    rr.set_time(timeline, timestamp=float(t), recording=recording)

    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    if x.ndim == 1 and x.shape[0] == 1:
        rr.log(entity_path, rr.Scalars(x), recording=recording)
    else:
        names = dim_names or [f"dim_{d}" for d in range(x.shape[0])]
        for d, name in enumerate(names):
            path = f"{name}/{entity_path}" if dim_first else f"{entity_path}/{name}"
            rr.log(
                path,
                rr.Scalars(x[d : d + 1]),
                recording=recording,
            )


def send_dim_blueprint(
    dim_names: list[str],
    *,
    recording: "rr.RecordingStream | None" = None,
) -> None:
    """Send a rerun blueprint that creates one TimeSeriesView per dimension.

    This ensures the default viewer layout groups all series (clean, filtered,
    raw chunks, …) for the same dimension into a single panel, rather than
    the auto-layout which may group by series.

    Parameters
    ----------
    dim_names : list of str
        Dimension names (e.g. ``["x", "y", "z"]``).  Each name must match
        the top-level entity path used when ``dim_first=True``.
    recording : RecordingStream or None
        Optional explicit recording stream.
    """
    rr = _ensure_rerun()
    import rerun.blueprint as rrb

    views = [
        rrb.TimeSeriesView(origin=f"/{dim}", name=dim)
        for dim in dim_names
    ]
    blueprint = rrb.Blueprint(rrb.Vertical(*views), auto_views=False)
    rr.send_blueprint(blueprint, recording=recording)


def configure_series_style(
    entity_path: str,
    *,
    color: tuple[int, int, int] | None = None,
    name: str | None = None,
    width: float | None = None,
    recording: "rr.RecordingStream | None" = None,
) -> None:
    """Set visual style for a time series entity (color, label, width).

    Parameters
    ----------
    entity_path : str
        Rerun entity path to style.
    color : (R, G, B) tuple or None
        Line color in 0-255 range.
    name : str or None
        Display name for the series.
    width : float or None
        Line width.
    recording : RecordingStream or None
        Optional explicit recording stream.
    """
    rr = _ensure_rerun()

    kwargs: dict = {}
    if color is not None:
        kwargs["colors"] = [color]
    if name is not None:
        kwargs["names"] = name
    if width is not None:
        kwargs["widths"] = width

    if kwargs:
        rr.log(
            entity_path,
            rr.SeriesLines(**kwargs),
            static=True,
            recording=recording,
        )
