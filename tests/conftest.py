"""Shared pytest configuration.

Adds a ``--visualize`` CLI flag that, when set, initializes a rerun recording
and exposes a ``rerun_recording`` fixture for test functions.

Usage::

    # Normal test run (no visualisation)
    pytest

    # Manual test with rerun viewer (spawns viewer window)
    pytest --visualize

    # Save recording to file instead of spawning viewer
    pytest --visualize --rrd-path=recording.rrd
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--visualize",
        action="store_true",
        default=False,
        help="Enable rerun.io visualization during tests.",
    )
    parser.addoption(
        "--rrd-path",
        type=str,
        default=None,
        help="Path to save .rrd recording file (implies --visualize).",
    )


@pytest.fixture(scope="session")
def visualize_enabled(request: pytest.FixtureRequest) -> bool:
    """Whether visualization is enabled for this test session."""
    return bool(
        request.config.getoption("--visualize")
        or request.config.getoption("--rrd-path")
    )


@pytest.fixture(scope="session", autouse=True)
def _rerun_session(request: pytest.FixtureRequest, visualize_enabled: bool):
    """Session-scoped fixture that initializes rerun once when --visualize is used."""
    if not visualize_enabled:
        return

    from python_filter_smoothing.visualize import init_recording

    rrd_path = request.config.getoption("--rrd-path")
    init_recording(
        app_id="python-filter-smoothing-tests",
        spawn=(rrd_path is None),
        save_path=rrd_path,
    )


@pytest.fixture()
def rerun_log(visualize_enabled: bool):
    """Fixture that returns a helper for conditional rerun logging.

    Returns a callable ``log(entity_path, t, x)`` that only logs when
    ``--visualize`` is active.  Tests can use it without caring whether
    visualization is on or off.

    Example::

        def test_something(rerun_log):
            ...
            rerun_log("online/ema/input", t_array, x_input)
            rerun_log("online/ema/output", t_array, x_output)
    """
    if not visualize_enabled:
        # Return a no-op callable so tests don't need ``if`` guards.
        def _noop(*_args, **_kwargs):
            pass

        return _noop

    from python_filter_smoothing.visualize import log_time_series

    return log_time_series


@pytest.fixture()
def rerun_log_scalar(visualize_enabled: bool):
    """Fixture that returns a helper for logging individual scalars.

    Useful for online / streaming filter tests.

    Example::

        def test_online(rerun_log_scalar):
            for ti, xi in zip(t, x):
                y = filt.update(ti, xi)
                rerun_log_scalar("online/ema/input", ti, xi)
                rerun_log_scalar("online/ema/output", ti, y)
    """
    if not visualize_enabled:

        def _noop(*_args, **_kwargs):
            pass

        return _noop

    from python_filter_smoothing.visualize import log_scalar

    return log_scalar
