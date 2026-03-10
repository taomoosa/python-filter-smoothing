#!/usr/bin/env python3
"""Offline filter example – compare all offline methods on noisy XYZ data.

Usage::

    # Spawn rerun viewer
    python examples/offline_example.py

    # Save to .rrd file instead
    python examples/offline_example.py --save offline.rrd
"""

from __future__ import annotations

import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _data import DIM_NAMES, generate_xyz_signal

from python_filter_smoothing import OfflineFilter
from python_filter_smoothing.visualize import (
    configure_series_style,
    init_recording,
    log_time_series,
    send_dim_blueprint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=str, default=None, help="Save .rrd file path")
    args = parser.parse_args()

    # --- data ---
    t, x_clean, x_noisy = generate_xyz_signal(
        duration=10.0, sample_rate=100.0, noise_std=0.15,
    )
    sample_rate = 100.0

    # --- rerun init ---
    init_recording(
        app_id="offline-filter-example",
        spawn=(args.save is None),
        save_path=args.save,
    )
    send_dim_blueprint(DIM_NAMES)

    # --- style (paths are {dim}/{series} because dim_first=True) ---
    for dim in DIM_NAMES:
        configure_series_style(f"{dim}/clean", color=(0, 200, 0), name="clean", width=1.5)
        configure_series_style(f"{dim}/input", color=(160, 160, 160), name="noisy", width=0.5)
        configure_series_style(f"{dim}/lowpass", color=(255, 80, 80), name="lowpass", width=1.5)
        configure_series_style(f"{dim}/polynomial", color=(80, 80, 255), name="polynomial", width=1.5)
        configure_series_style(f"{dim}/spline", color=(255, 160, 0), name="spline", width=1.5)
        configure_series_style(f"{dim}/linear_interp", color=(0, 200, 200), name="linear", width=1.5)
        configure_series_style(f"{dim}/savgol", color=(200, 0, 200), name="savgol", width=1.5)
        configure_series_style(f"{dim}/gaussian", color=(200, 200, 0), name="gaussian", width=1.5)
        configure_series_style(f"{dim}/median", color=(0, 120, 200), name="median", width=1.5)
        configure_series_style(f"{dim}/moving_avg", color=(120, 60, 0), name="moving_avg", width=1.5)

    # --- log input ---
    log_time_series("clean", t, x_clean, dim_names=DIM_NAMES, dim_first=True)
    log_time_series("input", t, x_noisy, dim_names=DIM_NAMES, dim_first=True)

    # --- offline filter ---
    filt = OfflineFilter(t, x_noisy)

    # 1) Lowpass filter
    x_lowpass = filt.lowpass_filter(
        cutoff_freq=0.08 * (sample_rate / 2),
        sample_rate=sample_rate,
        order=4,
    )
    log_time_series("lowpass", t, x_lowpass, dim_names=DIM_NAMES, dim_first=True)

    # 2) Polynomial fit (degree 12 to capture the shape well)
    x_poly = filt.polynomial_fit(degree=12)
    log_time_series("polynomial", t, x_poly, dim_names=DIM_NAMES, dim_first=True)

    # 3) Spline interpolation (query at a denser grid)
    t_dense = np.linspace(t[0], t[-1], len(t) * 2)
    x_spline = filt.spline_interpolate(t_dense, kind="cubic")
    log_time_series("spline", t_dense, x_spline, dim_names=DIM_NAMES, dim_first=True)

    # 4) Linear interpolation (same dense grid)
    x_linear = filt.linear_interpolate(t_dense)
    log_time_series("linear_interp", t_dense, x_linear, dim_names=DIM_NAMES, dim_first=True)

    # 5) Savitzky-Golay filter
    x_savgol = filt.savgol_filter(window_length=21, polyorder=3)
    log_time_series("savgol", t, x_savgol, dim_names=DIM_NAMES, dim_first=True)

    # 6) Gaussian smoothing
    x_gauss = filt.gaussian_filter(sigma=3.0)
    log_time_series("gaussian", t, x_gauss, dim_names=DIM_NAMES, dim_first=True)

    # 7) Median filter
    x_median = filt.median_filter(kernel_size=11)
    log_time_series("median", t, x_median, dim_names=DIM_NAMES, dim_first=True)

    # 8) Moving average
    x_mavg = filt.moving_average(window_size=15)
    log_time_series("moving_avg", t, x_mavg, dim_names=DIM_NAMES, dim_first=True)

    print("Done. Data logged to rerun.")


if __name__ == "__main__":
    main()
