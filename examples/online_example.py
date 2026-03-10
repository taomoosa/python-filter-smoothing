#!/usr/bin/env python3
"""Online filter example – compare EMA, moving average, and lowpass on noisy XYZ data.

Data is fed sample-by-sample to each filter, simulating a real-time stream.

Usage::

    # Spawn rerun viewer
    python examples/online_example.py

    # Save to .rrd file instead
    python examples/online_example.py --save online.rrd
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from _data import DIM_NAMES, generate_xyz_signal

from python_filter_smoothing import OnlineFilter
from python_filter_smoothing.visualize import (
    configure_series_style,
    init_recording,
    log_scalar,
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

    # --- rerun init ---
    init_recording(
        app_id="online-filter-example",
        spawn=(args.save is None),
        save_path=args.save,
    )
    send_dim_blueprint(DIM_NAMES)

    # --- styles (paths are {dim}/{series} because dim_first=True) ---
    for dim in DIM_NAMES:
        configure_series_style(f"{dim}/clean", color=(0, 200, 0), name="clean", width=1.5)
        configure_series_style(f"{dim}/input", color=(160, 160, 160), name="noisy", width=0.5)
        configure_series_style(f"{dim}/ema", color=(255, 80, 80), name="ema", width=1.5)
        configure_series_style(f"{dim}/moving_avg", color=(80, 80, 255), name="moving_avg", width=1.5)
        configure_series_style(f"{dim}/lowpass", color=(255, 160, 0), name="lowpass", width=1.5)
        configure_series_style(f"{dim}/one_euro", color=(200, 0, 200), name="one_euro", width=1.5)

    # --- filters ---
    filt_ema = OnlineFilter(method="ema", alpha=0.15)
    filt_mavg = OnlineFilter(method="moving_average", window=20)
    filt_lp = OnlineFilter(method="lowpass", cutoff_freq=3.0, sample_rate=100.0, order=2)
    filt_1e = OnlineFilter(method="one_euro", min_cutoff=1.0, beta=0.007, d_cutoff=1.0)

    # --- streaming loop ---
    for i, (ti, xi_clean, xi_noisy) in enumerate(zip(t, x_clean, x_noisy)):
        ti_f = float(ti)
        log_scalar("clean", ti_f, xi_clean, dim_names=DIM_NAMES, dim_first=True)
        log_scalar("input", ti_f, xi_noisy, dim_names=DIM_NAMES, dim_first=True)

        y_ema = filt_ema.update(ti_f, xi_noisy)
        y_mavg = filt_mavg.update(ti_f, xi_noisy)
        y_lp = filt_lp.update(ti_f, xi_noisy)
        y_1e = filt_1e.update(ti_f, xi_noisy)

        log_scalar("ema", ti_f, y_ema, dim_names=DIM_NAMES, dim_first=True)
        log_scalar("moving_avg", ti_f, y_mavg, dim_names=DIM_NAMES, dim_first=True)
        log_scalar("lowpass", ti_f, y_lp, dim_names=DIM_NAMES, dim_first=True)
        log_scalar("one_euro", ti_f, y_1e, dim_names=DIM_NAMES, dim_first=True)

    print("Done. Data logged to rerun.")


if __name__ == "__main__":
    main()
