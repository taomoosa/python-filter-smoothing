#!/usr/bin/env python3
"""Chunk filter example -- streaming chunks with per-timestep output.

Similar to the online example, this script loops through simulation time
and logs filtered output at every timestep.  New chunks of noisy data
arrive at regular intervals (whenever the simulation clock reaches the
chunk's start time), and the filters are re-evaluated after each chunk
is added.  Raw chunks -- including overlapping timestamps -- are plotted
alongside the filtered output so you can see how the filter evolves as
more data becomes available.

Usage::

    # Spawn rerun viewer
    python examples/chunk_example.py

    # Save to .rrd file instead
    python examples/chunk_example.py --save chunk.rrd
"""

from __future__ import annotations

import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _data import DIM_NAMES, generate_chunked_xyz

from python_filter_smoothing import ChunkFilter
from python_filter_smoothing.visualize import (
    configure_series_style,
    init_recording,
    log_scalar,
    log_time_series,
    send_dim_blueprint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=str, default=None, help="Save .rrd file path")
    args = parser.parse_args()

    # --- data ---
    t_full, x_clean, _, chunks = generate_chunked_xyz(
        duration=10.0,
        sample_rate=100.0,
        noise_std=0.15,
        n_chunks=10,
        overlap_samples=8,
        discontinuity_std=0.12,
    )

    # --- rerun init ---
    init_recording(
        app_id="chunk-filter-example",
        spawn=(args.save is None),
        save_path=args.save,
    )
    send_dim_blueprint(DIM_NAMES)

    # --- styles ---
    for dim in DIM_NAMES:
        configure_series_style(f"{dim}/clean", color=(0, 200, 0), name="clean", width=1.5)
        configure_series_style(
            f"{dim}/linear_latest", color=(255, 80, 80), name="linear_latest", width=1.5,
        )
        configure_series_style(
            f"{dim}/linear_mean", color=(80, 80, 255), name="linear_mean", width=1.5,
        )
        configure_series_style(
            f"{dim}/spline_latest", color=(255, 160, 0), name="spline_latest", width=1.5,
        )
        configure_series_style(
            f"{dim}/polynomial_latest", color=(200, 0, 200), name="poly_latest", width=1.5,
        )
        configure_series_style(
            f"{dim}/savgol_blend", color=(0, 200, 200), name="savgol_blend", width=1.5,
        )
        configure_series_style(
            f"{dim}/gaussian_blend", color=(200, 200, 0), name="gaussian_blend", width=1.5,
        )
        configure_series_style(
            f"{dim}/lowpass_blend", color=(0, 120, 200), name="lowpass_blend", width=1.5,
        )
        configure_series_style(
            f"{dim}/median_blend", color=(120, 60, 0), name="median_blend", width=1.5,
        )
        configure_series_style(
            f"{dim}/savgol_cosine", color=(160, 0, 80), name="savgol_cosine", width=1.5,
        )

    # --- filters ---
    filters: dict[str, ChunkFilter] = {
        "linear_latest": ChunkFilter(method="linear", overlap_strategy="latest"),
        "linear_mean": ChunkFilter(method="linear", overlap_strategy="mean"),
        "spline_latest": ChunkFilter(method="spline", overlap_strategy="latest"),
        "polynomial_latest": ChunkFilter(
            method="polynomial", overlap_strategy="latest", degree=8,
        ),
        "savgol_blend": ChunkFilter(
            method="savgol", overlap_strategy="blend", window_length=21, polyorder=3,
        ),
        "gaussian_blend": ChunkFilter(
            method="gaussian", overlap_strategy="blend", sigma=3.0,
        ),
        "lowpass_blend": ChunkFilter(
            method="lowpass", overlap_strategy="blend", cutoff_freq=4.0, sample_rate=100.0,
        ),
        "median_blend": ChunkFilter(
            method="median", overlap_strategy="blend", kernel_size=11,
        ),
        "savgol_cosine": ChunkFilter(
            method="savgol", overlap_strategy="cosine_blend",
            window_length=21, polyorder=3,
        ),
    }

    # Chunk arrival schedule: chunk i arrives when sim-time >= its first timestamp
    chunk_start_times = [float(t_c[0]) for t_c, _ in chunks]
    next_chunk_idx = 0
    has_data = False

    # Cache of filtered output (recomputed only when a new chunk arrives)
    filtered_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # --- time-stepping loop ---
    for step in range(len(t_full)):
        ti_f = float(t_full[step])

        # Log clean reference at every timestep
        log_scalar("clean", ti_f, x_clean[step], dim_names=DIM_NAMES, dim_first=True)

        # Check whether new chunk(s) should arrive at this moment
        chunks_added = False
        while (
            next_chunk_idx < len(chunks)
            and ti_f >= chunk_start_times[next_chunk_idx]
        ):
            t_c, x_c = chunks[next_chunk_idx]
            # Visualise the raw chunk (overlap regions included)
            log_time_series(
                f"raw_chunks/chunk_{next_chunk_idx:02d}",
                t_c,
                x_c,
                dim_names=DIM_NAMES,
                dim_first=True,
            )
            for filt in filters.values():
                filt.add_chunk(x_c, t=t_c)
            next_chunk_idx += 1
            chunks_added = True
            has_data = True

        # Recompute filtered output whenever new data was added
        if chunks_added:
            for name, filt in filters.items():
                t_out, x_out = filt.get_filtered(t_query=t_full)
                filtered_cache[name] = (t_out, x_out)

        # Log current timestep's filtered value from cache
        if has_data:
            for name in filters:
                if name not in filtered_cache:
                    continue
                t_out, x_out = filtered_cache[name]
                idx = np.searchsorted(t_out, ti_f)
                if idx < len(t_out) and abs(float(t_out[idx]) - ti_f) < 1e-9:
                    log_scalar(
                        name, ti_f, x_out[idx], dim_names=DIM_NAMES, dim_first=True,
                    )

    print("Done. Data logged to rerun.")


if __name__ == "__main__":
    main()
