#!/usr/bin/env python3
"""Async filter example -- multi-threaded chunk feeding and output.

Two threads run concurrently:

* **Producer** -- feeds chunks of noisy data into the shared AsyncFilter
  instances at regular intervals, simulating sensor data arriving in bursts.
* **Consumer** -- queries each filter at a fixed output rate (different from
  the input sample rate) and logs the smoothed values to rerun.

Because AsyncFilter is thread-safe, the producer can call ``update()``
while the consumer calls ``get_output()`` without explicit coordination.
Raw chunks (including overlapping timestamps) are visualised so you can
see how each chunk is absorbed by the filters.

Usage::

    # Spawn rerun viewer
    python examples/async_example.py

    # Save to .rrd file instead
    python examples/async_example.py --save async.rrd
"""

from __future__ import annotations

import argparse
import sys
import os
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _data import DIM_NAMES, generate_chunked_xyz

from python_filter_smoothing import AsyncFilter
from python_filter_smoothing.visualize import (
    configure_series_style,
    init_recording,
    log_scalar,
    log_time_series,
    send_dim_blueprint,
)

INPUT_SAMPLE_RATE = 30.0  # Hz -- rate of samples within each chunk
OUTPUT_RATE = 200.0  # Hz -- rate at which the consumer queries filters


# ---------------------------------------------------------------------------
# Producer thread
# ---------------------------------------------------------------------------
def _producer(
    filters: dict[str, AsyncFilter],
    chunks: list[tuple[np.ndarray, np.ndarray]],
    sim_time: list[float],
    sim_lock: threading.Lock,
) -> None:
    """Feed chunks into filters when simulation time reaches chunk start."""
    for i, (t_c, x_c) in enumerate(chunks):
        chunk_start = float(t_c[0])

        # Block until the consumer's simulation clock reaches this chunk
        while True:
            with sim_lock:
                if sim_time[0] >= chunk_start:
                    break
            # Yield to the consumer thread
            threading.Event().wait(timeout=0.0001)

        # Log the raw chunk for visualisation (overlap regions included)
        log_time_series(
            f"raw_chunks/chunk_{i:02d}",
            t_c,
            x_c,
            dim_names=DIM_NAMES,
            dim_first=True,
        )

        # Feed the chunk into each filter
        for filt in filters.values():
            filt.update_chunk(t_c, x_c)


# ---------------------------------------------------------------------------
# Consumer thread
# ---------------------------------------------------------------------------
def _consumer(
    filters: dict[str, AsyncFilter],
    t_output: np.ndarray,
    x_clean_output: np.ndarray,
    sim_time: list[float],
    sim_lock: threading.Lock,
) -> None:
    """Query filters at *OUTPUT_RATE* and log results per-timestep."""
    for ti_f, xi_clean in zip(t_output.astype(float), x_clean_output):
        # Advance the shared simulation clock
        with sim_lock:
            sim_time[0] = ti_f

        # Log clean reference
        log_scalar("clean", ti_f, xi_clean, dim_names=DIM_NAMES, dim_first=True)

        # Update current time for RAIL filters (used as blend start)
        for name, filt in filters.items():
            if hasattr(filt, "set_current_time"):
                filt.set_current_time(ti_f)

        # Query each filter at current output time
        for name, filt in filters.items():
            val = filt.get_output(t=ti_f)
            if val is not None:
                log_scalar(name, ti_f, val, dim_names=DIM_NAMES, dim_first=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=str, default=None, help="Save .rrd file path")
    args = parser.parse_args()

    # --- data (input chunks at INPUT_SAMPLE_RATE) ---
    t_full, x_clean, _, chunks = generate_chunked_xyz(
        duration=10.0,
        sample_rate=INPUT_SAMPLE_RATE,
        noise_std=0.02,
        n_chunks=10,
        overlap_samples=8,
        discontinuity_std=0.12,
    )

    # --- output timestamps at OUTPUT_RATE (different from input rate) ---
    t_output = np.arange(float(t_full[0]), float(t_full[-1]), 1.0 / OUTPUT_RATE)
    # Interpolate clean signal onto output grid for reference plotting
    x_clean_output = np.column_stack(
        [np.interp(t_output, t_full, x_clean[:, d]) for d in range(x_clean.shape[1])]
    )

    # --- rerun init ---
    init_recording(
        app_id="async-filter-example",
        spawn=(args.save is None),
        save_path=args.save,
    )
    send_dim_blueprint(DIM_NAMES)

    # --- styles ---
    for dim in DIM_NAMES:
        configure_series_style(f"{dim}/clean", color=(0, 200, 0), name="clean", width=1.5)
        configure_series_style(f"{dim}/ema", color=(255, 80, 80), name="ema", width=1.5)
        configure_series_style(f"{dim}/linear", color=(80, 80, 255), name="linear", width=1.5)
        configure_series_style(f"{dim}/spline", color=(255, 160, 0), name="spline", width=1.5)
        configure_series_style(f"{dim}/one_euro", color=(200, 0, 200), name="one_euro", width=1.5)
        configure_series_style(f"{dim}/moving_avg", color=(0, 200, 200), name="moving_avg", width=1.5)
        configure_series_style(f"{dim}/act", color=(120, 200, 60), name="act", width=1.5)
        configure_series_style(f"{dim}/rail", color=(200, 60, 120), name="rail", width=1.5)

    # --- shared filters (AsyncFilter is thread-safe) ---
    chunk_len = max(len(tc) for tc, _ in chunks)
    filters: dict[str, AsyncFilter] = {
        "ema": AsyncFilter(method="ema", buffer_size=chunk_len, alpha=0.1),
        "linear": AsyncFilter(method="linear", buffer_size=chunk_len),
        "spline": AsyncFilter(method="spline", buffer_size=chunk_len),
        "one_euro": AsyncFilter(
            method="one_euro", buffer_size=chunk_len,
            min_cutoff=1.0, beta=0.007, d_cutoff=1.0,
        ),
        "moving_avg": AsyncFilter(
            method="moving_average", buffer_size=chunk_len, window=15,
        ),
        "act": AsyncFilter(method="act", k=0.5, max_chunks=5),
        "rail": AsyncFilter(
            method="rail", poly_degree=3, dual_quintic=False, auto_align=True,
        ),
    }

    # --- shared simulation clock ---
    sim_time: list[float] = [float(t_full[0]) - 1.0]  # start before first sample
    sim_lock = threading.Lock()

    # --- launch threads ---
    prod = threading.Thread(
        target=_producer, args=(filters, chunks, sim_time, sim_lock), daemon=True,
    )
    cons = threading.Thread(
        target=_consumer,
        args=(filters, t_output, x_clean_output, sim_time, sim_lock),
    )

    prod.start()
    cons.start()

    cons.join()
    prod.join()

    print("Done. Data logged to rerun.")


if __name__ == "__main__":
    main()
