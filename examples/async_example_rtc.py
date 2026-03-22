#!/usr/bin/env python3
"""Real-Time Chunking (RTC) async example — simulated VLA inference.

Demonstrates :class:`AsyncFilterRTC` in a two-thread control loop
mimicking real VLA deployment:

* **Inference thread** — generates noisy action chunks with variable
  latency (100–400 ms).  Uses the frozen prefix from the RTC filter to
  condition each new chunk on the actions currently being executed.
* **Control thread** — runs at 50 Hz, queries the RTC filter for the
  action to send to the robot.

Compares multiple filtering strategies:

* ``rtc_blend``  — RTC with soft-mask blending at chunk boundaries.
* ``rtc_none``   — RTC with no blending (raw chunk switching).
* ``rtc_rail``   — RTC Hermite inpainting → RAIL polynomial smoothing
  + quintic boundary blending (composite pipeline).
* ``rail``       — RAIL polynomial trajectory fitting + quintic fusion.
* ``act``        — ACT temporal-ensembling (weighted chunk averaging).
* ``rtc+1euro``  — RTC soft_mask output post-filtered with 1€ filter
  for additional intra-chunk noise smoothing.

The ground-truth trajectory is a smooth pick-and-place motion (XYZ).

Usage::

    python examples/async_example_rtc.py                  # spawn rerun viewer
    python examples/async_example_rtc.py --save rtc.rrd   # save to file
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _data import DIM_NAMES

from python_filter_smoothing import AsyncFilter, OnlineFilter

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
DURATION = 6.0
CONTROL_HZ = 50.0
CHUNK_HZ = 10.0
CHUNK_HORIZON = 16
INFERENCE_LATENCY_RANGE = (0.10, 0.40)
NOISE_BASE = 0.005
NOISE_GROWTH = 0.04
DISCONTINUITY_STD = 0.02
SEED = 42


# ---------------------------------------------------------------------------
# Ground-truth trajectory (pick-and-place)
# ---------------------------------------------------------------------------
def _ground_truth(t: np.ndarray) -> np.ndarray:
    T = t[-1] - t[0]
    s = (t - t[0]) / T
    x = 0.3 * np.sin(np.pi * s) ** 2
    y = 0.15 * np.tanh(4.0 * (s - 0.5))
    z = (
        0.10 * np.exp(-((s - 0.15) ** 2) / 0.005)
        - 0.25 * np.exp(-((s - 0.15) ** 2) / 0.002)
        + 0.30 * np.exp(-((s - 0.40) ** 2) / 0.01)
        + 0.25 * np.exp(-((s - 0.65) ** 2) / 0.02)
        - 0.10 * np.exp(-((s - 0.85) ** 2) / 0.005)
    ) + 0.20
    return np.column_stack([x, y, z])


# ---------------------------------------------------------------------------
# Pseudo VLA model
# ---------------------------------------------------------------------------
class _PseudoVLA:
    def __init__(self, t_full, x_clean, seed):
        self._t = t_full
        self._x = x_clean
        self._rng = np.random.default_rng(seed)
        self._offset = np.zeros(3)

    def predict(self, t_obs, prefix_x=None):
        dt = 1.0 / CHUNK_HZ
        t_chunk = t_obs + np.arange(CHUNK_HORIZON) * dt
        t_chunk = t_chunk[t_chunk <= self._t[-1]]
        if len(t_chunk) == 0:
            t_chunk = np.array([min(t_obs, self._t[-1])])
        n = len(t_chunk)
        x_chunk = np.column_stack(
            [np.interp(t_chunk, self._t, self._x[:, d]) for d in range(3)]
        )
        hfrac = np.linspace(0, 1, n)
        noise = self._rng.normal(size=(n, 3)) * (
            NOISE_BASE + NOISE_GROWTH * hfrac
        )[:, None]
        x_chunk += noise + self._offset
        self._offset += self._rng.normal(scale=DISCONTINUITY_STD, size=3)

        # If a prefix is given, blend the first few actions to match it
        if prefix_x is not None:
            d = min(len(prefix_x), n)
            x_chunk[:d] = prefix_x[:d]
        return t_chunk, x_chunk


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------
def _inference_loop(
    vla, filters, sim_time, sim_lock, t_end, rng,
    use_prefix_key=None, chunk_log=None,
):
    chunk_duration = CHUNK_HORIZON / CHUNK_HZ
    next_query = 0.0

    while True:
        while True:
            with sim_lock:
                now = sim_time[0]
            if now >= next_query or now >= t_end:
                break
            threading.Event().wait(timeout=0.0001)
        with sim_lock:
            now = sim_time[0]
        if now >= t_end:
            break

        obs_time = now

        # Get prefix from the RTC filter (if applicable)
        prefix_x = None
        if use_prefix_key and use_prefix_key in filters:
            rtc_filt = filters[use_prefix_key]
            rtc_filt.start_inference()
            prefix_info = rtc_filt.get_prefix()
            if prefix_info is not None:
                prefix_x = prefix_info[1]

        # Simulate inference latency
        latency = rng.uniform(*INFERENCE_LATENCY_RANGE)
        arrival = obs_time + latency
        while True:
            with sim_lock:
                curr = sim_time[0]
            if curr >= arrival or curr >= t_end:
                break
            threading.Event().wait(timeout=0.0001)

        t_chunk, x_chunk = vla.predict(obs_time, prefix_x=prefix_x)

        if chunk_log is not None:
            chunk_log.append((t_chunk.copy(), x_chunk.copy()))

        for name, filt in filters.items():
            if hasattr(filt, 'set_current_time'):
                with sim_lock:
                    filt.set_current_time(sim_time[0])
            filt.update_chunk(t_chunk, x_chunk)

        next_query = obs_time + chunk_duration * 0.5


def _control_loop(filters, sim_time, sim_lock, t_end, results,
                  post_filters=None):
    dt = 1.0 / CONTROL_HZ
    while True:
        with sim_lock:
            now = sim_time[0]
        if now >= t_end:
            break
        for name, filt in filters.items():
            if hasattr(filt, 'set_current_time'):
                filt.set_current_time(now)
            out = filt.get_output(now)
            if out is not None:
                results[name].append((now, out.copy()))

                # Apply post-filters to rtc_blend output
                if name == "rtc_blend" and post_filters is not None:
                    smoothed = np.array([
                        post_filters[d].update(now, out[d]).item()
                        for d in range(len(out))
                    ])
                    results["rtc+1euro"].append((now, smoothed.copy()))

        with sim_lock:
            sim_time[0] += dt
        threading.Event().wait(timeout=0.0001)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=str, default=None,
                        help="Save .rrd to file instead of spawning viewer.")
    args = parser.parse_args()

    # Ground truth
    t_full = np.linspace(0, DURATION, int(DURATION * 200))
    x_clean = _ground_truth(t_full)

    # Filters to compare
    filters = {
        "rtc_blend": AsyncFilter(
            method="rtc",
            prediction_horizon=CHUNK_HORIZON,
            dt=1.0 / CHUNK_HZ,
            blend_mode="soft_mask",
            initial_delay=2,
        ),
        "rtc_none": AsyncFilter(
            method="rtc",
            prediction_horizon=CHUNK_HORIZON,
            dt=1.0 / CHUNK_HZ,
            blend_mode="none",
            initial_delay=2,
        ),
        "rail": AsyncFilter(
            method="rail",
            poly_degree=3,
            blend_order="cubic",
        ),
        "rtc_rail": AsyncFilter(
            method="rtc_rail",
            prediction_horizon=CHUNK_HORIZON,
            dt=1.0 / CHUNK_HZ,
            inpainting="hermite",
            initial_delay=2,
            poly_degree=1,
            blend_order="cubic",
        ),
        "act": AsyncFilter(
            method="act",
            k=0.01,
            max_chunks=5,
        ),
    }

    # Post-processing: 1€ filter applied to rtc_blend output
    one_euro_filters = {
        d: OnlineFilter(
            method="one_euro", min_cutoff=1.5, beta=0.05, d_cutoff=1.0,
        )
        for d in range(3)
    }

    results: dict[str, list] = {n: [] for n in filters}
    results["rtc+1euro"] = []  # post-filtered series
    chunk_log: list[tuple[np.ndarray, np.ndarray]] = []
    sim_time = [0.0]
    sim_lock = threading.Lock()
    rng = np.random.default_rng(SEED)
    vla = _PseudoVLA(t_full, x_clean, seed=SEED + 1)

    inf_th = threading.Thread(
        target=_inference_loop,
        args=(vla, filters, sim_time, sim_lock, DURATION, rng, "rtc_blend"),
        kwargs={"chunk_log": chunk_log},
    )
    ctrl_th = threading.Thread(
        target=_control_loop,
        args=(filters, sim_time, sim_lock, DURATION, results),
        kwargs={"post_filters": one_euro_filters},
    )
    inf_th.start()
    ctrl_th.start()
    inf_th.join(timeout=30)
    ctrl_th.join(timeout=30)

    # --- Print summary statistics ---
    print(f"\n{'Method':<16} {'RMSE X':>10} {'RMSE Y':>10} {'RMSE Z':>10}")
    print("-" * 50)
    for name, data in results.items():
        if not data:
            print(f"{name:<16} {'(no data)':>10}")
            continue
        ts = np.array([d[0] for d in data])
        xs = np.stack([d[1] for d in data])
        gt = np.column_stack(
            [np.interp(ts, t_full, x_clean[:, d]) for d in range(3)]
        )
        rmse = np.sqrt(np.mean((xs - gt) ** 2, axis=0))
        print(f"{name:<16} {rmse[0]:10.5f} {rmse[1]:10.5f} {rmse[2]:10.5f}")

    # --- Rerun visualisation (optional) ---
    try:
        from python_filter_smoothing.visualize import (
            configure_series_style,
            init_recording,
            log_time_series,
            send_dim_blueprint,
        )
    except ImportError:
        print("\n(Install rerun-sdk for visualisation: pip install rerun-sdk)")
        return

    if args.save:
        init_recording("rtc_example", save_path=args.save)
    else:
        init_recording("rtc_example", spawn=True)

    send_dim_blueprint(DIM_NAMES)

    log_time_series("ground_truth", t_full, x_clean,
                     dim_names=DIM_NAMES, dim_first=True)

    colors = {
        "rtc_blend": [50, 140, 255],
        "rtc_none": [255, 160, 50],
        "rail": [50, 200, 100],
        "rtc_rail": [0, 200, 200],
        "act": [180, 50, 220],
        "rtc+1euro": [255, 80, 80],
    }

    # Style (paths are {dim}/{series} because dim_first=True)
    for dim in DIM_NAMES:
        configure_series_style(
            f"{dim}/ground_truth", color=[80, 80, 80], width=2.0,
            name="ground_truth",
        )
        for name, col in colors.items():
            configure_series_style(
                f"{dim}/{name}", color=col, name=name,
            )

    # Log raw input chunks (each as a separate series for visibility)
    chunk_colors = [
        [200, 200, 200],  # light grey base; alternates for contrast
        [170, 170, 170],
    ]
    for ci, (t_c, x_c) in enumerate(chunk_log):
        tag = f"chunks/chunk_{ci:02d}"
        log_time_series(tag, t_c, x_c, dim_names=DIM_NAMES, dim_first=True)
        for dim in DIM_NAMES:
            configure_series_style(
                f"{dim}/{tag}",
                color=chunk_colors[ci % 2],
                width=0.5,
            )

    for name, data in results.items():
        if not data:
            continue
        ts = np.array([d[0] for d in data])
        xs = np.stack([d[1] for d in data])
        log_time_series(name, ts, xs, dim_names=DIM_NAMES, dim_first=True)

    print("\nVisualisation sent to rerun." if not args.save
          else f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
