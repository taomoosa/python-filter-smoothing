#!/usr/bin/env python3
"""VLA async inference example -- simulated Vision-Language-Action model.

Simulates the asynchronous control loop used when deploying VLA models
(e.g. π₀, SmolVLA, ACT, Diffusion Policy) on a real robot:

* **Inference thread** — a pseudo-VLA model observes the current state and
  produces an *action chunk* (a short trajectory of future actions) after a
  variable inference latency.  Chunks overlap: the model is queried again
  before the current chunk is fully consumed.
* **Control thread** — runs at a fixed high-frequency control rate and
  queries each filter to obtain a smooth, continuous action to send to the
  robot.

The ground-truth trajectory mimics a pick-and-place motion:

* X: forward reach → retract  (smooth bell curve)
* Y: lateral S-curve
* Z: descend → grasp → lift → transport → place

Noise characteristics emulate real VLA predictions:

* Gaussian noise whose standard deviation grows toward the end of each
  chunk (far-future actions are less certain).
* Small inter-chunk discontinuities (each chunk is predicted independently).
* Variable inference latency (100–400 ms).

Usage::

    python examples/vla_example.py            # spawn rerun viewer
    python examples/vla_example.py --save vla.rrd   # save to file
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

from python_filter_smoothing import AsyncFilter
from python_filter_smoothing.visualize import (
    configure_series_style,
    init_recording,
    log_scalar,
    log_time_series,
    send_dim_blueprint,
)

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
DURATION = 8.0  # seconds of trajectory
CONTROL_HZ = 50.0  # robot control frequency (output)
CHUNK_HZ = 10.0  # action prediction frequency within each chunk
CHUNK_HORIZON = 16  # number of predicted timesteps per chunk
INFERENCE_TRIGGER = 0.5  # re-query when chunk queue ≤ 50 %
INFERENCE_LATENCY_RANGE = (0.10, 0.40)  # seconds (min, max)
NOISE_BASE = 0.005  # base noise std at start of chunk
NOISE_GROWTH = 0.04  # additional noise std at end of chunk
DISCONTINUITY_STD = 0.02  # inter-chunk offset jitter
SEED = 12345


# ---------------------------------------------------------------------------
# Ground-truth trajectory (pick-and-place style)
# ---------------------------------------------------------------------------
def _ground_truth(t: np.ndarray) -> np.ndarray:
    """Generate a smooth pick-and-place XYZ trajectory.

    Returns shape (N, 3).
    """
    T = t[-1] - t[0]
    s = (t - t[0]) / T  # normalised [0, 1]

    # X: forward reach and retract (bell curve)
    x = 0.3 * np.sin(np.pi * s) ** 2

    # Y: lateral S-curve (smooth sigmoid-like)
    y = 0.15 * np.tanh(4.0 * (s - 0.5))

    # Z: descend → grasp → lift → transport → place
    #    Composed of smooth bumps
    z = (
        0.10 * np.exp(-((s - 0.15) ** 2) / 0.005)  # lower to grasp
        - 0.25 * np.exp(-((s - 0.15) ** 2) / 0.002)  # dip
        + 0.30 * np.exp(-((s - 0.40) ** 2) / 0.01)  # lift
        + 0.25 * np.exp(-((s - 0.65) ** 2) / 0.02)  # transport
        - 0.10 * np.exp(-((s - 0.85) ** 2) / 0.005)  # place
    )
    z += 0.20  # baseline height

    return np.column_stack([x, y, z])


# ---------------------------------------------------------------------------
# Simulated VLA inference (produces action chunks)
# ---------------------------------------------------------------------------
class _PseudoVLA:
    """Generates noisy action chunks mimicking a VLA model."""

    def __init__(self, t_full: np.ndarray, x_clean: np.ndarray, seed: int):
        self._t_full = t_full
        self._x_clean = x_clean
        self._rng = np.random.default_rng(seed)
        self._offset = np.zeros(3)  # accumulated inter-chunk drift

    def predict(self, t_obs: float) -> tuple[np.ndarray, np.ndarray]:
        """Predict an action chunk starting from *t_obs*.

        Returns (t_chunk, x_chunk) with shape (H, ) and (H, 3).
        """
        dt = 1.0 / CHUNK_HZ
        t_chunk = t_obs + np.arange(CHUNK_HORIZON) * dt
        # Trim to stay within trajectory duration
        t_chunk = t_chunk[t_chunk <= self._t_full[-1]]
        if len(t_chunk) == 0:
            t_chunk = np.array([min(t_obs, self._t_full[-1])])

        # Interpolate ground truth at chunk timestamps
        n_pts = len(t_chunk)
        x_chunk = np.column_stack(
            [
                np.interp(t_chunk, self._t_full, self._x_clean[:, d])
                for d in range(3)
            ]
        )

        # Add noise that grows along the chunk horizon
        horizon_frac = np.linspace(0, 1, n_pts)
        noise_std = NOISE_BASE + NOISE_GROWTH * horizon_frac
        noise = self._rng.normal(size=(n_pts, 3)) * noise_std[:, None]
        x_chunk += noise

        # Add inter-chunk discontinuity
        x_chunk += self._offset
        self._offset += self._rng.normal(scale=DISCONTINUITY_STD, size=3)

        return t_chunk, x_chunk


# ---------------------------------------------------------------------------
# Inference thread (producer)
# ---------------------------------------------------------------------------
def _inference_thread(
    vla: _PseudoVLA,
    filters: dict[str, AsyncFilter],
    sim_time: list[float],
    sim_lock: threading.Lock,
    t_end: float,
    rng: np.random.Generator,
    all_chunks: list,
) -> None:
    """Periodically run VLA inference and feed chunks into filters."""
    chunk_duration = CHUNK_HORIZON / CHUNK_HZ
    next_query_time = 0.0  # simulation time at which to trigger next query
    chunk_idx = 0

    while True:
        # Wait until simulation clock reaches next_query_time
        while True:
            with sim_lock:
                now = sim_time[0]
            if now >= next_query_time or now >= t_end:
                break
            threading.Event().wait(timeout=0.0001)

        with sim_lock:
            now = sim_time[0]
        if now >= t_end:
            break

        # Simulate variable inference latency
        obs_time = now
        latency = rng.uniform(*INFERENCE_LATENCY_RANGE)

        # Wait for latency to elapse (in simulation time)
        arrival_time = obs_time + latency
        while True:
            with sim_lock:
                curr = sim_time[0]
            if curr >= arrival_time or curr >= t_end:
                break
            threading.Event().wait(timeout=0.0001)

        # Predict action chunk (observation was taken at obs_time)
        t_chunk, x_chunk = vla.predict(obs_time)

        # Log raw chunk
        log_time_series(
            f"raw_chunks/chunk_{chunk_idx:02d}",
            t_chunk,
            x_chunk,
            dim_names=DIM_NAMES,
            dim_first=True,
        )

        # Store for post-hoc analysis
        all_chunks.append((t_chunk.copy(), x_chunk.copy()))

        # Feed into all filters
        for filt in filters.values():
            filt.update_chunk(t_chunk, x_chunk)

        chunk_idx += 1
        # Schedule next inference at INFERENCE_TRIGGER fraction of chunk consumed
        next_query_time = obs_time + chunk_duration * INFERENCE_TRIGGER


# ---------------------------------------------------------------------------
# Control thread (consumer)
# ---------------------------------------------------------------------------
def _control_thread(
    filters: dict[str, AsyncFilter],
    t_ctrl: np.ndarray,
    x_gt_ctrl: np.ndarray,
    sim_time: list[float],
    sim_lock: threading.Lock,
) -> None:
    """Run the control loop at CONTROL_HZ, querying filters each step."""
    for ti, xi_gt in zip(t_ctrl.astype(float), x_gt_ctrl):
        with sim_lock:
            sim_time[0] = ti

        log_scalar("ground_truth", ti, xi_gt, dim_names=DIM_NAMES, dim_first=True)

        for name, filt in filters.items():
            # Update current time for RAIL filters (used as blend start)
            if hasattr(filt, "set_current_time"):
                filt.set_current_time(ti)
            val = filt.get_output(t=ti)
            if val is not None:
                log_scalar(name, ti, val, dim_names=DIM_NAMES, dim_first=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--save", type=str, default=None, help="Save .rrd file path")
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)

    # --- ground-truth trajectory ---
    t_full = np.arange(0, DURATION, 1.0 / CONTROL_HZ)
    x_gt = _ground_truth(t_full)

    # --- control-rate timestamps and ground-truth for plotting ---
    t_ctrl = t_full.copy()
    x_gt_ctrl = x_gt.copy()

    # --- rerun init ---
    init_recording(
        app_id="vla-inference-example",
        spawn=(args.save is None),
        save_path=args.save,
    )
    send_dim_blueprint(DIM_NAMES)

    # --- series styles ---
    for dim in DIM_NAMES:
        configure_series_style(
            f"{dim}/ground_truth", color=(0, 200, 0), name="ground_truth", width=2.0,
        )
        configure_series_style(f"{dim}/act", color=(255, 80, 80), name="act", width=1.5)
        configure_series_style(
            f"{dim}/rail", color=(80, 80, 255), name="rail (dual)", width=1.5,
        )
        configure_series_style(
            f"{dim}/rail_single", color=(120, 120, 255), name="rail (single)", width=1.5,
        )
        configure_series_style(
            f"{dim}/ema", color=(255, 160, 0), name="ema", width=1.5,
        )
        configure_series_style(
            f"{dim}/one_euro", color=(200, 0, 200), name="one_euro", width=1.5,
        )
        configure_series_style(
            f"{dim}/spline", color=(0, 200, 200), name="spline", width=1.5,
        )

    # --- filters ---
    buf = CHUNK_HORIZON * 4
    filters: dict[str, AsyncFilter] = {
        "act": AsyncFilter(method="act", buffer_size=buf, k=0.8, max_chunks=6),
        "rail": AsyncFilter(
            method="rail", buffer_size=buf, poly_degree=3,
            dual_quintic=True, auto_align=True,
        ),
        "rail_single": AsyncFilter(
            method="rail", buffer_size=buf, poly_degree=3,
            dual_quintic=False,
        ),
        "ema": AsyncFilter(method="ema", buffer_size=buf, alpha=0.15),
        "one_euro": AsyncFilter(
            method="one_euro", buffer_size=buf,
            min_cutoff=2.0, beta=0.01, d_cutoff=1.0,
        ),
        "spline": AsyncFilter(method="spline", buffer_size=buf),
    }

    # --- shared simulation clock ---
    sim_time: list[float] = [float(t_ctrl[0]) - 1.0]
    sim_lock = threading.Lock()
    all_chunks: list[tuple[np.ndarray, np.ndarray]] = []

    # --- VLA model ---
    vla = _PseudoVLA(t_full, x_gt, seed=SEED)

    # --- launch threads ---
    inference = threading.Thread(
        target=_inference_thread,
        args=(vla, filters, sim_time, sim_lock, float(t_ctrl[-1]), rng, all_chunks),
        daemon=True,
    )
    control = threading.Thread(
        target=_control_thread,
        args=(filters, t_ctrl, x_gt_ctrl, sim_time, sim_lock),
    )

    inference.start()
    control.start()

    control.join()
    inference.join(timeout=1.0)

    print(
        f"Done. {len(all_chunks)} action chunks generated over {DURATION:.0f}s "
        f"({CONTROL_HZ:.0f}Hz control, {CHUNK_HZ:.0f}Hz chunk prediction)."
    )


if __name__ == "__main__":
    main()
