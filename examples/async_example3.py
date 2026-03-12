#!/usr/bin/env python3
"""Single-filter VLA simulation with reactive chunk generation.

Demonstrates a realistic VLA (Vision-Language-Action) control loop where
a single async filter is selected at the command line.  Unlike the
multi-filter comparison examples, chunk generation here is **reactive**:

* Each new chunk starts from the filter's current output position (plus a
  small observation offset), mimicking a VLA model that conditions its
  prediction on the robot's actual state.
* The closest point on the base trajectory is found, and the chunk follows
  the base path from there — with both high-frequency Gaussian noise and
  low-frequency drift noise added.
* Simulated inference latency is applied before the chunk is fed to the
  filter.

Usage::

    python examples/single_filter_vla_example.py                     # default: rail
    python examples/single_filter_vla_example.py --method act
    python examples/single_filter_vla_example.py --method ema --save out.rrd
"""

from __future__ import annotations

import argparse
import os
import sys
import threading

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
DURATION = 8.0            # trajectory length (seconds)
CONTROL_HZ = 50.0        # robot control loop rate
CHUNK_HZ = 10.0          # action prediction rate within a chunk
CHUNK_HORIZON = 16        # timesteps per chunk
INFERENCE_TRIGGER = 0.5   # re-query at this fraction of chunk consumed
INFERENCE_LATENCY = (0.10, 0.40)   # uniform random latency range (s)
OBS_OFFSET_STD = 0.005    # observation offset (simulates state error)
NOISE_HF_STD = 0.008      # high-frequency noise std
NOISE_LF_STD = 0.025      # low-frequency drift noise amplitude
NOISE_LF_PERIOD = 1.5     # low-frequency noise period (seconds)
SEED = 54321

AVAILABLE_METHODS = [
    "rail", "act", "ema", "linear", "spline", "one_euro", "moving_average",
]

# Default filter keyword arguments per method
_FILTER_KWARGS: dict[str, dict] = {
    "rail": dict(
        poly_degree=3, dual_quintic=True, auto_align=True,
        blend_order="cubic", blend_start_source="actual_output",
    ),
    "act": dict(k=0.01, max_chunks=6),
    "ema": dict(alpha=0.15),
    "linear": dict(),
    "spline": dict(),
    "one_euro": dict(min_cutoff=2.0, beta=0.01, d_cutoff=1.0),
    "moving_average": dict(window=8),
}


# ---------------------------------------------------------------------------
# Ground-truth base trajectory (pick-and-place)
# ---------------------------------------------------------------------------
def _ground_truth(t: np.ndarray) -> np.ndarray:
    """Smooth pick-and-place XYZ trajectory. Returns shape (N, 3)."""
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
    )
    z += 0.20
    return np.column_stack([x, y, z])


# ---------------------------------------------------------------------------
# Reactive chunk generator
# ---------------------------------------------------------------------------
class _ReactiveVLA:
    """Generates action chunks whose start point depends on filter output.

    Parameters
    ----------
    t_base, x_base : ndarray
        Dense base trajectory used for chunk path generation.
    seed : int
        RNG seed.
    """

    def __init__(
        self, t_base: np.ndarray, x_base: np.ndarray, seed: int
    ) -> None:
        self._t = t_base
        self._x = x_base           # (N, 3)
        self._rng = np.random.default_rng(seed)
        # Low-frequency drift state (random walk, one per dimension)
        self._lf_phase = self._rng.uniform(0, 2 * np.pi, size=3)

    def generate_chunk(
        self, current_pos: np.ndarray, t_now: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a noisy chunk that starts near *current_pos*.

        1. Offset *current_pos* by a small observation error.
        2. Find the closest point on the base trajectory.
        3. Walk forward from there for CHUNK_HORIZON steps.
        4. Add high-frequency Gaussian noise + low-frequency drift.

        Returns (t_chunk, x_chunk) with shapes (H,) and (H, 3).
        """
        # --- observation offset ---
        obs_pos = current_pos + self._rng.normal(scale=OBS_OFFSET_STD, size=3)

        # --- find closest base-trajectory point ---
        dists = np.linalg.norm(self._x - obs_pos, axis=1)
        idx_start = int(np.argmin(dists))

        # --- generate chunk timestamps & path ---
        dt = 1.0 / CHUNK_HZ
        t_chunk = t_now + np.arange(CHUNK_HORIZON) * dt
        # Trim to trajectory end
        t_chunk = t_chunk[t_chunk <= self._t[-1]]
        if len(t_chunk) == 0:
            t_chunk = np.array([min(t_now, self._t[-1])])
        n_pts = len(t_chunk)

        # Interpolate base path starting from the matched index
        t_base_start = self._t[idx_start]
        t_query = t_base_start + np.arange(n_pts) * dt
        t_query = np.clip(t_query, self._t[0], self._t[-1])

        x_chunk = np.column_stack(
            [np.interp(t_query, self._t, self._x[:, d]) for d in range(3)]
        )
        # Shift so the first point matches obs_pos (smooth handoff)
        x_chunk += obs_pos - x_chunk[0]

        # --- high-frequency noise ---
        hf_noise = self._rng.normal(scale=NOISE_HF_STD, size=(n_pts, 3))

        # --- low-frequency drift noise (sinusoidal with slowly drifting phase) ---
        self._lf_phase += self._rng.normal(scale=0.3, size=3)
        lf_t = np.linspace(0, 1, n_pts)
        lf_noise = np.column_stack(
            [
                NOISE_LF_STD * np.sin(
                    2 * np.pi * lf_t / NOISE_LF_PERIOD + self._lf_phase[d]
                )
                for d in range(3)
            ]
        )

        x_chunk += hf_noise + lf_noise
        return t_chunk, x_chunk


# ---------------------------------------------------------------------------
# Inference thread (producer)
# ---------------------------------------------------------------------------
def _inference_thread(
    vla: _ReactiveVLA,
    filt: AsyncFilter,
    sim_time: list[float],
    sim_lock: threading.Lock,
    t_end: float,
    rng: np.random.Generator,
    all_chunks: list,
) -> None:
    """Periodically run reactive VLA inference and feed chunks to filter."""
    chunk_duration = CHUNK_HORIZON / CHUNK_HZ
    next_query_time = 0.0
    chunk_idx = 0

    while True:
        # Wait until sim clock reaches next_query_time
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

        # Record observation time and get current filter output
        obs_time = now
        if hasattr(filt, "set_current_time"):
            filt.set_current_time(obs_time)
        current_output = filt.get_output(t=obs_time)
        if current_output is None:
            # Before first chunk: use base trajectory at obs_time
            current_output = vla._x[
                int(np.argmin(np.abs(vla._t - obs_time)))
            ]

        # Simulate inference latency
        latency = rng.uniform(*INFERENCE_LATENCY)
        arrival_time = obs_time + latency
        while True:
            with sim_lock:
                curr = sim_time[0]
            if curr >= arrival_time or curr >= t_end:
                break
            threading.Event().wait(timeout=0.0001)

        with sim_lock:
            curr = sim_time[0]
        if curr >= t_end:
            break

        # Generate reactive chunk (starts near current filter output)
        t_chunk, x_chunk = vla.generate_chunk(current_output, obs_time)

        # Log raw chunk for visualisation
        log_time_series(
            f"raw_chunks/chunk_{chunk_idx:02d}",
            t_chunk,
            x_chunk,
            dim_names=DIM_NAMES,
            dim_first=True,
        )
        all_chunks.append((t_chunk.copy(), x_chunk.copy()))

        # Update current time and feed to filter
        with sim_lock:
            now_after = sim_time[0]
        if hasattr(filt, "set_current_time"):
            filt.set_current_time(now_after)
        filt.update_chunk(t_chunk, x_chunk)

        chunk_idx += 1
        next_query_time = obs_time + chunk_duration * INFERENCE_TRIGGER


# ---------------------------------------------------------------------------
# Control thread (consumer)
# ---------------------------------------------------------------------------
def _control_thread(
    filt: AsyncFilter,
    t_ctrl: np.ndarray,
    x_gt_ctrl: np.ndarray,
    sim_time: list[float],
    sim_lock: threading.Lock,
) -> None:
    """Run the control loop at CONTROL_HZ, querying the filter each step."""
    for ti, xi_gt in zip(t_ctrl.astype(float), x_gt_ctrl):
        with sim_lock:
            sim_time[0] = ti

        if hasattr(filt, "set_current_time"):
            filt.set_current_time(ti)

        log_scalar(
            "ground_truth", ti, xi_gt, dim_names=DIM_NAMES, dim_first=True
        )

        val = filt.get_output(t=ti)
        if val is not None:
            log_scalar(
                "filtered", ti, val, dim_names=DIM_NAMES, dim_first=True
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--method",
        type=str,
        default="rail",
        choices=AVAILABLE_METHODS,
        help="Async filter method to use (default: rail)",
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Save .rrd file path"
    )
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)

    # --- ground-truth trajectory ---
    t_full = np.arange(0, DURATION, 1.0 / CONTROL_HZ)
    x_gt = _ground_truth(t_full)

    t_ctrl = t_full.copy()
    x_gt_ctrl = x_gt.copy()

    # --- rerun init ---
    init_recording(
        app_id=f"single-filter-vla-{args.method}",
        spawn=(args.save is None),
        save_path=args.save,
    )
    send_dim_blueprint(DIM_NAMES)

    # --- series styles ---
    for dim in DIM_NAMES:
        configure_series_style(
            f"{dim}/ground_truth",
            color=(0, 200, 0),
            name="ground_truth",
            width=2.0,
        )
        configure_series_style(
            f"{dim}/filtered",
            color=(80, 80, 255),
            name=args.method,
            width=2.0,
        )

    # --- filter ---
    buf = CHUNK_HORIZON * 4
    kwargs = _FILTER_KWARGS.get(args.method, {})
    filt = AsyncFilter(method=args.method, buffer_size=buf, **kwargs)
    print(f"Using filter: {args.method}  ({type(filt).__name__})")

    # --- reactive VLA model ---
    vla = _ReactiveVLA(t_full, x_gt, seed=SEED)

    # --- shared simulation clock ---
    sim_time: list[float] = [float(t_ctrl[0]) - 1.0]
    sim_lock = threading.Lock()
    all_chunks: list[tuple[np.ndarray, np.ndarray]] = []

    # --- launch threads ---
    inference = threading.Thread(
        target=_inference_thread,
        args=(vla, filt, sim_time, sim_lock, float(t_ctrl[-1]), rng, all_chunks),
        daemon=True,
    )
    control = threading.Thread(
        target=_control_thread,
        args=(filt, t_ctrl, x_gt_ctrl, sim_time, sim_lock),
    )

    inference.start()
    control.start()
    control.join()
    inference.join(timeout=2.0)

    print(f"Done. {len(all_chunks)} chunks generated, logged to rerun.")


if __name__ == "__main__":
    main()
