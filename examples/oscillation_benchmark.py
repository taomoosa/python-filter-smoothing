#!/usr/bin/env python3
"""Oscillation suppression benchmark — ACT vs RAIL vs RTC-RAIL.

Demonstrates how action-chunk filtering methods handle **mode-switching
oscillation**: a scenario where the pseudo-policy alternates between two
distinct baselines depending on the current position, producing chunks
that flip-flop between two attractors.

Setup
-----
* 1-D action (scalar position).
* Two monotonic baselines that cross at a configurable time:
  ``y_A(t) = +slope·(t − t_cross)`` (increasing) and
  ``y_B(t) = −slope·(t − t_cross)`` (decreasing).
  Near the crossing both baselines are close, creating a natural
  oscillation zone where the policy frequently switches.
* **Per-filter feedback**: each filter maintains its own position state.
  The policy selects a baseline stochastically, biased toward the one
  closer to **that filter's current output** (softmax of neg distance).
  This creates a realistic feedback loop:

  - ACT oscillates → stays near midline → ~50/50 coin flip →
    more oscillation (vicious cycle)
  - RTC-RAIL commits to one baseline → distance grows →
    biased toward same baseline → self-stabilisation (virtuous cycle)

* The chunk is generated **solely from the state at inference start**:
  it ramps from the selected baseline's value at ``t_obs`` with that
  baseline's fixed slope.  Future time evolution is not consulted.
* Gaussian noise is added to each chunk.

Filters compared
----------------
1. ``act``        – ACT temporal ensembling (k=0.01, max_chunks=5)
2. ``rail``       – RAIL polynomial fit + quintic blend
3. ``rtc_rail``   – RTC-RAIL hermite inpainting + cubic blend
4. ``rtc_rail_q`` – RTC-RAIL hermite inpainting + quintic blend

Usage::

    python examples/oscillation_benchmark.py          # run and print results
    python examples/oscillation_benchmark.py --seed 0 # custom seed
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from python_filter_smoothing import (
    AsyncFilter,
    AsyncFilterRTC,
    AsyncFilterRTCRAIL,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
DURATION = 24.0       # total simulation time [s]
CONTROL_HZ = 100.0    # control loop rate
CHUNK_HZ = 10.0      # action-chunk sample rate
CHUNK_HORIZON = 16   # H steps per chunk
LATENCY_RANGE = (0.15, 0.30)  # inference latency [s]
NOISE_STD = 0.02     # Gaussian noise on chunk actions
SOFTMAX_TEMP = 0.15  # softmax temperature for baseline selection
MIN_SWITCH_PROB = 0.3  # floor probability for choosing the farther baseline
SEED = 42

# Baseline parameters (monotonic, crossing at midpoint)
BASELINE_SLOPE = 0.05   # [units / s]
BASELINE_CROSS_TIME = DURATION / 3.0  # baselines cross at t = 4s


# ---------------------------------------------------------------------------
# Two monotonic baselines (crossing at midpoint)
# ---------------------------------------------------------------------------
def baseline_a(t: np.ndarray) -> np.ndarray:
    """Baseline A: monotonically increasing."""
    return BASELINE_SLOPE * (t - BASELINE_CROSS_TIME)


def baseline_b(t: np.ndarray) -> np.ndarray:
    """Baseline B: monotonically decreasing."""
    return -BASELINE_SLOPE * (t - BASELINE_CROSS_TIME)


# ---------------------------------------------------------------------------
# Pseudo-policy: stochastic baseline selection + noise
# ---------------------------------------------------------------------------
class BimodalPolicy:
    """Generates action chunks by probabilistically following one of two baselines.

    At each inference step, the policy evaluates both baselines at the
    observation time, selects one stochastically (biased toward the
    closer baseline via softmax of negative distance), and generates a
    chunk that ramps from the selected baseline's value at ``t_obs``
    with that baseline's fixed slope.

    To support per-filter feedback, the selection and noise randomness
    are passed in externally (``u_select`` and ``noise``).  All filters
    share the same random draws, so the only source of divergence is
    the position-dependent probability.
    """

    def __init__(self, temperature: float = SOFTMAX_TEMP,
                 min_switch_prob: float = MIN_SWITCH_PROB):
        self._temp = temperature
        self._min_sw = min_switch_prob

    def predict(
        self,
        t_obs: float,
        current_pos: float,
        u_select: float,
        noise: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int, float]:
        """Return (t_chunk, x_chunk, choice, prob_a).

        Parameters
        ----------
        t_obs : float
            Observation (inference start) time.
        current_pos : float
            Current position of the filter being served.
        u_select : float
            Shared uniform draw in [0, 1) for baseline selection.
        noise : np.ndarray, shape (H,)
            Shared Gaussian noise for the chunk.
        """
        dt = 1.0 / CHUNK_HZ
        t_chunk = t_obs + np.arange(CHUNK_HORIZON) * dt

        # Evaluate both baselines at observation time only
        ya = float(baseline_a(np.array([t_obs]))[0])
        yb = float(baseline_b(np.array([t_obs]))[0])

        # Distance-based selection probability (closer → more likely)
        da = abs(current_pos - ya)
        db = abs(current_pos - yb)
        logits = np.array([-da, -db]) / self._temp
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()

        # Clamp to ensure minimum switching probability
        probs = np.clip(probs, self._min_sw, 1.0 - self._min_sw)
        probs /= probs.sum()

        choice = 0 if u_select < probs[0] else 1
        prob_a = float(probs[0])

        # Generate chunk from state at t_obs only:
        # start at baseline value at t_obs, ramp with fixed slope
        y0 = ya if choice == 0 else yb
        slope = BASELINE_SLOPE if choice == 0 else -BASELINE_SLOPE
        dt_local = t_chunk - t_obs
        x_chunk = y0 + slope * dt_local + noise

        return t_chunk, x_chunk, choice, prob_a


# ---------------------------------------------------------------------------
# Filter configurations
# ---------------------------------------------------------------------------
def make_filters() -> dict:
    H = CHUNK_HORIZON
    DT = 1.0 / CHUNK_HZ
    return {
        "act": AsyncFilter(method="act", k=0.01, max_chunks=5),
        "rail": AsyncFilter(
            method="rail",
            poly_degree=3,
            blend_order="quintic",
            blend_duration=0.2,
        ),
        "rtc_rail": AsyncFilter(
            method="rtc_rail",
            prediction_horizon=H,
            dt=DT,
            min_execution_horizon=4,
            initial_delay=2,
            inpainting="hermite",
            poly_degree=3,
            blend_order="cubic",
        ),
        "rtc_rail_q": AsyncFilter(
            method="rtc_rail",
            prediction_horizon=H,
            dt=DT,
            min_execution_horizon=4,
            initial_delay=2,
            inpainting="hermite",
            poly_degree=3,
            blend_order="quintic",
        ),
    }


# ---------------------------------------------------------------------------
# Simulation loop (single-threaded, deterministic)
# ---------------------------------------------------------------------------
def run_simulation(seed: int = SEED) -> dict:
    """Run the oscillation benchmark and return results dict.

    Each filter maintains its own position state.  All filters share
    the same random coin-flip (``u_select``) and noise for each chunk,
    so the **only** source of divergence is the position-dependent
    baseline-selection probability.  This cleanly isolates how
    per-filter feedback amplifies or dampens oscillation.
    """
    rng = np.random.default_rng(seed)
    filters = make_filters()
    policy = BimodalPolicy()

    # Per-filter position tracking
    positions = {name: 0.0 for name in filters}

    dt_ctrl = 1.0 / CONTROL_HZ

    # Scheduling: pre-generate chunk arrival schedule (shared across filters)
    sim_t = 0.0
    chunk_schedule = []  # (t_obs, t_arrival)
    while sim_t < DURATION:
        latency = rng.uniform(*LATENCY_RANGE)
        chunk_schedule.append((sim_t, sim_t + latency))
        sim_t += max(latency, CHUNK_HORIZON / CHUNK_HZ * 0.5)

    # Pre-generate shared randomness for all chunks
    select_rng = np.random.default_rng(seed + 1)
    noise_rng = np.random.default_rng(seed + 2)
    u_selects = select_rng.uniform(size=len(chunk_schedule))
    noises = [noise_rng.normal(0, NOISE_STD, size=CHUNK_HORIZON)
              for _ in chunk_schedule]

    # Time-series storage
    ctrl_times = np.arange(0, DURATION, dt_ctrl)
    outputs = {name: np.full(len(ctrl_times), np.nan) for name in filters}
    chunk_logs = {name: [] for name in filters}

    chunk_idx = 0
    for step_i, t in enumerate(ctrl_times):
        # --- deliver any chunks whose arrival time has passed ---
        while chunk_idx < len(chunk_schedule) and chunk_schedule[chunk_idx][1] <= t:
            t_obs, t_arrival = chunk_schedule[chunk_idx]

            for name, filt in filters.items():
                # Each filter gets its own chunk from its own position,
                # but the same coin-flip and noise
                t_chunk, x_chunk, choice, prob_a = policy.predict(
                    t_obs, positions[name],
                    u_selects[chunk_idx], noises[chunk_idx],
                )
                x_2d = x_chunk[:, None]

                chunk_logs[name].append({
                    "t_obs": t_obs,
                    "choice": int(choice),
                    "prob_a": float(prob_a),
                    "t_chunk": t_chunk.tolist(),
                    "x_chunk": x_chunk.tolist(),
                })

                if hasattr(filt, "set_current_time"):
                    filt.set_current_time(t)
                if isinstance(filt, (AsyncFilterRTC, AsyncFilterRTCRAIL)):
                    filt.start_inference()
                filt.update_chunk(t_chunk, x_2d)

            chunk_idx += 1

        # --- query each filter and update its position ---
        for name, filt in filters.items():
            if hasattr(filt, "set_current_time"):
                filt.set_current_time(t)
            out = filt.get_output(t)
            if out is not None:
                outputs[name][step_i] = float(out[0])
                positions[name] = float(out[0])

    # --- Compute metrics ---
    ba = baseline_a(ctrl_times)
    bb = baseline_b(ctrl_times)

    metrics = {}
    for name in filters:
        y = outputs[name]
        valid = ~np.isnan(y)
        if valid.sum() < 10:
            metrics[name] = {"error": "insufficient data"}
            continue

        yv = y[valid]
        tv = ctrl_times[valid]
        bav = baseline_a(tv)
        bbv = baseline_b(tv)

        # Distance to nearest baseline at each point
        dist_nearest = np.minimum(np.abs(yv - bav), np.abs(yv - bbv))

        # Oscillation metric: count sign changes in (y - midline)
        # where midline = 0. More sign changes = more oscillation
        sign_changes = np.sum(np.abs(np.diff(np.sign(yv)))) / 2

        # Velocity and jerk
        vel = np.diff(yv) / np.diff(tv)
        jerk = np.diff(vel) / np.diff(tv[:-1]) if len(vel) > 1 else np.array([0.0])

        # Zero-crossing rate of velocity (direction reversals)
        vel_sign_changes = np.sum(np.abs(np.diff(np.sign(vel)))) / 2

        # Chunk choice statistics
        choices = [c["choice"] for c in chunk_logs[name]]
        n_chunks = len(choices)
        switches = sum(1 for i in range(1, n_chunks) if choices[i] != choices[i - 1])
        avg_prob_a = np.mean([c["prob_a"] for c in chunk_logs[name]]) if n_chunks else 0.5

        metrics[name] = {
            "rms_dist_nearest_baseline": float(np.sqrt(np.mean(dist_nearest ** 2))),
            "mean_dist_nearest_baseline": float(np.mean(dist_nearest)),
            "sign_changes": int(sign_changes),
            "velocity_reversals": int(vel_sign_changes),
            "jerk_rms": float(np.sqrt(np.mean(jerk ** 2))),
            "jerk_max": float(np.max(np.abs(jerk))),
            "valid_fraction": float(valid.mean()),
            "chunk_switches": int(switches),
            "chunk_switch_rate": float(switches / max(n_chunks - 1, 1)),
            "avg_prob_a": float(avg_prob_a),
        }

    return {
        "ctrl_times": ctrl_times.tolist(),
        "baseline_a": ba.tolist(),
        "baseline_b": bb.tolist(),
        "outputs": {k: v.tolist() for k, v in outputs.items()},
        "metrics": metrics,
        "chunks": chunk_logs,
        "params": {
            "duration": DURATION,
            "control_hz": CONTROL_HZ,
            "chunk_hz": CHUNK_HZ,
            "chunk_horizon": CHUNK_HORIZON,
            "latency_range": list(LATENCY_RANGE),
            "noise_std": NOISE_STD,
            "softmax_temp": SOFTMAX_TEMP,
            "min_switch_prob": MIN_SWITCH_PROB,
            "seed": seed,
            "per_filter_feedback": True,
        },
    }


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------
def print_summary(results: dict) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("Oscillation Suppression Benchmark Results")
    lines.append("=" * 80)
    p = results["params"]
    lines.append(f"Duration: {p['duration']}s | Control: {p['control_hz']}Hz | "
                 f"Chunk: {p['chunk_hz']}Hz x {p['chunk_horizon']} steps | "
                 f"Seed: {p['seed']}")
    lines.append(f"Latency: {p['latency_range']}s | Noise σ: {p['noise_std']} | "
                 f"Softmax temp: {p['softmax_temp']} | "
                 f"Per-filter feedback: {p.get('per_filter_feedback', False)}")
    lines.append("-" * 80)
    lines.append(f"{'Method':<16} {'SignChg':>8} {'VelRev':>8} "
                 f"{'JerkRMS':>10} {'DistNear':>10} {'SwRate':>8} {'Valid%':>8}")
    lines.append("-" * 80)

    for name in ["act", "rail", "rtc_rail", "rtc_rail_q"]:
        m = results["metrics"].get(name, {})
        if "error" in m:
            lines.append(f"{name:<16} {'(insufficient data)':>58}")
            continue
        lines.append(
            f"{name:<16} "
            f"{m['sign_changes']:>8d} "
            f"{m['velocity_reversals']:>8d} "
            f"{m['jerk_rms']:>10.4f} "
            f"{m['mean_dist_nearest_baseline']:>10.4f} "
            f"{m.get('chunk_switch_rate', 0):>7.1%} "
            f"{m['valid_fraction'] * 100:>7.1f}%"
        )
    lines.append("=" * 80)
    lines.append("")
    lines.append("Key: SignChg  = midline zero-crossings (oscillation indicator)")
    lines.append("     VelRev  = velocity reversals (direction changes)")
    lines.append("     JerkRMS = RMS jerk (smoothness, lower = smoother)")
    lines.append("     DistNear= mean distance to nearest baseline")
    lines.append("     SwRate  = chunk baseline switch rate (per-filter)")
    lines.append("     Valid%  = fraction of timesteps with filter output")
    text = "\n".join(lines)
    print(text)
    return text


# ---------------------------------------------------------------------------
# Rerun visualization
# ---------------------------------------------------------------------------
def visualize_results(results: dict, *, save_path: str | None = None) -> None:
    """Send benchmark results to the Rerun viewer."""
    from python_filter_smoothing.visualize import (
        configure_series_style,
        init_recording,
        log_time_series,
    )
    import rerun as rr
    import rerun.blueprint as rrb

    if save_path:
        init_recording("oscillation_benchmark", save_path=save_path)
    else:
        init_recording("oscillation_benchmark", spawn=True)

    t = np.array(results["ctrl_times"])
    ba = np.array(results["baseline_a"])
    bb = np.array(results["baseline_b"])

    # Blueprint: 4 panels — position, velocity, selection probability, chunks
    views = [
        rrb.TimeSeriesView(
            origin="/position",
            name="Position (filter outputs)",
        ),
        rrb.TimeSeriesView(
            origin="/velocity",
            name="Velocity (smoothness)",
        ),
        rrb.TimeSeriesView(
            origin="/selection",
            name="Baseline A probability (per filter)",
        ),
        rrb.TimeSeriesView(
            origin="/chunks",
            name="Raw input chunks (ACT)",
        ),
    ]
    blueprint = rrb.Blueprint(
        rrb.Vertical(*views),
        auto_views=False,
    )
    rr.send_blueprint(blueprint)

    # --- Baselines ---
    log_time_series("position/baseline_A", t, ba)
    configure_series_style(
        "position/baseline_A", color=[180, 180, 180], width=2.0,
        name="baseline A (+slope)",
    )
    log_time_series("position/baseline_B", t, bb)
    configure_series_style(
        "position/baseline_B", color=[140, 140, 140], width=2.0,
        name="baseline B (−slope)",
    )

    # --- Filter outputs ---
    filter_styles = {
        "act":        {"color": [220, 60, 60],   "name": "ACT",               "width": 1.5},
        "rail":       {"color": [60, 180, 60],   "name": "RAIL",              "width": 1.5},
        "rtc_rail":   {"color": [40, 140, 255],  "name": "RTC-RAIL (cubic)",  "width": 2.0},
        "rtc_rail_q": {"color": [200, 100, 255], "name": "RTC-RAIL (quintic)","width": 1.5},
    }

    for name, style in filter_styles.items():
        y = np.array(results["outputs"][name])
        valid = ~np.isnan(y)
        if valid.sum() < 2:
            continue
        tv = t[valid]
        yv = y[valid]

        # Position
        log_time_series(f"position/{name}", tv, yv)
        configure_series_style(f"position/{name}", **style)

        # Velocity
        vel = np.diff(yv) / np.diff(tv)
        t_vel = (tv[:-1] + tv[1:]) / 2.0
        log_time_series(f"velocity/{name}", t_vel, vel)
        configure_series_style(f"velocity/{name}",
                               color=style["color"], name=style["name"])

    # --- Per-filter baseline selection probability ---
    # 0.5 reference line
    sel_ref_t = np.array([t[0], t[-1]])
    log_time_series("selection/ref_50pct", sel_ref_t, np.array([0.5, 0.5]))
    configure_series_style("selection/ref_50pct",
                           color=[180, 180, 180], width=1.0, name="50%")

    chunks_dict = results["chunks"]
    for name, style in filter_styles.items():
        if name not in chunks_dict:
            continue
        clist = chunks_dict[name]
        if not clist:
            continue
        t_sel = np.array([c["t_obs"] for c in clist])
        p_a = np.array([c["prob_a"] for c in clist])
        log_time_series(f"selection/{name}", t_sel, p_a)
        configure_series_style(f"selection/{name}",
                               color=style["color"], name=style["name"])

    # --- Raw chunks (show ACT's chunks as representative) ---
    act_chunks = chunks_dict.get("act", [])
    chunk_colors_a = [[255, 200, 150], [255, 180, 120]]  # warm for baseline A
    chunk_colors_b = [[150, 200, 255], [120, 180, 255]]  # cool for baseline B
    for ci, ch in enumerate(act_chunks):
        t_c = np.array(ch["t_chunk"])
        x_c = np.array(ch["x_chunk"])
        choice = ch["choice"]
        tag = f"chunks/chunk_{ci:03d}"
        log_time_series(tag, t_c, x_c)
        colors = chunk_colors_a if choice == 0 else chunk_colors_b
        configure_series_style(tag, color=colors[ci % 2], width=0.7,
                               name=f"chunk {ci} ({'A' if choice == 0 else 'B'})")

    print("Visualization sent to Rerun." if not save_path
          else f"Saved to {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=SEED, help="RNG seed")
    parser.add_argument("--save-json", type=str, default=None,
                        help="Save full results to JSON file")
    parser.add_argument("--save", type=str, default=None,
                        help="Save .rrd to file instead of spawning viewer")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip Rerun visualization")
    args = parser.parse_args()

    results = run_simulation(seed=args.seed)
    summary = print_summary(results)

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results saved to {args.save_json}")

    if not args.no_viz:
        try:
            visualize_results(results, save_path=args.save)
        except ImportError:
            print("\n(Install rerun-sdk for visualization: pip install rerun-sdk)")


if __name__ == "__main__":
    main()
