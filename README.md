# python-filter-smoothing

Python library for smoothing and filtering time series data.  Supports offline
batch processing, online sample-by-sample filtering, overlapping-chunk
processing, and thread-safe asynchronous filtering.

## Installation

```bash
# Core (numpy + scipy)
uv pip install -e .

# With development tools (pytest)
uv pip install -e ".[dev]"

# With rerun.io visualization
uv pip install -e ".[viz]"

# Everything
uv pip install -e ".[dev,viz]"
```

## Quick Start

```python
import numpy as np
from python_filter_smoothing import OfflineFilter, OnlineFilter, ChunkFilter, AsyncFilter

# --- Offline: process an entire time series at once ---
t = np.linspace(0, 1, 200)
x = np.sin(2 * np.pi * t) + np.random.randn(200) * 0.3
filt = OfflineFilter(t, x)
x_smooth = filt.savgol_filter(window_length=21, polyorder=3)

# --- Online: one sample at a time ---
filt = OnlineFilter(method="ema", alpha=0.2)
for ti, xi in zip(t, x):
    filt.update(ti, xi)
print(filt.get_value())

# --- Chunk: overlapping chunks ---
filt = ChunkFilter(method="spline", overlap_strategy="cosine_blend")
filt.add_chunk(x[:60], t[:60])
filt.add_chunk(x[40:120], t[40:120])  # overlaps [40:60]
result = filt.get_filtered()

# --- Async: thread-safe, query at any time ---
filt = AsyncFilter(method="ema", buffer_size=100, alpha=0.3)
filt.update_chunk(t[:50], x[:50])      # add a whole chunk
output = filt.get_output(t=0.25)       # query from another thread
```

---

## Filter Types

### OfflineFilter

Processes an entire time series at once.  All methods return the smoothed
data array.

```python
filt = OfflineFilter(t, x)   # x: shape (N,) or (N, D)
```

| Method | Key Parameters | Description |
|--------|---------------|-------------|
| `linear_interpolate(t_query)` | `t_query` | Piecewise-linear interpolation |
| `lowpass_filter(cutoff_freq, sample_rate, order=4)` | `cutoff_freq`, `sample_rate` | Zero-phase Butterworth lowpass |
| `polynomial_fit(degree, t_query=None)` | `degree` | Least-squares polynomial fit |
| `spline_interpolate(t_query, kind="cubic")` | `kind` | Scipy `interp1d` spline |
| `savgol_filter(window_length, polyorder)` | `window_length`, `polyorder` | Savitzky-Golay (preserves peaks) |
| `gaussian_filter(sigma)` | `sigma` | Gaussian kernel smoothing |
| `median_filter(kernel_size)` | `kernel_size` | Robust to outlier spikes |
| `moving_average(window_size)` | `window_size` | Sliding-window box-car average |
| `fir_filter(numtaps, cutoff_freq, sample_rate, ...)` | `numtaps`, `cutoff_freq`, `sample_rate`, `window="hamming"`, `pass_zero=True` | Zero-phase FIR filter |
| `iir_filter(cutoff_freq, sample_rate, ...)` | `iir_type="butterworth"`, `btype="low"`, `order=4`, `rp`, `rs` | Zero-phase IIR (Butterworth, Chebyshev I/II, Elliptic, Bessel) |
| `kalman_smooth(...)` | `process_noise=0.01`, `measurement_noise=0.1`, `state_model="position"` | Kalman smoother (RTS) with optional custom F/H/Q/R |

### OnlineFilter

Causal, sample-by-sample processing.  Created via factory or direct subclass.

```python
filt = OnlineFilter(method="ema", alpha=0.3)
filt.update(t, x)       # feed one sample
y = filt.get_value()     # read current state
filt.reset()             # clear state
```

| Method | Class | Key Parameters |
|--------|-------|---------------|
| `ema` | `OnlineFilterEMA` | `alpha=0.3` — smoothing factor ∈ (0, 1] |
| `moving_average` | `OnlineFilterMovingAverage` | `window=10` — sliding window length |
| `lowpass` | `OnlineFilterLowpass` | `cutoff_freq=0.1`, `sample_rate=1.0`, `order=2` |
| `one_euro` | `OnlineFilterOneEuro` | `min_cutoff=1.0`, `beta=0.0`, `d_cutoff=1.0` |
| `fir` | `OnlineFilterFIR` | `numtaps=31`, `cutoff_freq=5.0`, `sample_rate=100.0`, `window="hamming"` |
| `iir` | `OnlineFilterIIR` | `cutoff_freq=5.0`, `sample_rate=100.0`, `iir_type="butterworth"`, `btype="low"`, `rp`, `rs` |
| `kalman` | `OnlineFilterKalman` | `process_noise=0.01`, `measurement_noise=0.1`, `state_model="position"`, `dt=0.01` |

**One Euro Filter** ([Casiez et al., CHI 2012](https://doi.org/10.1145/2207676.2208639)):
adaptively adjusts the lowpass cutoff based on signal speed — low cutoff for
slow movements (more smoothing), high cutoff for fast movements (less lag).

**IIR Filter** — supports 5 filter families, each with different characteristics:

| `iir_type` | Description | Extra Parameters |
|------------|-------------|------------------|
| `butterworth` | Maximally flat passband (default) | — |
| `chebyshev1` | Sharper roll-off, passband ripple | `rp` (ripple dB) |
| `chebyshev2` | Equiripple stopband | `rs` (attenuation dB) |
| `elliptic` | Sharpest roll-off, ripple in both bands | `rp`, `rs` |
| `bessel` | Best phase linearity (minimal group delay distortion) | — |

All IIR types support `btype` = `"low"`, `"high"`, `"bandpass"`, `"bandstop"`.
For bandpass/bandstop, pass `cutoff_freq` as a list of two frequencies.

**Kalman Filter** — two built-in state models, or fully custom matrices:

| `state_model` | State | Description |
|---------------|-------|-------------|
| `position` | `[pos]` | Random-walk model. Good for general denoising. |
| `position_velocity` | `[pos, vel]` | Constant-velocity model. Better for tracking. Requires `dt`. |

For custom models, pass `F` (transition), `H` (observation), `Q` (process noise),
`R` (measurement noise) matrices directly.

### ChunkFilter

Accepts overlapping chunks of data, merges them, and returns a globally
smoothed result.

```python
filt = ChunkFilter(method="spline", overlap_strategy="cosine_blend")
filt.add_chunk(x_chunk, t_chunk)    # add chunks incrementally
result = filt.get_filtered(t_query) # query smoothed output
```

| Method | Class | Key Parameters |
|--------|-------|---------------|
| `linear` | `ChunkFilterLinear` | — |
| `spline` | `ChunkFilterSpline` | `kind="cubic"` |
| `polynomial` | `ChunkFilterPolynomial` | `degree=3` |
| `savgol` | `ChunkFilterSavgol` | `window_length=11`, `polyorder=3` |
| `gaussian` | `ChunkFilterGaussian` | `sigma=3.0` |
| `lowpass` | `ChunkFilterLowpass` | `cutoff_freq=5.0`, `sample_rate=100.0`, `order=4` |
| `median` | `ChunkFilterMedian` | `kernel_size=5` |
| `fir` | `ChunkFilterFIR` | `numtaps=31`, `cutoff_freq=5.0`, `sample_rate=100.0`, `window="hamming"` |
| `iir` | `ChunkFilterIIR` | `cutoff_freq=5.0`, `sample_rate=100.0`, `iir_type="butterworth"`, `btype="low"` |
| `kalman` | `ChunkFilterKalman` | `process_noise=0.01`, `measurement_noise=0.1`, `state_model="position"` |

**Overlap strategies** — how samples with overlapping timestamps are merged:

| Strategy | Description |
|----------|-------------|
| `latest` | Keep the most recently added sample (default) |
| `mean` | Average all samples at overlapping timestamps |
| `blend` | Linear crossfade in the overlap region |
| `cosine_blend` | Cosine (C¹-smooth) crossfade in the overlap region |

### AsyncFilter

Thread-safe filter for data arriving asynchronously.  Maintains a circular
buffer of recent samples and supports interpolated output at arbitrary query
times.  Output between updates is smoothly interpolated via PCHIP.

```python
filt = AsyncFilter(method="ema", buffer_size=100, alpha=0.3)

# Thread A: feed data
filt.update(t, x)                  # single sample
filt.update_chunk(t_array, x_array) # whole chunk

# Thread B: query output at any time
y = filt.get_output(t=current_time)
```

| Method | Class | Key Parameters | Description |
|--------|-------|---------------|-------------|
| `ema` | `AsyncFilterEMA` | `alpha=0.3` | Exponential moving average |
| `linear` | `AsyncFilterLinear` | — | Linear interpolation over buffer |
| `spline` | `AsyncFilterSpline` | — | Cubic spline over buffer |
| `one_euro` | `AsyncFilterOneEuro` | `min_cutoff`, `beta`, `d_cutoff` | Adaptive lowpass |
| `moving_average` | `AsyncFilterMovingAverage` | `window=10` | Sliding-window average |
| `act` | `AsyncFilterACT` | `k=0.01`, `max_chunks=10` | ACT temporal ensembling |
| `rail` | `AsyncFilterRAIL` | `poly_degree=3`, `blend_duration=None`, `dual_quintic=True`, `auto_align=False` | RAIL trajectory post-processing |

#### Action Chunk Methods

The `act` and `rail` methods are designed for **action chunk** filtering in
VLA (Vision-Language-Action) robot control pipelines, where a model
asynchronously predicts multi-step action trajectories and a control loop
must output smooth commands at a fixed rate.

**ACT Temporal Ensembling** ([Zhao et al., RSS 2023](https://tonyzhaozh.github.io/aloha/);
[LeRobot](https://huggingface.co/docs/lerobot/async)):

Maintains a window of recent action chunks.  At query time, computes a weighted
average across all chunks that cover the query timestamp.  Newer chunks receive
exponentially higher weight: `w = exp(-k × age)`.

```python
filt = AsyncFilter(method="act", k=0.01, max_chunks=10)
filt.update_chunk(t_chunk, x_chunk)
y = filt.get_output(t=now)
```

**RAIL** ([Cheng et al., arXiv:2512.24673](https://arxiv.org/abs/2512.24673)):

Two-stage trajectory post-processing ensuring C² continuity:
1. **Intra-chunk smoothing**: polynomial fit per dimension to filter prediction noise.
2. **Inter-chunk fusion**: quintic polynomial blend
   (`scipy.interpolate.BPoly.from_derivatives`) matching position, velocity,
   and acceleration at chunk boundaries.

Additional parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dual_quintic` | `True` | Use dual-quintic spline (Eq. 11-13 of the paper), splitting the blend into two halves to prevent overshoot from Runge's phenomenon.  Set `False` for single quintic. |
| `auto_align` | `False` | Automatically correct new-chunk timestamps via temporal alignment (Eq. 10) that maximises motion-direction consistency.  Requires `set_current_time()`. |
| `align_window` | `None` | Search window (seconds) for auto-alignment.  `None` = 50 % of chunk duration. |

Use `set_current_time(t)` in your control loop so that the blend region starts
at the actual switch time (as in Algorithm 1 of the paper):

```python
filt = AsyncFilter(method="rail", poly_degree=3, dual_quintic=True, auto_align=True)

# Control loop (runs at high frequency)
while running:
    t_now = get_clock()
    filt.set_current_time(t_now)      # update before chunk arrival
    # ... (inference thread calls filt.update_chunk(t_chunk, x_chunk))
    y = filt.get_output(t=t_now)      # smooth output
    send_to_robot(y)
```

---

## Visualization

Time series input/output can be visualized using [rerun.io](https://rerun.io/).

### Visualize during manual testing

```bash
uv pip install -e ".[viz]"

# Spawn the rerun viewer and stream data during tests
pytest --visualize

# Or save to an .rrd file for later viewing
pytest --visualize --rrd-path=recording.rrd
```

### Visualization API

```python
from python_filter_smoothing.visualize import (
    init_recording,
    log_time_series,
    log_scalar,
    send_dim_blueprint,
    configure_series_style,
)

init_recording(spawn=True)

# Log batch data
log_time_series("input/noisy", t, x, dim_names=["X", "Y", "Z"], dim_first=True)

# Log single sample (online filters)
log_scalar("output/ema", t=0.5, x=np.array([1.0, 2.0, 3.0]),
           dim_names=["X", "Y", "Z"], dim_first=True)

# Create per-dimension panel layout
send_dim_blueprint(["X", "Y", "Z"])

# Style a series
configure_series_style("output/ema", color=(255, 0, 0), name="EMA", width=2.0)
```

The `dim_first=True` option creates separate panels for each dimension (X, Y, Z),
with all filter outputs overlaid in each panel — making it easy to compare
methods per dimension.

---

## Examples

The `examples/` folder contains runnable scripts with rerun visualization:

```bash
python examples/offline_example.py    # All offline methods on noisy XYZ data
python examples/online_example.py     # Real-time sample-by-sample filtering
python examples/chunk_example.py      # Streaming chunks with overlap handling
python examples/async_example.py      # Multi-threaded async filtering (all methods)
python examples/async_example2.py     # Simulated inference with ACT & RAIL
python examples/async_example3.py     # Single-filter reactive chunk generation
```

Each script accepts `--save <path>.rrd` to write a recording file instead of
spawning the viewer.

| Example | What it demonstrates |
|---------|---------------------|
| `offline_example.py` | All offline methods compared side-by-side |
| `online_example.py` | EMA, moving average, lowpass, one-euro on streaming data |
| `chunk_example.py` | Chunk-based processing with overlap strategies |
| `async_example.py` | Producer/consumer threads with all async methods |
| `async_example2.py` | Simulated async inference with variable latency, ACT and RAIL |
| `async_example3.py` | Single-filter simulation with reactive (state-dependent) chunk generation |

---

## Architecture

The library uses a **base class + subclass** pattern.  Each filter type has an
abstract base class defining the interface and common logic, with concrete
subclasses implementing specific algorithms.  Factory functions provide
backward-compatible creation:

```
OnlineFilterBase (ABC)          → OnlineFilter("ema")     → OnlineFilterEMA
                                → OnlineFilter("lowpass")  → OnlineFilterLowpass
                                → OnlineFilter("kalman")   → OnlineFilterKalman
                                  ...

ChunkFilterBase (ABC)           → ChunkFilter("spline")   → ChunkFilterSpline
  ├─ overlap merging              ChunkFilter("kalman")    → ChunkFilterKalman
  └─ get_filtered()               ...

AsyncFilterBase (ABC)           → AsyncFilter("ema")      → AsyncFilterEMA
  ├─ thread-safe locking          AsyncFilter("act")       → AsyncFilterACT
  ├─ circular buffer              AsyncFilter("rail")      → AsyncFilterRAIL
  └─ PCHIP output interpolation   ...
```

---

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run with visualization (spawns rerun viewer)
uv run pytest tests/ --visualize

# Run specific module
uv run pytest tests/test_async_filter.py -v
```

210 tests covering all filter types, edge cases, and thread safety.

---

## ⚠️ Disclaimer

**This project's code is substantially generated by AI (GitHub Copilot / Claude)
and has NOT undergone sufficient verification or formal review.** Use at your own
risk.  The authors make no guarantees regarding correctness, numerical stability,
or suitability for safety-critical applications (including but not limited to
real robot control).  Users are strongly encouraged to independently verify
filter behaviour for their specific use case before deployment.

---

## References & Acknowledgements

Several filter algorithms in this library are independent reimplementations
based on concepts described in the following published works:

- **One Euro Filter**: G. Casiez, N. Roussel, D. Vogel.
  "1€ Filter: A Simple Speed-Based Low-Pass Filter for Noisy Input in
  Interactive Systems." *CHI 2012*.
  [DOI:10.1145/2207676.2208639](https://doi.org/10.1145/2207676.2208639) —
  Original reference implementations are BSD/MIT licensed.

- **ACT Temporal Ensembling**: T. Z. Zhao, V. Kumar, S. Levine, C. Finn.
  "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware."
  *RSS 2023*.
  [Project page](https://tonyzhaozh.github.io/aloha/) —
  Original code is MIT licensed.

- **LeRobot Async Inference**: Hugging Face.
  [Documentation](https://huggingface.co/docs/lerobot/async) —
  Apache-2.0 licensed.

- **RAIL**: B. Cheng et al.
  "VLA-RAIL: A Real-Time Asynchronous Inference Linker for VLA Models and
  Robots." [arXiv:2512.24673](https://arxiv.org/abs/2512.24673) —
  Our implementation is based on the mathematical methods (polynomial
  smoothing, quintic C² blending) described in the paper, not derived from
  the authors' code.

This library does **not** contain copied source code from any of the above
projects.  All implementations were written independently from the published
algorithmic descriptions.

---

## License

This project is licensed under the [MIT License](LICENSE).

### Dependency Licenses

| Package | License |
|---------|---------|
| [NumPy](https://numpy.org/) | BSD-3-Clause |
| [SciPy](https://scipy.org/) | BSD-3-Clause |
| [rerun-sdk](https://rerun.io/) (optional) | MIT / Apache-2.0 |
| [pytest](https://pytest.org/) (dev) | MIT |
