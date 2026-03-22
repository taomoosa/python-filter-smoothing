"""python-filter-smoothing: Filter and smoothing utilities for time series data.

Four filter classes cover the main processing modes:

* :class:`OfflineFilter`  – batch processing of a complete time series.
* :class:`OnlineFilter`   – causal, sample-by-sample processing.
* :class:`ChunkFilter`    – incremental chunk-based processing with overlap handling.
* :class:`AsyncFilter`    – thread-safe asynchronous filtering with arbitrary query times.
"""

from .offline import OfflineFilter
from .online import (
    OnlineFilter,
    OnlineFilterBase,
    OnlineFilterEMA,
    OnlineFilterFIR,
    OnlineFilterIIR,
    OnlineFilterKalman,
    OnlineFilterLowpass,
    OnlineFilterMovingAverage,
    OnlineFilterOneEuro,
)
from .chunk import (
    ChunkFilter,
    ChunkFilterBase,
    ChunkFilterFIR,
    ChunkFilterGaussian,
    ChunkFilterIIR,
    ChunkFilterKalman,
    ChunkFilterLinear,
    ChunkFilterLowpass,
    ChunkFilterMedian,
    ChunkFilterPolynomial,
    ChunkFilterSavgol,
    ChunkFilterSpline,
)
from .async_filter import (
    AsyncFilter,
    AsyncFilterACT,
    AsyncFilterBase,
    AsyncFilterEMA,
    AsyncFilterLinear,
    AsyncFilterMovingAverage,
    AsyncFilterOneEuro,
    AsyncFilterRAIL,
    AsyncFilterRTC,
    AsyncFilterRTCRAIL,
    AsyncFilterSpline,
)
from .rtc_utils import (
    rtc_pigdm_denoise_step,
    rtc_pigdm_guidance,
    rtc_soft_mask,
    rtc_training_prepare_batch,
    rtc_training_sample,
)

__all__ = [
    "OfflineFilter",
    "OnlineFilter",
    "OnlineFilterBase",
    "OnlineFilterEMA",
    "OnlineFilterMovingAverage",
    "OnlineFilterLowpass",
    "OnlineFilterOneEuro",
    "OnlineFilterFIR",
    "OnlineFilterIIR",
    "OnlineFilterKalman",
    "ChunkFilter",
    "ChunkFilterBase",
    "ChunkFilterLinear",
    "ChunkFilterSpline",
    "ChunkFilterPolynomial",
    "ChunkFilterSavgol",
    "ChunkFilterGaussian",
    "ChunkFilterLowpass",
    "ChunkFilterMedian",
    "ChunkFilterFIR",
    "ChunkFilterIIR",
    "ChunkFilterKalman",
    "AsyncFilter",
    "AsyncFilterACT",
    "AsyncFilterBase",
    "AsyncFilterEMA",
    "AsyncFilterLinear",
    "AsyncFilterSpline",
    "AsyncFilterOneEuro",
    "AsyncFilterMovingAverage",
    "AsyncFilterRAIL",
    "AsyncFilterRTC",
    "AsyncFilterRTCRAIL",
    "rtc_pigdm_guidance",
    "rtc_pigdm_denoise_step",
    "rtc_soft_mask",
    "rtc_training_prepare_batch",
    "rtc_training_sample",
]

# Visualization helpers are available as python_filter_smoothing.visualize
# but not auto-imported to avoid hard dependency on rerun-sdk.
