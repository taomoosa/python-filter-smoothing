"""python-filter-smoothing: Filter and smoothing utilities for time series data.

Four classes cover the main processing modes described in the project spec:

* :class:`OfflineFilter`  – entire time series given at once, processed offline.
* :class:`OnlineFilter`   – one sample at a time, processed online.
* :class:`ChunkFilter`    – chunks of samples added online (may overlap in time).
* :class:`AsyncFilter`    – thread-safe, data arrives at random intervals.
"""

from .offline import OfflineFilter
from .online import OnlineFilter
from .chunk import ChunkFilter
from .async_filter import AsyncFilter

__all__ = ["OfflineFilter", "OnlineFilter", "ChunkFilter", "AsyncFilter"]
