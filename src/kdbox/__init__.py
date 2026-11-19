"""KD-Box: Interactive exploration of large-scale multivariate time-series data.

Public API
----------
- `KDBoxWidget` --- Single-variable interactive density and query widget.
- `MultivariateKDBox` --- Coordinated multi-view wrapper with synced overlay plot.
- `from_dataframe` --- Resample, bound, and align long-format Pandas DataFrames.
- `align_time_series` --- Start-, end-, center-, or [0, 1]-align 2D time matrices.
- `PreparedTimeSeries` --- Structured container returned by `from_dataframe`.
"""

from .utils import PreparedTimeSeries, align_time_series, from_dataframe
from .widget import KDBoxWidget, MultivariateKDBox

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "KDBoxWidget",
    "MultivariateKDBox",
    "PreparedTimeSeries",
    "align_time_series",
    "from_dataframe",
]
