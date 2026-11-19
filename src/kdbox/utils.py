"""Preprocessing and alignment utilities for KD-Box time-series datasets.

Real-world sensor and telemetry tables often arrive as long-format DataFrames
with irregular sampling intervals, variable durations, missing values, and
outlier spikes. Because the Numba KD-tree engine operates on rigid
`(n_lines, n_steps)` float32 matrices, this module provides helpers to
clean, bound, align, and resample datasets prior to visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class PreparedTimeSeries:
    """Structured output returned by `from_dataframe`.

    Attributes
    ----------
    time_data : np.ndarray of float32, shape (n_lines, n_steps)
        Uniformly resampled, contiguous timestamp matrix.
    value_data : np.ndarray or dict[str, np.ndarray]
        Single `(n_lines, n_steps)` float32 array when `value_cols` was a
        string, or a mapping `{col_name: array}` when a sequence was passed.
    ids : np.ndarray of object, shape (n_lines,)
        Original series identifiers corresponding to each row index, enabling
        direct lookup from `widget.get_selected_indices()`.
    attributes : dict[str, np.ndarray] or None
        Static per-series categorical metadata arrays (one per column in
        `attr_cols`), or `None` if no attribute columns were requested.
    """

    time_data: np.ndarray
    value_data: Union[np.ndarray, Dict[str, np.ndarray]]
    ids: np.ndarray
    attributes: Optional[Dict[str, np.ndarray]] = None


def align_time_series(time_data: np.ndarray, align: str = "start") -> np.ndarray:
    """Shift or normalize a 2D time matrix for visual shape comparison.

    Arithmetic is performed in `float64` before casting to contiguous
    `float32` so large absolute timestamps (such as Unix epochs) retain
    sub-second precision after shifting.

    Parameters
    ----------
    time_data : array-like of shape (n_lines, n_steps)
        Two-dimensional array of timestamps with `n_steps >= 2`.
    align : {"start", "end", "center", "normalize"}, default="start"
        Alignment strategy applied independently to each row:

        - `"start"`: shift every series to start at `t = 0`.
        - `"end"`: shift every series to end at `t = 0`.
        - `"center"`: shift every series so its temporal midpoint is `0`.
        - `"normalize"`: rescale every series linearly onto `[0.0, 1.0]`.

    Returns
    -------
    aligned : np.ndarray of float32, shape (n_lines, n_steps)
        Contiguous aligned timestamp matrix.

    Raises
    ------
    ValueError
        If `time_data` is not 2D with at least 2 steps, or if `align` is
        not one of the supported modes.
    """
    t_arr = np.asarray(time_data, dtype=np.float64)
    if t_arr.ndim != 2 or t_arr.shape[1] < 2:
        raise ValueError(
            f"time_data must be a 2D array with n_steps >= 2, got shape {t_arr.shape}."
        )

    if align == "start":
        out = t_arr - t_arr[:, 0:1]
    elif align == "end":
        out = t_arr - t_arr[:, -1:]
    elif align == "center":
        out = t_arr - 0.5 * (t_arr[:, 0:1] + t_arr[:, -1:])
    elif align == "normalize":
        starts = t_arr[:, 0:1]
        spans = t_arr[:, -1:] - starts
        # Avoid division by zero for degenerate constant-time rows.
        spans = np.where(spans > 0.0, spans, 1.0)
        out = (t_arr - starts) / spans
    else:
        raise ValueError(
            "align must be one of 'start', 'end', 'center', or 'normalize'."
        )

    return np.ascontiguousarray(out, dtype=np.float32)


def _resample_1d(
    t_src: np.ndarray,
    y_src: np.ndarray,
    t_dst: np.ndarray,
    kind: str,
) -> np.ndarray:
    """Resample a sorted 1D curve onto `t_dst`.

    Uses zero-dependency NumPy implementations for `"linear"`,
    `"previous"`, `"next"`, and `"nearest"`, falling back to
    `scipy.interpolate.interp1d` for higher-order spline kinds.
    """
    if kind == "linear":
        return np.interp(t_dst, t_src, y_src)
    if kind == "previous":
        idx = np.clip(np.searchsorted(t_src, t_dst, side="right") - 1, 0, len(y_src) - 1)
        return y_src[idx]
    if kind == "next":
        idx = np.clip(np.searchsorted(t_src, t_dst, side="left"), 0, len(y_src) - 1)
        return y_src[idx]
    if kind == "nearest":
        idx_right = np.clip(np.searchsorted(t_src, t_dst, side="left"), 0, len(y_src) - 1)
        idx_left = np.clip(idx_right - 1, 0, len(y_src) - 1)
        use_left = np.abs(t_dst - t_src[idx_left]) <= np.abs(t_src[idx_right] - t_dst)
        return np.where(use_left, y_src[idx_left], y_src[idx_right])

    try:
        from scipy.interpolate import interp1d
    except ImportError as exc:
        raise ImportError(
            f"Interpolation kind={kind!r} requires scipy. Install scipy or use "
            "'linear', 'previous', 'next', or 'nearest'."
        ) from exc
    return interp1d(t_src, y_src, kind=kind, assume_sorted=True)(t_dst)


def _parse_numeric_bounds(
    bounds: Sequence[Optional[float]],
    name: str,
) -> Tuple[Optional[float], Optional[float]]:
    """Validate and coerce a `(min, max)` pair of optional finite bounds."""
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        raise ValueError(f"{name} must be a (min, max) pair of length 2.")
    lo, hi = bounds
    lo_f = float(lo) if lo is not None else None
    hi_f = float(hi) if hi is not None else None
    if lo_f is not None and not np.isfinite(lo_f):
        raise ValueError(f"{name} lower bound must be finite or None.")
    if hi_f is not None and not np.isfinite(hi_f):
        raise ValueError(f"{name} upper bound must be finite or None.")
    if lo_f is not None and hi_f is not None and lo_f > hi_f:
        raise ValueError(f"{name} lower bound ({lo_f}) cannot exceed upper bound ({hi_f}).")
    return lo_f, hi_f


def _align_1d(t_raw: np.ndarray, align: Optional[str]) -> np.ndarray:
    """Apply a single-series alignment transform in float64."""
    if align is None:
        return t_raw
    if align == "start":
        return t_raw - t_raw[0]
    if align == "end":
        return t_raw - t_raw[-1]
    if align == "center":
        return t_raw - 0.5 * (t_raw[0] + t_raw[-1])
    if align == "normalize":
        span = t_raw[-1] - t_raw[0]
        return (t_raw - t_raw[0]) / span if span > 0.0 else np.zeros_like(t_raw)
    raise ValueError("align must be None or one of 'start', 'end', 'center', 'normalize'.")


def from_dataframe(
    df: "pd.DataFrame",
    id_col: str,
    time_col: str,
    value_cols: Union[str, Sequence[str]],
    n_steps: int = 100,
    attr_cols: Optional[Sequence[str]] = None,
    align: Optional[str] = "start",
    kind: Union[str, Dict[str, str]] = "linear",
    time_unit: str = "s",
    min_points: int = 2,
    time_range: Optional[Tuple[Optional[Any], Optional[Any]]] = None,
    time_out_of_range: str = "slice",
    duration_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    max_duration_quantile: Optional[float] = None,
    value_range: Optional[
        Union[
            Tuple[Optional[float], Optional[float]],
            Dict[str, Tuple[Optional[float], Optional[float]]],
        ]
    ] = None,
    value_out_of_range: str = "clip",
) -> PreparedTimeSeries:
    """Convert a long-format Pandas DataFrame into uniform KD-Box arrays.

    Handles ragged series lengths, datetime parsing, outlier bounding,
    per-series time alignment, and synchronized multivariate filtering. Any row
    or series discarded by a time, duration, or value constraint is dropped
    identically across `time_data`, all `value_cols`, `ids`, and
    `attributes`.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format table containing at least `id_col`, `time_col`, and
        `value_cols`.
    id_col : str
        Column identifying individual time series.
    time_col : str
        Numeric or datetime/ISO-string timestamp column.
    value_cols : str or sequence of str
        One or more numeric measurement columns to resample.
    n_steps : int, default=100
        Number of uniform sample points per output series (must be `>= 2`).
    attr_cols : sequence of str, optional
        Static metadata columns to extract (first non-null value per series).
    align : {"start", "end", "center", "normalize"} or None, default="start"
        Per-series time alignment applied before `time_range` evaluation.
        Pass `None` to preserve relative/absolute timestamps.
    kind : str or dict[str, str], default="linear"
        Interpolation method passed to `_resample_1d`. May be a single
        string applied to all columns or a per-column dict (e.g.,
        `{"level": "linear", "pump": "previous"}`).
    time_unit : {"ns", "us", "ms", "s", "m", "h", "d"}, default="s"
        Numeric time unit used when converting non-numeric datetime columns.
    min_points : int, default=2
        Minimum number of unique finite timestamps required to retain a series.
    time_range : tuple of (t_min, t_max), optional
        Bounds evaluated in the aligned time frame (either side may be `None`).
        When `time_col` is datetime and `align is None`, bounds may also be
        datetime strings or `pandas.Timestamp` objects.
    time_out_of_range : {"slice", "drop_points", "drop_series"}, default="slice"
        Policy for handling timestamps outside `time_range`:

        - `"slice"`: truncate each series to `time_range` and interpolate
          boundary endpoints so all `n_steps` fall inside the window.
        - `"drop_points"`: drop raw rows outside `time_range` synchronously
          across all columns prior to resampling.
        - `"drop_series"`: drop any entire series extending outside
          `time_range`.
    duration_range : tuple of (min_dur, max_dur), optional
        Whole-series duration bounds (in `time_unit`) applied before alignment.
    max_duration_quantile : float in (0, 1], optional
        Drop series whose duration exceeds this empirical quantile (e.g.,
        `0.95` trims the longest 5% of outlier series).
    value_range : (v_min, v_max) or dict[str, (v_min, v_max)], optional
        Value bounds applied across all value columns or selected columns only.
    value_out_of_range : {"clip", "drop_points", "drop_series"}, default="clip"
        Policy for values outside `value_range`:

        - `"clip"`: clamp bounded columns in place.
        - `"drop_points"`: drop rows where any bounded column is out of range
          (synchronously removing that timestamp from all variables).
        - `"drop_series"`: drop any entire series where any bounded column
          ever falls out of range.

    Returns
    -------
    PreparedTimeSeries
        Dataclass holding `time_data`, `value_data`, `ids`, and
        `attributes`.

    Raises
    ------
    KeyError
        If requested columns are missing from `df` or `value_range` refers
        to a column not present in `value_cols`.
    ValueError
        If parameters are invalid or no valid series remain after filtering.
    """
    import pandas as pd

    if n_steps < 2:
        raise ValueError("n_steps must be >= 2.")
    if min_points < 2:
        raise ValueError("min_points must be >= 2.")
    if align not in (None, "start", "end", "center", "normalize"):
        raise ValueError("align must be None or one of 'start', 'end', 'center', 'normalize'.")
    if time_out_of_range not in ("slice", "drop_points", "drop_series"):
        raise ValueError("time_out_of_range must be one of 'slice', 'drop_points', or 'drop_series'.")
    if value_out_of_range not in ("clip", "drop_points", "drop_series"):
        raise ValueError("value_out_of_range must be one of 'clip', 'drop_points', or 'drop_series'.")
    if max_duration_quantile is not None:
        if not (0.0 < float(max_duration_quantile) <= 1.0):
            raise ValueError("max_duration_quantile must be in the interval (0, 1].")

    is_single_val = isinstance(value_cols, str)
    val_col_list = [value_cols] if is_single_val else list(value_cols)
    if len(val_col_list) == 0:
        raise ValueError("value_cols must specify at least one column.")

    attr_col_list = list(attr_cols) if attr_cols is not None else []
    required_cols = [id_col, time_col] + val_col_list + attr_col_list
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in DataFrame: {missing}")

    # Normalize value_range into a validated per-column dictionary.
    col_val_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    if value_range is not None:
        if isinstance(value_range, dict):
            unknown = [k for k in value_range.keys() if k not in val_col_list]
            if unknown:
                raise KeyError(f"value_range contains columns not in value_cols: {unknown}")
            for col, b in value_range.items():
                col_val_bounds[col] = _parse_numeric_bounds(b, f"value_range[{col!r}]")
        else:
            parsed = _parse_numeric_bounds(value_range, "value_range")
            for col in val_col_list:
                col_val_bounds[col] = parsed

    dur_bounds = (
        _parse_numeric_bounds(duration_range, "duration_range")
        if duration_range is not None
        else None
    )

    work = df[required_cols].copy()
    is_datetime = not pd.api.types.is_numeric_dtype(work[time_col])
    min_ts = None
    scale_factor = 1.0

    # Express timestamps as float64 offsets from the global minimum timestamp
    # to avoid float32 precision loss on epoch-scale dates.
    if is_datetime:
        unit_scale = {
            "ns": 1e9, "us": 1e6, "ms": 1e3, "s": 1.0,
            "m": 1.0 / 60.0, "h": 1.0 / 3600.0, "d": 1.0 / 86400.0,
        }
        if time_unit not in unit_scale:
            raise ValueError(f"time_unit must be one of {list(unit_scale.keys())}.")
        scale_factor = unit_scale[time_unit]
        dt_series = pd.to_datetime(work[time_col], utc=True, errors="coerce")
        min_ts = dt_series.min()
        if pd.isna(min_ts):
            raise ValueError(f"Column {time_col!r} contains no valid datetime values.")
        work["_kd_time_num"] = (dt_series - min_ts).dt.total_seconds() * scale_factor
    else:
        work["_kd_time_num"] = work[time_col].astype(np.float64)

    parsed_time_range: Optional[Tuple[Optional[float], Optional[float]]] = None
    if time_range is not None:
        if not isinstance(time_range, (tuple, list)) or len(time_range) != 2:
            raise ValueError("time_range must be a (t_min, t_max) pair of length 2.")
        converted = []
        for bound in time_range:
            if bound is None:
                converted.append(None)
            elif is_datetime and align is None and not isinstance(bound, (int, float, np.number)):
                ts_val = pd.to_datetime(bound, utc=True)
                converted.append((ts_val - min_ts).total_seconds() * scale_factor)
            else:
                converted.append(float(bound))
        parsed_time_range = _parse_numeric_bounds(converted, "time_range")

    # Synchronously drop rows containing NaN/Inf in time or any value column.
    finite_mask = np.ones(len(work), dtype=bool)
    for col in ["_kd_time_num"] + val_col_list:
        vals = work[col].to_numpy(dtype=np.float64, copy=False)
        finite_mask &= np.isfinite(vals)
    work = work.loc[finite_mask]

    # Apply value bounds synchronously across all columns.
    if col_val_bounds:
        if value_out_of_range == "clip":
            for col, (lo, hi) in col_val_bounds.items():
                if lo is not None or hi is not None:
                    work[col] = np.clip(work[col].to_numpy(dtype=np.float64), lo, hi)
        else:
            row_ok = np.ones(len(work), dtype=bool)
            for col, (lo, hi) in col_val_bounds.items():
                vals = work[col].to_numpy(dtype=np.float64, copy=False)
                if lo is not None:
                    row_ok &= vals >= lo
                if hi is not None:
                    row_ok &= vals <= hi

            if value_out_of_range == "drop_points":
                work = work.loc[row_ok]
            elif value_out_of_range == "drop_series":
                bad_ids = set(work.loc[~row_ok, id_col].unique())
                if bad_ids:
                    work = work.loc[~work[id_col].isin(bad_ids)]

    if len(work) == 0:
        raise ValueError("No valid time series remained after value filtering.")

    # Pass 1: Sort, deduplicate timestamps, and apply whole-series duration bounds.
    prepared_groups = []
    durations = []

    for uid, grp in work.groupby(id_col, sort=False):
        grp = grp.sort_values("_kd_time_num", kind="mergesort")
        grp = grp.drop_duplicates(subset=["_kd_time_num"], keep="last")
        t_raw = grp["_kd_time_num"].to_numpy(dtype=np.float64)

        if len(t_raw) < min_points or t_raw[-1] <= t_raw[0]:
            continue

        dur = float(t_raw[-1] - t_raw[0])
        if dur_bounds is not None:
            d_lo, d_hi = dur_bounds
            if (d_lo is not None and dur < d_lo) or (d_hi is not None and dur > d_hi):
                continue

        y_dict = {col: grp[col].to_numpy(dtype=np.float64) for col in val_col_list}
        attr_dict = {}
        for col in attr_col_list:
            non_null = grp[col].dropna()
            attr_dict[col] = non_null.iloc[0] if len(non_null) > 0 else None

        prepared_groups.append((uid, t_raw, dur, y_dict, attr_dict))
        durations.append(dur)

    if len(prepared_groups) == 0:
        raise ValueError("No valid time series remained after grouping and duration filtering.")

    if max_duration_quantile is not None:
        q_cap = float(np.quantile(np.asarray(durations, dtype=np.float64), float(max_duration_quantile)))
        prepared_groups = [g for g in prepared_groups if g[2] <= q_cap]

    # Pass 2: Align, apply time_range policy, and resample onto uniform grid.
    max_lines = len(prepared_groups)
    t_out = np.empty((max_lines, n_steps), dtype=np.float32)
    v_out = {col: np.empty((max_lines, n_steps), dtype=np.float32) for col in val_col_list}
    attrs_out: Dict[str, list] = {col: [] for col in attr_col_list}
    valid_ids = []
    valid_idx = 0

    for uid, t_raw, _, y_dict, attr_dict in prepared_groups:
        t_aligned = _align_1d(t_raw, align)

        # Apply time_range synchronously to t_aligned and all variable columns.
        if parsed_time_range is not None:
            t_lo, t_hi = parsed_time_range

            if time_out_of_range == "drop_series":
                if (t_lo is not None and t_aligned[0] < t_lo) or (t_hi is not None and t_aligned[-1] > t_hi):
                    continue
                t_sample = np.linspace(t_aligned[0], t_aligned[-1], n_steps, dtype=np.float64)

            elif time_out_of_range == "drop_points":
                t_mask = np.ones(len(t_aligned), dtype=bool)
                if t_lo is not None:
                    t_mask &= t_aligned >= t_lo
                if t_hi is not None:
                    t_mask &= t_aligned <= t_hi
                t_aligned = t_aligned[t_mask]
                if len(t_aligned) < min_points or t_aligned[-1] <= t_aligned[0]:
                    continue
                y_dict = {col: arr[t_mask] for col, arr in y_dict.items()}
                t_sample = np.linspace(t_aligned[0], t_aligned[-1], n_steps, dtype=np.float64)

            else:  # "slice"
                eff_lo = t_aligned[0] if t_lo is None else max(t_aligned[0], t_lo)
                eff_hi = t_aligned[-1] if t_hi is None else min(t_aligned[-1], t_hi)
                if eff_hi <= eff_lo:
                    continue
                t_sample = np.linspace(eff_lo, eff_hi, n_steps, dtype=np.float64)
        else:
            t_sample = np.linspace(t_aligned[0], t_aligned[-1], n_steps, dtype=np.float64)

        t_out[valid_idx] = t_sample.astype(np.float32)
        for col in val_col_list:
            interp_kind = kind.get(col, "linear") if isinstance(kind, dict) else kind
            v_out[col][valid_idx] = _resample_1d(
                t_aligned, y_dict[col], t_sample, interp_kind
            ).astype(np.float32)

        for col in attr_col_list:
            attrs_out[col].append(attr_dict[col])

        valid_ids.append(uid)
        valid_idx += 1

    if valid_idx == 0:
        raise ValueError("No valid time series remained after time filtering and resampling.")

    t_final = np.ascontiguousarray(t_out[:valid_idx], dtype=np.float32)
    if is_single_val:
        v_final: Union[np.ndarray, Dict[str, np.ndarray]] = np.ascontiguousarray(
            v_out[val_col_list[0]][:valid_idx], dtype=np.float32
        )
    else:
        v_final = {
            col: np.ascontiguousarray(v_out[col][:valid_idx], dtype=np.float32)
            for col in val_col_list
        }

    attrs_final = (
        {col: np.array(vals, dtype=object) for col, vals in attrs_out.items()}
        if attr_col_list
        else None
    )

    return PreparedTimeSeries(
        time_data=t_final,
        value_data=v_final,
        ids=np.array(valid_ids, dtype=object),
        attributes=attrs_final,
    )
