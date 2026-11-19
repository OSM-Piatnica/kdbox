"""Interactive Jupyter widgets for univariate and multivariate KD-Box exploration.

This module provides the Python-to-JavaScript bridge built on `anywidget`
and `ipywidgets`:

- `KDBoxWidget`: Single-variable interactive density viewer supporting
  Brush (Timebox), Radius Nearest Line (RNL), Angular, and Pin spatial queries,
  categorical attribute filtering, and real-time representative line extraction.
- `MultivariateKDBox`: Coordinated multi-view container that links
  multiple `KDBoxWidget` instances via cross-dimensional masking,
  synchronized attribute filters, and an auto-scaling Matplotlib overlay of
  representative lines with highlighted query time intervals.

Notes
-----
Cross-widget masks (`external_mask` and `attr_mask`) are deliberately kept
as plain Python attributes rather than synced `traitlets` to avoid ZMQ
WebSocket serialization overhead and NumPy boolean-array comparison errors.
"""

from __future__ import annotations

import base64
from functools import reduce
from io import BytesIO
import json
import math
from pathlib import Path
import traceback
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import anywidget
import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image
import traitlets

# Import the Backend Engine.
from .kd_tree import (
    angular_query,
    build_kd_tree,
    create_tvs_segments,
    get_representative_lines,
    rasterize_density,
    rnl_query,
    timebox_query
)


# =============================================================================
# Input Parsing, Validation and Formatting Helpers
# =============================================================================


def _unpack_single_axis_data(
    data: Any,
    explicit_label: str = "",
    arg_name: str = "data",
) -> Tuple[Any, str]:
    """Unpack either a raw array-like or a single-entry `{label: array}` dict.

    Parameters
    ----------
    data : array-like or dict[str, array-like]
        Raw 2D data array or single-key dictionary mapping an axis label to
        the data array.
    explicit_label : str, default=""
        Optional explicit label passed via keyword argument. Takes precedence
        over a dictionary key if both are non-empty.
    arg_name : str, default="data"
        Parameter name used in error messages.

    Returns
    -------
    array : Any
        The unpacked array-like object.
    label : str
        Resolved stripped label string (empty string if unlabeled).

    Raises
    ------
    ValueError
        If data is a dictionary with zero or more than one entry.
    """
    if isinstance(data, dict):
        if len(data) != 1:
            raise ValueError(
                f"{arg_name} dictionary must contain exactly one {{label: array}} pair."
            )
        key, val = next(iter(data.items()))
        extracted = str(key).strip() if key is not None else ""
        final_label = str(explicit_label).strip() if explicit_label else extracted
        return val, final_label
    return data, str(explicit_label or "").strip()


def _validate_and_prepare_arrays(
    time_data: Any,
    value_data: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and coerce inputs into contiguous 2D `float32` arrays.

    Ensures shape consistency, minimum length, and finiteness before passing
    buffers into Numba `@njit` routines where out-of-bounds reads or `NaN`
    values would cause undefined behavior or broken JSON serialization.

    Parameters
    ----------
    time_data, value_data : array-like
        Input matrices expected to have identical 2D shape `(n_lines, n_steps)`.

    Returns
    -------
    t_arr, v_arr : tuple of np.ndarray of float32
        C-contiguous 2D arrays verified to contain only finite values.

    Raises
    ------
    ValueError
        If arrays are not 2D, have mismatched shapes, have fewer than 1 line
        or 2 time steps, or contain `NaN` / `Inf` values.
    """
    t_arr = np.ascontiguousarray(time_data, dtype=np.float32)
    v_arr = np.ascontiguousarray(value_data, dtype=np.float32)

    if t_arr.ndim != 2 or v_arr.ndim != 2:
        raise ValueError(
            f"time_data and value_data must be 2D arrays of shape (n_lines, n_steps), "
            f"got shapes {t_arr.shape} and {v_arr.shape}."
        )
    if t_arr.shape != v_arr.shape:
        raise ValueError(
            f"time_data shape {t_arr.shape} does not match value_data shape {v_arr.shape}."
        )

    n_lines, n_steps = t_arr.shape
    if n_lines < 1:
        raise ValueError("Datasets must contain at least 1 time series (n_lines >= 1).")
    if n_steps < 2:
        raise ValueError("Each time series must contain at least 2 time steps (n_steps >= 2).")
    if not np.all(np.isfinite(t_arr)):
        raise ValueError("time_data contains NaN or Inf values. Clean or impute before plotting.")
    if not np.all(np.isfinite(v_arr)):
        raise ValueError("value_data contains NaN or Inf values. Clean or impute before plotting.")

    return t_arr, v_arr


def _compute_safe_bounds(arr: np.ndarray) -> Tuple[float, float]:
    """Compute `(min_val, max_val)` with symmetric padding for constant arrays.

    Prevents `ZeroDivisionError` during pixel-to-data scaling when a variable
    or time axis has zero variance.
    """
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        pad = 0.5 if lo == 0.0 else max(0.5, abs(lo) * 0.05)
        lo -= pad
        hi += pad
    return lo, hi


def _masks_equal(m1: Optional[np.ndarray], m2: Optional[np.ndarray]) -> bool:
    """Safely compare two optional 1D index masks without NumPy truth-value ambiguity."""
    if m1 is None and m2 is None:
        return True
    if m1 is None or m2 is None:
        return False
    return bool(np.array_equal(m1, m2))


def _dedupe_key(base_key: str, seen: Set[str]) -> str:
    """Return a unique key by appending `(2)`, `(3)`, etc. on collisions."""
    if base_key not in seen:
        seen.add(base_key)
        return base_key
    suffix = 2
    while f"{base_key} ({suffix})" in seen:
        suffix += 1
    unique_key = f"{base_key} ({suffix})"
    seen.add(unique_key)
    return unique_key


def _format_attr_val(val: Any) -> str:
    """Convert a scalar categorical value into a display-ready string.

    Groups `None` and `NaN` under `"<Missing>"` and strips trailing
    `.0` from integral floats (common in Pandas columns with nulls).
    """
    if val is None:
        return "<Missing>"
    try:
        # `val != val` catches float('nan'), np.nan, and pandas.NA without importing pandas.
        if val != val or (isinstance(val, (float, np.floating)) and np.isnan(val)):
            return "<Missing>"
    except Exception:
        pass
    if isinstance(val, (bool, np.bool_)):
        return str(bool(val))
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        if val.is_integer():
            return str(int(val))
    return str(val)


def _normalize_attributes(
    attributes: Any,
    n_lines: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[Dict[str, Any]]]]:
    """Normalize optional metadata into string arrays and frequency-sorted UI options.

    Parameters
    ----------
    attributes : dict, pandas.DataFrame, pandas.Series, 1D array-like, or None
        Per-series categorical metadata aligned with row indices `0..n_lines-1`.
    n_lines : int
        Expected number of series.

    Returns
    -------
    attr_arrays : dict[str, np.ndarray]
        Mapping `{attr_name: 1D object array of str}` used for fast `np.isin` masking.
    attr_options : dict[str, list[dict[str, Any]]]
        JSON-serializable UI metadata `{attr_name: [{"value": str, "count": int}, ...]}`
        sorted by descending frequency, then alphabetically.

    Raises
    ------
    ValueError
        If any attribute sequence length does not equal `n_lines`.
    """
    if attributes is None:
        return {}, {}

    raw_dict: Dict[str, List[Any]] = {}
    if isinstance(attributes, dict):
        for key, vals in attributes.items():
            raw_dict[str(key)] = list(vals)
    elif hasattr(attributes, "columns") and hasattr(attributes, "__getitem__"):
        # Duck-typed pandas.DataFrame without requiring a hard pandas import.
        for col in attributes.columns:
            raw_dict[str(col)] = list(attributes[col])
    elif hasattr(attributes, "name") and hasattr(attributes, "tolist"):
        # Duck-typed pandas.Series.
        name = str(attributes.name) if attributes.name is not None else "Attribute"
        raw_dict[name] = attributes.tolist()
    else:
        raw_dict["Attribute"] = list(attributes)

    attr_arrays: Dict[str, np.ndarray] = {}
    attr_options: Dict[str, List[Dict[str, Any]]] = {}

    for name, vals in raw_dict.items():
        if len(vals) != n_lines:
            raise ValueError(
                f"Attribute {name!r} has length {len(vals)}, expected n_lines={n_lines}."
            )
        formatted = np.array([_format_attr_val(v) for v in vals], dtype=object)
        attr_arrays[name] = formatted

        unique_vals, counts = np.unique(formatted, return_counts=True)
        # Primary sort: count descending; secondary tie-breaker: value ascending.
        order = np.lexsort((unique_vals.astype(str), -counts))
        attr_options[name] = [
            {"value": str(unique_vals[idx]), "count": int(counts[idx])}
            for idx in order
        ]

    return attr_arrays, attr_options


# =============================================================================
# Univariate Interactive Widget
# =============================================================================


class KDBoxWidget(anywidget.AnyWidget):
    """Interactive Jupyter widget for exploring large-scale univariate time series.

    Renders a Numba-rasterized 2D density field onto an HTML5 Canvas with
    real-time Brush (Timebox), RNL (Radius), Angular, and Pin queries,
    categorical attribute filtering, and representative line overlays.

    Parameters
    ----------
    time_data : array-like or dict[str, array-like]
        2D timestamp matrix of shape `(n_lines, n_steps)`, or a single-key
        dict `{x_label: array}`.
    value_data : array-like or dict[str, array-like]
        2D measurement matrix matching `time_data.shape`, or a single-key
        dict `{y_label: array}`.
    attributes : dict, pandas.DataFrame, pandas.Series, 1D array-like, optional
        Static per-series categorical metadata enabling interactive attribute
        filtering (OR within an attribute, AND across attributes).
    x_label : str, default=""
        Optional X-axis title displayed below the canvas. Overrides any key
        provided via `time_data`.
    y_label : str, default=""
        Optional Y-axis title and plot header badge. Overrides any key
        provided via `value_data`.
    **kwargs
        Additional `traitlets` keyword arguments forwarded to
        :class:`anywidget.AnyWidget` (e.g., `width`, `height`,
        `colormap`, `rep_count`, `diversity`, `rep_method`,
        `strict_mode`, `show_attr_controls`).

    Examples
    --------
    >>> w = KDBoxWidget({"Time (s)": t_arr}, {"Temperature": temp_arr})
    >>> w  # Display in Jupyter cell
    >>> sel_t, sel_v = w.get_selected_data()
    """

    # Load frontend ESM relative to this module so installed packages locate
    # widget.js regardless of the notebook's working directory.
    _esm = Path(__file__).resolve().parent / "widget.js"

    # UI dimensions, data bounds, and optional axis labels.
    width = traitlets.Int(1000).tag(sync=True)
    height = traitlets.Int(500).tag(sync=True)
    bounds = traitlets.Dict().tag(sync=True)
    x_label = traitlets.Unicode("").tag(sync=True)
    y_label = traitlets.Unicode("").tag(sync=True)

    # Render outputs and active query/exploration state synced with JS.
    density_image = traitlets.Unicode("").tag(sync=True)
    query_data = traitlets.Unicode("[]").tag(sync=True)
    # Allowed: "global", "attribute", "coordinated", "union", "intersection", or query ID.
    explore_mode = traitlets.Unicode("global").tag(sync=True)

    # Categorical attribute filtering metadata and active selections.
    attr_options = traitlets.Dict(default_value={}).tag(sync=True)
    attr_filters = traitlets.Unicode("{}").tag(sync=True)
    show_attr_controls = traitlets.Bool(True).tag(sync=True)
    
    rep_lines_data = traitlets.List(default_value=[]).tag(sync=True)
    stats_data = traitlets.Dict(default_value={}).tag(sync=True)

    # Interactive control panel parameters.
    colormap = traitlets.Unicode("magma").tag(sync=True)
    tool_mode = traitlets.Unicode("brush").tag(sync=True)
    rep_method = traitlets.Unicode("heuristic").tag(sync=True)
    rep_count = traitlets.Int(3).tag(sync=True)
    diversity = traitlets.Float(0.1).tag(sync=True)    
    normalize_density = traitlets.Bool(True).tag(sync=True)
    strict_mode = traitlets.Bool(False).tag(sync=True)

    # Monotonic counter incremented after local state settles so MultivariateKDBox
    # coordinates views in strict sequence.
    _coordination_trigger = traitlets.Int(0).tag(sync=True)
    _initialized: bool = False

    def __init__(
        self,
        time_data: Any,
        value_data: Any,
        attributes: Any = None,
        x_label: str = "",
        y_label: str = "",
        **kwargs: Any,
    ) -> None:
        raw_t, resolved_x_label = _unpack_single_axis_data(time_data, x_label, arg_name="time_data")
        raw_v, resolved_y_label = _unpack_single_axis_data(value_data, y_label, arg_name="value_data")
        t_arr, v_arr = _validate_and_prepare_arrays(raw_t, raw_v)
        n_lines, n_steps = t_arr.shape
        attr_arrays, parsed_attr_options = _normalize_attributes(attributes, n_lines)

        super().__init__(
            x_label=resolved_x_label,
            y_label=resolved_y_label,
            attr_options=parsed_attr_options,
            **kwargs
        )
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Widget dimensions must be positive, got width={self.width}, height={self.height}."
            )

        self.time_data: np.ndarray = t_arr
        self.value_data: np.ndarray = v_arr
        self.n_lines: int = n_lines
        self.n_steps: int = n_steps
        self.attr_arrays: Dict[str, np.ndarray] = attr_arrays
        self.attr_mask: Optional[np.ndarray] = None
        self.external_mask: Optional[np.ndarray] = None
        self._current_rep_lines: np.ndarray = np.array([], dtype=np.int32)

        # Compute global bounds (with safe padding for constant series).
        self.t_min, self.t_max = _compute_safe_bounds(self.time_data)
        self.v_min, self.v_max = _compute_safe_bounds(self.value_data)
        self.t_span: float = max(1e-5, self.t_max - self.t_min)
        self.v_span: float = max(1e-5, self.v_max - self.v_min)
        self.bounds = {
            "t_min": self.t_min, "t_max": self.t_max, 
            "v_min": self.v_min, "v_max": self.v_max
        }

        # Build the Numba spatial index once at initialization.
        self.segments = create_tvs_segments(self.time_data, self.value_data)
        self.tree_arrays = build_kd_tree(self.segments, max_leaf_size=32, max_depth=15)

        # Pre-compute global max pixel density for absolute (non-normalized) rendering.
        all_lines = np.arange(self.n_lines, dtype=np.int32)
        global_grid = rasterize_density(
            self.segments, all_lines, 
            self.width, self.height, 
            self.t_min, self.t_max, self.v_min, self.v_max, 
            self.n_steps
        )
        self.global_max_density: int = max(1, int(np.max(global_grid)))

        # Local query result caches.
        self.all_lines: np.ndarray = all_lines
        self.query_results: Dict[str, np.ndarray] = {}
        self.query_time_spans: Dict[str, Tuple[float, float]] = {}
        self.union_lines: np.ndarray = np.array([], dtype=np.int32)
        self.intersect_lines: np.ndarray = np.array([], dtype=np.int32)

        if self.attr_filters and self.attr_filters != "{}":
            self._recompute_attr_mask(self.attr_filters)
            if self.attr_mask is not None and self.explore_mode == "global":
                self.explore_mode = "attribute"

        self._initialized = True
        self._execute_queries(self.query_data)

    # -------------------------------------------------------------------------
    # Masking and Coordination Helpers
    # -------------------------------------------------------------------------

    def _recompute_attr_mask(self, f_str: str) -> None:
        """Recompute `self.attr_mask` using OR within attributes and AND across attributes."""
        try:
            f_dict = json.loads(f_str) if f_str else {}
        except Exception:
            f_dict = {}

        combined = np.ones(self.n_lines, dtype=bool)
        any_active = False
        for attr_name, selected_vals in f_dict.items():
            if selected_vals and attr_name in self.attr_arrays:
                combined &= np.isin(self.attr_arrays[attr_name], selected_vals)
                any_active = True

        self.attr_mask = np.where(combined)[0].astype(np.int32) if any_active else None

    @traitlets.observe("attr_filters")
    def _on_attr_filters_changed(self, change: Dict[str, Any]) -> None:
        if not self._initialized:
            return
        prev_had_attr = self.attr_mask is not None
        self._recompute_attr_mask(change["new"])

        # Auto-focus Attribute Filter when first enabled; fall back cleanly when cleared.
        if self.attr_mask is not None and not prev_had_attr and self.explore_mode == "global":
            self.explore_mode = "attribute"
        elif self.attr_mask is None and self.explore_mode == "attribute":
            self.explore_mode = "coordinated" if self.external_mask is not None else "global"
        else:
            self._update_stats_and_render()
            self._coordination_trigger += 1

    def set_external_mask(self, mask: Optional[np.ndarray]) -> None:
        """Apply a cross-widget line-index mask from `MultivariateKDBox`.

        Skips re-rasterization when `mask` equals the current `external_mask`.
        """
        if _masks_equal(self.external_mask, mask):
            return

        prev_was_none = self.external_mask is None
        self.external_mask = mask

        # Automatically switch between "global" and "coordinated" when no local query is active.
        if mask is not None and prev_was_none and self.explore_mode in ["global", "attribute"]:
            self.explore_mode = "coordinated"
        elif mask is None and self.explore_mode == "coordinated":
            self.explore_mode = "attribute" if self.attr_mask is not None else "global"
        else:
            self._update_stats_and_render()

    def has_active_local_filter(self) -> bool:
        """Return `True` if a local spatial query is currently explored."""
        return len(self.query_results) > 0 and self.explore_mode not in [
            "global", "attribute", "coordinated"
        ]

    def get_active_time_intervals(self) -> List[Tuple[float, float]]:
        """Return `(t_min, t_max)` spans for currently explored local queries."""
        if not self.has_active_local_filter():
            return []
        if self.explore_mode in ["union", "intersection"]:
            return list(self.query_time_spans.values())
        span = self.query_time_spans.get(self.explore_mode)
        return [span] if span is not None else []

    # -------------------------------------------------------------------------
    # Public Data and Attribute Export API
    # -------------------------------------------------------------------------

    def get_selected_indices(self) -> np.ndarray:
        """Return a 1D `int32` copy of currently active series row indices."""
        return self._get_effective_active_lines().copy()

    def get_representative_indices(self) -> np.ndarray:
        """Return a 1D `int32` copy of current representative series row indices."""
        return self._current_rep_lines.copy()

    def get_selected_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return `(time_data, value_data)` arrays for the active selection."""
        active_lines = self.get_selected_indices()
        return self.time_data[active_lines], self.value_data[active_lines]

    def get_representative_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return `(time_data, value_data)` arrays for current representative lines."""
        rep_lines = self.get_representative_indices()
        return self.time_data[rep_lines], self.value_data[rep_lines]

    def get_selected_attributes(self) -> Dict[str, np.ndarray]:
        """Return `{attr_name: array}` metadata for the currently active series."""
        active_lines = self.get_selected_indices()
        return {k: arr[active_lines] for k, arr in self.attr_arrays.items()}

    def get_representative_attributes(self) -> Dict[str, np.ndarray]:
        """Return `{attr_name: array}` metadata for current representative series."""
        rep_lines = self.get_representative_indices()
        return {k: arr[rep_lines] for k, arr in self.attr_arrays.items()}

    # -------------------------------------------------------------------------
    # Query Execution and Rendering Pipeline
    # -------------------------------------------------------------------------

    def _pixels_to_data(self, px: float, py: float) -> Tuple[float, float]:
        """Convert top-down HTML5 Canvas pixel coordinates into data `(t, v)`."""
        t_scale = self.t_span / self.width
        v_scale = self.v_span / self.height
        return self.t_min + (px * t_scale), self.v_max - (py * v_scale)

    def _calc_stats(self, lines: np.ndarray) -> Dict[str, Union[int, float]]:
        """Compute summary statistics displayed in the Information Table."""
        if len(lines) == 0:
            return {"count": 0, "points": 0, "min": 0, "max": 0, "mean": 0, "var": 0}
        vals = self.value_data[lines]
        return {
            "count": int(len(lines)),
            "points": int(len(lines) * self.n_steps),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "var": float(np.var(vals))
        }

    @traitlets.observe("query_data", "strict_mode")
    def _on_query_changed(self, change: Dict[str, Any]) -> None:
        if not self._initialized:
            return
        self._execute_queries(self.query_data)

    def _execute_queries(self, q_str: str) -> None:
        """Parse JSON query shapes from JS and evaluate them against the KD-tree."""
        try:
            queries = json.loads(q_str) if q_str else []
            self.query_results = {}
            self.query_time_spans = {}
            
            t_scale = self.width / self.t_span
            v_scale = self.height / self.v_span
            is_strict_brush = bool(self.strict_mode)
            
            for q in queries:
                qid = q["id"]
                active_lines = np.array([], dtype=np.int32)
                
                if q["type"] == "brush":
                    q_t_min, _ = self._pixels_to_data(q["x_min"], 0)
                    q_t_max, _ = self._pixels_to_data(q["x_min"] + q["w"], 0)
                    _, q_v_max = self._pixels_to_data(0, q["y_min"]) 
                    _, q_v_min = self._pixels_to_data(0, q["y_min"] + q["h"])
                    active_lines = timebox_query(
                        self.tree_arrays, self.segments,
                        q_t_min, q_t_max, 
                        q_v_min, q_v_max, 
                        self.n_lines,
                        strict=is_strict_brush
                        )
                    self.query_time_spans[qid] = (q_t_min, q_t_max)

                elif q["type"] in ("rnl", "pin"):
                    q_t, q_v = self._pixels_to_data(q["cx"], q["cy"])
                    # Pin queries reuse the RNL kernel with a tight 3px probe radius.
                    radius_px = 3.0 if q["type"] == "pin" else float(q["r"])
                    active_lines = rnl_query(
                        self.tree_arrays, self.segments,
                        q_t, q_v, radius_px, t_scale, v_scale, self.n_lines,
                    )
                    dt = radius_px / t_scale
                    self.query_time_spans[qid] = (max(self.t_min, q_t - dt), min(self.t_max, q_t + dt))

                elif q["type"] == "ang":
                    t1, _ = self._pixels_to_data(q["sx"], q["sy"])
                    t2, _ = self._pixels_to_data(q["ex"], q["ey"])
                    
                    dx, dy = q["ex"] - q["sx"], q["ey"] - q["sy"] 
                    if abs(dx) < 1e-9:
                        dx = 1e-9
                    # Normalize vector so slope angle is always measured forward in time.
                    if dx < 0:
                        dx, dy = -dx, -dy
                    target_angle_rad = math.atan(dy / dx)
                    tol_deg = float(q.get("tol_deg", 15.0))
                    
                    q_t_min, q_t_max = min(t1, t2), max(t1, t2)
                    # Angular queries evaluate net interval trend (strict_mode applies only to Brush).
                    active_lines = angular_query(
                        self.tree_arrays, self.segments,
                        q_t_min, q_t_max,
                        target_angle_rad, math.radians(tol_deg),
                        t_scale, v_scale,
                        self.n_lines,
                        strict=False
                        )
                    self.query_time_spans[qid] = (q_t_min, q_t_max)
                    
                self.query_results[qid] = active_lines

            # Compute sets.
            if len(self.query_results) > 0:
                list_of_lines = list(self.query_results.values())
                self.union_lines = reduce(np.union1d, list_of_lines)
                self.intersect_lines = reduce(np.intersect1d, list_of_lines)
            else:
                self.union_lines = np.array([], dtype=np.int32)
                self.intersect_lines = np.array([], dtype=np.int32)
                
            # Signal the coordinator that local state is securely updated.
            self._update_stats_and_render()
            self._coordination_trigger += 1

        except Exception as exc:
            print(f"Widget Query Error: {exc}")
            traceback.print_exc()
            
    @traitlets.observe("rep_method", "rep_count", "diversity", "normalize_density", "explore_mode")
    def _on_params_changed(self, change: Dict[str, Any]) -> None:
        if not self._initialized:
            return
        self._update_stats_and_render()
        # Observe so the slider instantly updates the Matplotlib plot.
        if change["name"] in ["explore_mode", "rep_method", "rep_count", "diversity"]:
            self._coordination_trigger += 1

    def _get_local_active_lines(self) -> np.ndarray:
        """Return line indices matched solely by this widget's active spatial mode."""
        if self.explore_mode in ["global", "attribute", "coordinated"]:
            return self.all_lines
        if self.explore_mode == "union":
            return self.union_lines
        if self.explore_mode == "intersection":
            return self.intersect_lines
        return self.query_results.get(self.explore_mode, self.all_lines)

    def _get_combined_constraint_mask(self) -> np.ndarray:
        """Intersect active attribute and external coordination masks."""
        if self.attr_mask is not None and self.external_mask is not None:
            return np.intersect1d(self.attr_mask, self.external_mask)
        if self.attr_mask is not None:
            return self.attr_mask
        if self.external_mask is not None:
            return self.external_mask
        return self.all_lines

    def _get_effective_active_lines(self) -> np.ndarray:
        """Compute the final displayed line indices for the current `explore_mode`."""
        if self.explore_mode == "global":
            return self.all_lines
        if self.explore_mode == "attribute":
            return self.attr_mask if self.attr_mask is not None else self.all_lines
        if self.explore_mode == "coordinated":
            if self.external_mask is not None:
                return (
                    np.intersect1d(self.external_mask, self.attr_mask)
                    if self.attr_mask is not None
                    else self.external_mask
                )
            return self.attr_mask if self.attr_mask is not None else self.all_lines

        local_lines = self._get_local_active_lines()
        return np.intersect1d(local_lines, self._get_combined_constraint_mask())

    def _update_stats_and_render(self) -> None:
        """Recompute table statistics, grayscale density PNG, and representative lines."""
        try:
            constraint_mask = self._get_combined_constraint_mask()
            active_lines = self._get_effective_active_lines()

            stats: Dict[str, Any] = {
                "global": self._calc_stats(self.all_lines),
                "queries": {
                    qid: self._calc_stats(np.intersect1d(lines, constraint_mask))
                    for qid, lines in self.query_results.items()
                },
            }
            
            # Apply masks.
            if self.attr_mask is not None:
                stats["attribute"] = self._calc_stats(self.attr_mask)
            if self.external_mask is not None:
                coord_lines = (
                    np.intersect1d(self.external_mask, self.attr_mask)
                    if self.attr_mask is not None
                    else self.external_mask
                )
                stats["coordinated"] = self._calc_stats(coord_lines)

            if len(self.query_results) > 1:
                stats["union"] = self._calc_stats(np.intersect1d(self.union_lines, constraint_mask))
                stats["intersection"] = self._calc_stats(
                    np.intersect1d(self.intersect_lines, constraint_mask)
                )

            self.stats_data = stats
            
            if len(active_lines) == 0:
                self.density_image = ""
                self.rep_lines_data = []
                self._current_rep_lines = np.array([], dtype=np.int32)
                return

            # Calculate Density Grid.
            grid = rasterize_density(
                self.segments, active_lines, 
                self.width, self.height, 
                self.t_min, self.t_max, self.v_min, self.v_max, 
                self.n_steps
            )

            # Apply Normalization Logic.
            if self.normalize_density:
                max_density = np.max(grid)
            else:
                max_density = self.global_max_density
                
            if max_density == 0: 
                max_density = 1
            
            # Clip at 255 to prevent overflow when global scaling is active,
            # but selection is denser.            
            # Encode an 8-bit grayscale mask; colormapping, gamma, and opacity run
            # client-side on Canvas for 60fps slider responsiveness.
            img_array = np.clip((grid / max_density) * 255, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode="L")
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            self.density_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            # Extract Representative Lines.
            reps = get_representative_lines(
                self.segments, active_lines, grid,
                self.width, self.height,
                self.t_min, self.t_max, self.v_min, self.v_max,
                self.n_steps,
                num_reps=max(1, int(self.rep_count)),
                diversity_threshold=float(self.diversity),
                use_meanshift=(self.rep_method == "meanshift")
            )

            # Save for export.
            self._current_rep_lines = reps

            # Project representative curves back into HTML5 Canvas pixel coordinates.
            t_scale = self.width / self.t_span
            v_scale = self.height / self.v_span
            
            rep_data = []
            for rep_idx in reps:
                px = (self.time_data[rep_idx] - self.t_min) * t_scale
                py = self.height - ((self.value_data[rep_idx] - self.v_min) * v_scale)
                rep_data.append(np.column_stack((px, py)).tolist())

            self.rep_lines_data = rep_data

        except Exception as exc:
            print(f"Widget State Update Error: {exc}")
            traceback.print_exc()

# =============================================================================
# The Coordinated Multivariate Wrapper
# =============================================================================

class MultivariateKDBox(widgets.VBox):
    """Coordinated multi-view wrapper linking multiple `KDBoxWidget` views.

    Row `i` across all variables must correspond to the same entity (e.g.,
    sensor or trip). Brushing or filtering in any child widget automatically
    intersects matching entity indices across all sibling views and updates a
    shared Matplotlib overlay of each variable's representative lines.

    Parameters
    ----------
    time_data : array-like or dict[str, array-like]
        Shared 2D timestamp matrix `(n_lines, n_steps)`, or a single-key dict
        `{x_label: array}`.
    value_data : dict[str, array-like] or sequence of array-like
        Either a labeled mapping `{var_name: array}` or an unlabeled sequence
        of 2D arrays (assigned overlay names `var1`, `var2`, ... unless
        `y_labels` is supplied).
    attributes : dict, pandas.DataFrame, pandas.Series, 1D array-like, optional
        Shared per-series categorical metadata synchronized across all views.
    x_label : str, default=""
        Optional X-axis title displayed on the overlay plot and child widgets.
    y_labels : sequence of str, optional
        Optional variable labels when `value_data` is passed as a sequence.
    show_all_attr_controls : bool, default=False
        When `False`, only the first child widget displays the Attribute
        Filters sidebar (still synchronized across all views) to keep stacked
        layouts compact. Set `True` to show attribute controls on every child.
    **kwargs
        Additional layout keyword arguments forwarded to `ipywidgets.VBox`.

    Examples
    --------
    >>> multi = MultivariateKDBox(
    ...     {"Time (s)": time_data},
    ...     {"Fuel Level": level, "Pump": pump},
    ...     attributes={"Supplier": supplier},
    ... )
    >>> multi  # Display in Jupyter cell
    >>> idx = multi.get_selected_indices()
    """

    def __init__(
        self,
        time_data: Any,
        value_data: Union[Dict[str, Any], Sequence[Any]],
        attributes: Any = None,
        x_label: str = "",
        y_labels: Optional[Sequence[str]] = None,
        show_all_attr_controls: bool = False,
        **kwargs: Any
    ) -> None:
        self.out = widgets.Output(
            layout=widgets.Layout(display="flex", justify_content="center")
        )
        raw_t, resolved_x_label = _unpack_single_axis_data(time_data, x_label, arg_name="time_data")
        self.time_data: np.ndarray = np.ascontiguousarray(raw_t, dtype=np.float32)
        self.x_label: str = resolved_x_label
        self.kd_widgets: Dict[str, KDBoxWidget] = {}
        self.widget_display_labels: Dict[str, str] = {}
        self._is_syncing: bool = False

        # Parse labeled dict vs. unlabeled sequence, guaranteeing unique keys.
        entries: List[Tuple[str, str, Any]] = []
        seen_keys: Set[str] = set()
        if isinstance(value_data, dict):
            for i, (key, v_arr) in enumerate(value_data.items()):
                clean_label = str(key).strip() if key is not None else ""
                base_key = clean_label if clean_label else f"var{i + 1}"
                entries.append((_dedupe_key(base_key, seen_keys), clean_label, v_arr))
        else:
            for i, v_arr in enumerate(value_data):
                user_label = ""
                if y_labels is not None and i < len(y_labels) and y_labels[i]:
                    user_label = str(y_labels[i]).strip()
                base_key = user_label if user_label else f"var{i + 1}"
                entries.append((_dedupe_key(base_key, seen_keys), user_label, v_arr))

        if len(entries) == 0:
            raise ValueError("value_data must contain at least one variable array.")
        
        for i, (plot_key, widget_y_label, v_data) in enumerate(entries):
            child = KDBoxWidget(
                self.time_data,
                v_data,
                attributes=attributes,
                x_label=self.x_label,
                y_label=widget_y_label,
                show_attr_controls=(show_all_attr_controls or i == 0),
            )
            self.kd_widgets[plot_key] = child
            self.widget_display_labels[plot_key] = widget_y_label
            # Observe the sequencing trigger (instead of the raw query data) to
            # sync the masking constraints.   
            child.observe(self._on_child_query_change, names=["_coordination_trigger"])

        first_w = next(iter(self.kd_widgets.values()))
        self.t_min: float = first_w.t_min
        self.t_max: float = first_w.t_max

        super().__init__([self.out] + list(self.kd_widgets.values()), **kwargs)
        self._update_overlay_plot()

    # -------------------------------------------------------------------------
    # Top-Level Multivariate Export API
    # -------------------------------------------------------------------------

    def get_selected_indices(self) -> np.ndarray:
        """Return 1D `int32` indices matching the coordinated selection across all views."""
        active_sets = [w._get_effective_active_lines() for w in self.kd_widgets.values()]
        return reduce(np.intersect1d, active_sets).copy()

    def get_selected_data(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Return `(time_data, {var_label: value_data})` for the coordinated selection."""
        idx = self.get_selected_indices()
        return self.time_data[idx], {
            label: w.value_data[idx] for label, w in self.kd_widgets.items()
        }

    def get_selected_attributes(self) -> Dict[str, np.ndarray]:
        """Return `{attr_name: array}` metadata for the coordinated selection."""
        idx = self.get_selected_indices()
        first_w = next(iter(self.kd_widgets.values()))
        return {k: arr[idx] for k, arr in first_w.attr_arrays.items()}

    def get_representative_indices(self) -> Dict[str, np.ndarray]:
        """Return `{var_label: int32 array}` of representative indices per variable."""
        return {label: w.get_representative_indices() for label, w in self.kd_widgets.items()}

    def get_representative_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Return `{var_label: (time_data, value_data)}` for each variable's rep lines."""
        return {label: w.get_representative_data() for label, w in self.kd_widgets.items()}

    def get_representative_attributes(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Return `{var_label: {attr_name: array}}` for each variable's rep lines."""
        return {label: w.get_representative_attributes() for label, w in self.kd_widgets.items()}

    # -------------------------------------------------------------------------
    # Multi-View Synchronization and Overlay Rendering
    # -------------------------------------------------------------------------

    def _on_child_query_change(self, change: Dict[str, Any]) -> None:
        """Propagate attribute filters and cross-widget spatial masks without re-entry."""
        if self._is_syncing:
            return
        self._is_syncing = True
        try:
            # Synchronize attribute filters across all child widgets.
            source_widget = change.get("owner")
            if source_widget is not None:
                src_attr_filters = source_widget.attr_filters
                for child in self.kd_widgets.values():
                    if child is not source_widget and child.attr_filters != src_attr_filters:
                        child.attr_filters = src_attr_filters

            # Intersect active spatial filters from all other views for each target widget.
            for target_label, target_w in self.kd_widgets.items():
                other_masks = []
                for label, child in self.kd_widgets.items():
                    if label != target_label and child.has_active_local_filter():
                        local_lines = child._get_local_active_lines()
                        if child.attr_mask is not None:
                            local_lines = np.intersect1d(local_lines, child.attr_mask)
                        other_masks.append(local_lines)

                # Apply the mask.
                if len(other_masks) > 0:
                    target_w.set_external_mask(reduce(np.intersect1d, other_masks))
                else:
                    target_w.set_external_mask(None)

            # Explicitly command Matplotlib overlay to redraw only after all
            # external_masks have been applied and internal states have settled.
            self._update_overlay_plot()

        finally:
            self._is_syncing = False

    def _update_overlay_plot(self, change: Optional[Dict[str, Any]] = None) -> None:
        """Render the synchronized Matplotlib overlay of representative lines."""
        self.out.clear_output(wait=True)
        with self.out:
            # Create a compact, wide plot for the header.
            n_vars = len(self.kd_widgets)
            fig, ax1 = plt.subplots(figsize=(10, 3.2))
            colors = plt.cm.tab10.colors
            
            if n_vars > 1:
                ax1.spines["right"].set_visible(False)

            ax1.set_xlim(self.t_min, self.t_max)

            for i, (plot_label, child) in enumerate(self.kd_widgets.items()):
                color = colors[i % len(colors)]
                t_data, v_data = child.get_representative_data()
                
                # Handle axis positioning to prevent overlap.
                if i == 0:
                    ax = ax1
                    ax.spines["left"].set_color(color)
                    ax.spines["left"].set_linewidth(1.5)
                    if n_vars <= 2:
                        ax.set_ylabel(plot_label, color=color, fontweight="bold")
                else:
                    ax = ax1.twinx()
                    # Alternate outward-offset Y-axes across right/left sides so 3+
                    # variables stay legible without overlapping ticks or labels.
                    side = "right" if i % 2 == 1 else "left"
                    offset = 45 * ((i - 1) // 2) if side == "right" else 45 * (i // 2)

                    ax.yaxis.set_label_position(side)
                    ax.yaxis.set_ticks_position(side)
                    ax.spines["right"].set_visible(side == "right")
                    ax.spines["left"].set_visible(side == "left")
                    ax.spines[side].set_position(("outward", offset))
                    ax.spines[side].set_color(color)
                    ax.spines[side].set_linewidth(1.5)

                    if n_vars <= 2:
                        ax.set_ylabel(
                            plot_label, color=color, fontweight="bold", rotation=270, labelpad=15
                        )

                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.tick_params(axis='y', colors=color, labelsize=9)

                # Shade active query time intervals from this variable for temporal context.
                for t_span_min, t_span_max in child.get_active_time_intervals():
                    ax1.axvspan(t_span_min, t_span_max, color=color, alpha=0.10)
                    ax1.axvline(t_span_min, color=color, linestyle="--", alpha=0.45, linewidth=1)
                    ax1.axvline(t_span_max, color=color, linestyle="--", alpha=0.45, linewidth=1)

                # Plot the Representative Lines.
                if len(t_data) > 0:
                    v_min_rep, v_max_rep = np.min(v_data), np.max(v_data)
                    legend_label = f"{plot_label} [{v_min_rep:.1f}, {v_max_rep:.1f}]"
                    for line_t, line_v in zip(t_data, v_data):
                        ax.plot(line_t, line_v, color=color, alpha=0.8, linewidth=2, label=legend_label)
                else:
                    ax.set_ylim(child.v_min, child.v_max)

            # Unify legend entries across all twin axes.
            handles, labels = [], []
            for ax in fig.axes:
                h_list, l_list = ax.get_legend_handles_labels()
                for hd, lb in zip(h_list, l_list):
                    if lb not in labels:
                        handles.append(hd)
                        labels.append(lb)
                        
            if handles:
                # Place legend below the plot, centered, in multiple columns.
                ax1.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.25),
                    ncol=min(3, len(labels)),
                    frameon=True
                )
            
            if self.x_label:
                ax1.set_xlabel(self.x_label)
            plt.title("Coordinated Multivariate Representative Lines")
            
            plt.tight_layout()
            if plt.get_backend().lower() != "agg":
                plt.show()
            # Explicitly close the figure to avoid accumulating Matplotlib memory.
            plt.close(fig)
