# KD-Box User Guide

This guide covers the complete workflow for exploring univariate and multivariate time-series datasets in JupyterLab / Jupyter Notebook with `kdbox` --- from preprocessing raw DataFrames to interpreting every visual cue and exporting filtered selections back into Python.

---

## 1. Preparing Your Data

The spatial engine operates on 2D numeric matrices of shape `(n_lines, n_steps)` where every row is one time series (entity) and every column is a sample point. You can either pass NumPy arrays directly or convert a long-format Pandas DataFrame with `from_dataframe`.

### 1.1 From a Long-Format Pandas DataFrame (`from_dataframe`)

Real-world telemetry tables often contain irregular sampling intervals, variable series durations, datetime strings, missing values, and sensor glitches. `from_dataframe` handles cleaning, bounding, alignment, and resampling in a single pass:

```python
import pandas as pd
from kdbox import from_dataframe

prep = from_dataframe(
    df,
    id_col="unique_id",          # Column identifying each time series
    time_col="timestamp",        # Numeric or datetime/ISO-string column
    value_cols=["level", "pump"],# Single column name (str) or list of columns
    attr_cols=["supplier_id"],   # Optional static per-series metadata columns
    n_steps=100,                 # Uniform resampled length per series (>= 2)
    align="start",               # "start" | "end" | "center" | "normalize" | None
    kind={"level": "linear", "pump": "previous"},
    time_unit="s",               # Unit for datetime conversion ("ns","us","ms","s","m","h","d")
    min_points=2,                # Drop series with fewer unique valid timestamps
)
```

`from_dataframe` returns a `PreparedTimeSeries` dataclass with four fields:
* `prep.time_data`  ---  Contiguous `float32` array of shape `(n_lines, n_steps)`.
* `prep.value_data`  ---  A single `float32` array (if `value_cols` was a `str`) or a dict `{col_name: array}` (if `value_cols` was a list).
* `prep.ids`  ---  1D array of original `id_col` values aligned row-for-row with the arrays (used to map widget selections back to your source DataFrame).
* `prep.attributes`  ---  Dict `{attr_col: 1D array}` of static metadata (the first non-null value per series), or `None` if `attr_cols` was omitted.

#### Interpolation Modes (`kind`)
Pass a single string for all columns or a per-column dictionary:
* `"linear"` *(default)*  ---  Continuous linear interpolation (best for smooth physical measurements like temperature, pressure, or fuel level).
* `"previous"` / `"next"` / `"nearest"`  ---  Stepwise hold / nearest-neighbor sampling (best for discrete states, flags, or pump status without introducing artificial intermediate values).
* Higher-order SciPy splines (e.g., `"cubic"`, `"quadratic"`) are supported when `scipy` is installed (`kdbox[data]`).

#### Time, Duration, and Value Bounding
Because `kdbox` renders into a fixed global bounding box, extreme outliers (such as a single corrupted sensor spike or a handful of unusually long runs) can compress the rest of the density field. `from_dataframe` provides synchronized bounding options --- **whenever a row or series is dropped by any constraint, it is dropped identically across `time_data`, every column in `value_data`, `ids`, and `attributes`**:

| Parameter | Options / Format | Behavior |
| :--- | :--- | :--- |
| `duration_range` | `(min_dur, max_dur)` | Drops entire series whose total duration (in `time_unit`, measured *before* alignment) falls outside the range. Either side may be `None`. |
| `max_duration_quantile` | `float` in `(0, 1]` | Drops entire series longer than the empirical quantile (e.g., `0.95` trims the longest 5% of runs without hardcoding a cutoff). |
| `time_range` | `(t_min, t_max)` | Bounds evaluated in the **aligned** time frame (e.g., `(0, 300)` with `align="start"` means the first 300s; `(-300, 0)` with `align="end"` means the final 300s). Accepts datetime strings/`Timestamp` when `align=None` on datetime columns. |
| `time_out_of_range` | `"slice"` *(default)*, `"drop_points"`, `"drop_series"` | **`"slice"`** truncates each series to `time_range` and interpolates exact boundary endpoints so all `n_steps` fall inside the window; **`"drop_points"`** drops raw rows outside `time_range` before resampling; **`"drop_series"`** discards any series extending outside `time_range`. |
| `value_range` | `(v_min, v_max)` or `{col: (v_min, v_max)}` | Value bounds applied across all value columns or selected columns only. Either bound may be `None`. |
| `value_out_of_range` | `"clip"` *(default)*, `"drop_points"`, `"drop_series"` | **`"clip"`** clamps bounded columns in place; **`"drop_points"`** drops any timestamp row where *any* bounded column is out of range across all variables (bridging isolated glitches); **`"drop_series"`** discards the entire series across all variables if *any* bounded column ever exceeds its range. |

### 1.2 Aligning Pre-Built Time Matrices (`align_time_series`)

If you already have a 2D `time_data` NumPy array, use `align_time_series(time_data, align=...)` directly:
* `"start"`  ---  Shifts every series to begin at $t = 0$.
* `"end"`  ---  Shifts every series to end at $t = 0$ (useful for comparing behavior leading up to a failure or completion event).
* `"center"`  ---  Centers every series around its temporal midpoint ($t = 0$).
* `"normalize"`  ---  Rescales every series linearly onto $[0.0, 1.0]$ to compare shapes independently of duration.

### 1.3 Direct NumPy Array Requirements
When bypassing `from_dataframe`, your arrays must satisfy:
* Both `time_data` and every `value_data` array are 2D with identical shape `(n_lines, n_steps)`, `n_lines >= 1`, and `n_steps >= 2`.
* All values are finite (no `NaN` or `Inf`). Non-contiguous or `float64` arrays are automatically coerced to contiguous `float32`.
* Constant-value arrays (where `min == max`) are supported and automatically padded symmetrically so axes and queries stay well-defined.

---

## 2. Creating Widgets

### 2.1 Univariate Widget (`KDBoxWidget`)

```python
from kdbox import KDBoxWidget

# Symmetrical dictionary syntax (automatically sets X and Y axis labels)
w = KDBoxWidget(
    {"Elapsed Time (s)": prep.time_data},
    {"Fuel Level": prep.value_data},
    attributes=prep.attributes,
)
w
```

* **Optional Labels:** Passing plain arrays (`KDBoxWidget(time_data, value_data)`) omits axis titles and the plot header badge. You can supply labels either via single-key dicts (`{"Time (s)": time_data}`) or explicit `x_label="..."` / `y_label="..."` keyword arguments (explicit kwargs take precedence if both are provided).
* **Custom Initial State (`**kwargs`):** Pass any synced traitlet at construction time --- for example:
  `KDBoxWidget(t, v, width=800, height=400, colormap="viridis", rep_method="meanshift", rep_count=5, diversity=0.25, strict_mode=True)`.

### 2.2 Coordinated Multivariate Widget (`MultivariateKDBox`)

Row `i` across every variable array must represent the **same physical entity** (e.g., Sensor `i` or Trip `i`):

```python
from kdbox import MultivariateKDBox

# Symmetrical labeled dictionary syntax
multi = MultivariateKDBox(
    {"Elapsed Time (s)": prep.time_data},
    {
        "Fuel Level": prep.value_data["level"],
        "Pump State": prep.value_data["pump"],
    },
    attributes=prep.attributes,
    show_all_attr_controls=False,  # Default: show Attribute Filters sidebar on 1st widget only
)
multi
```

* **Unlabeled Sequence Input:** You can also pass a plain list of arrays: `MultivariateKDBox(time_data, [var_a, var_b, var_c])`. In this mode, individual child widgets display no axis labels (matching univariate behavior), while the top Matplotlib overlay automatically assigns placeholder names `var1`, `var2`, `var3`, etc. (or custom names via `y_labels=["A", "B", "C"]`).
* **Accepted `attributes` Formats:** Pass a `dict` of 1D sequences (`{"Sector": arr1, "Model": arr2}`), a Pandas `DataFrame` (each column becomes a filterable attribute), a Pandas `Series`, or a single 1D array/list. Every attribute sequence must have length `n_lines`. Missing values (`None` / `NaN`) are automatically grouped under `"<Missing>"`.

---

## 3. Interactive Canvas & Query Tools

Hovering anywhere over the plot displays a **dashed crosshair** with a white-haloed **`T:` / `V:` coordinate readout** in data units.

### 3.1 Query Tools

Select the active drawing tool from the **Tool** dropdown in the left sidebar. You can draw multiple shapes of different types on the same canvas; each query is assigned a distinct color.

| Tool | How to Draw | How to Move | How to Resize / Adjust | Selection Rule |
| :--- | :--- | :--- | :--- | :--- |
| **Brush (Timebox)** | Click and drag diagonally to draw a rectangle. | Drag anywhere inside the box interior (`move` cursor). | Drag any of the **4 edges** or **4 corners** (`ew`, `ns`, `nwse`, `nesw` cursors). | **Box Intersect** (*Strict* unchecked, default): selects every series intersecting the rectangle.<br>**Strict Timebox** (*Strict* checked): selects only series that stay inside $[v_{\min}, v_{\max}]$ for the *entire* $[t_{\min}, t_{\max}]$ span (rejecting lines that exit through the top or bottom edge). |
| **RNL (Radius)** | Click the desired center and drag outward to set the radius. | Drag anywhere inside the circle interior. | Drag anywhere along the **circle circumference** (cursor adapts radially around the circle). | Selects every series passing within the drawn pixel radius of the circle center. |
| **Angular** | Click the start vertex and drag right/up/down to set duration and target slope. | Drag the **start vertex** or anywhere inside the **shaded wedge**. | Drag the **center tip dot** to change direction & time span. Drag the **upper or lower corner dots** of the wedge to widen or narrow the angle tolerance (`1°`--`85°`). | Selects series whose **net trend angle** across the wedge's time span falls within $\theta \pm \Delta\theta$. |
| **Pin (Hover)** | Click (or drag) to drop a point marker. | Drag the pin dot to reposition. | Fixed 3-pixel probe radius. | Selects every series passing within 3 pixels of the pin --- ideal for tracing tight bundles or specific crossing points. |

#### Angular Readout Convention
* While drawing or adjusting an Angular wedge, a live haloed badge renders beside the tip showing **`θ: +24.5° (±15°)`**.
* Angles are reported in intuitive visual orientation: **positive (`+`)** for upward-sloping trends, **negative (`-`)** for downward-sloping trends, and **`0.0°`** for flat horizontal trends.
* Once released, the target angle and tolerance remain visible in the Information Table row (e.g., `ANG 1 (+24.5° ± 15°)`).

#### Strict Query Mode Visibility
* The **Strict query mode** checkbox applies **only** to `Brush (Timebox)` queries and automatically hides when `RNL`, `Angular`, or `Pin` is selected.
* **When to turn Strict Mode OFF (default):** Selecting local peaks, troughs, or wide noisy regions where lines naturally enter and exit through the top or bottom edges of the rectangle.
* **When to turn Strict Mode ON:** Enforcing hard corridor constraints (e.g., *"find devices whose temperature never left 20--30 °C during the entire selected interval"*).

#### Automatic Focus & Micro-Click Cleanup
* Accidental clicks smaller than 2 pixels (for Brush, RNL, or Angular) are automatically discarded on mouse release.
* Drawing your **first** spatial query automatically switches the view to explore that query.
* Drawing a **second** (or subsequent) spatial query automatically switches the view to **`Intersection`** so you can progressively refine a subset across multiple time windows.

---

## 4. Controls Sidebar Reference

Every slider is paired with a **two-way synchronized numeric input box**. Drag sliders for quick visual exploration, or type exact values into the number box (e.g., `0.02` for `Diversity`) and press `Enter` for fine-grained control.

### 4.1 Density & Layer Controls
* **Colormap (`magma` / `viridis`):** Perceptually uniform color scales applied client-side. Changing the colormap does not recompute the backend density grid.
* **Normalize density (checked by default):**
  * *Checked (Local Normalization):* Scales the brightest pixel of the *currently displayed subset* to full colormap intensity (255). Best for inspecting internal structure inside small selections (e.g., 20 lines).
  * *Unchecked (Global Normalization):* Scales pixel intensities relative to the maximum pixel density of the entire unfiltered dataset. Best for judging how dense a selected subset is compared to the global population.
* **Contrast (Gamma):** Applies nonlinear power-law contrast adjustment ($I' = (I/255)^\gamma$) at 60 FPS in the browser. Values **`< 1.0`** brighten sparse/low-density regions and outlier trails; values **`> 1.0`** darken sparse clutter to emphasize high-density cores. (Slider: `0.1--3.0`; numeric box: `0.05--10.0`.)
* **Data Opacity & Rep Lines Opacity:** Independently fade the density heatmap (`0.0--1.0`) or the representative line overlay (`0.0--1.0`). Setting `Rep Lines Opacity` to `0` hides representative lines without disabling their computation.

### 4.2 Representative Lines Controls
Representative lines are drawn as thick high-contrast polylines (green, cyan, magenta, yellow, red) overlaid on the density field.

* **Rep Lines Method:**
  * **`Heuristic (Density + Slope)`** *(default)*: Picks the highest-density series first, then adds subsequent high-density series whose mean segment-slope difference from previously picked curves meets `Diversity`. Best on clean, low-noise curves when you want to differentiate slope/curvature shapes.
  * **`Mean-Shift (Curve Modes)`**: Runs density-weighted Gaussian Mean-Shift clustering in normalized curve space and snaps each major mode to its closest real series (medoid). **Recommended** for noisy telemetry, multi-modal clusters, or vertically offset parallel bundles (where the slope heuristic can starve secondary bundles).
* **Rep Lines Count:** Maximum number of representative lines to extract (slider: `1--10`; numeric box: `1--50`). Fewer lines may be returned if fewer distinct modes or diverse curves exist under the current `Diversity` setting.
* **Diversity:** Controls how far apart representative curves must be (slider: `0.0--10.0`; numeric box: `0.0--100.0`, step `0.01`). Increasing `Diversity` merges similar trends into fewer, more distinct curves; setting it to `0.0` returns the top-`k` densest curves.

### 4.3 Categorical Attribute Filters
Displayed automatically whenever `attributes` are passed to the widget (and hidden otherwise):
* **Multi-Value (`OR`) within an attribute:** Selecting values from a dropdown adds removable filter pills (e.g., `[North ×] [South ×]`). Clicking a pill removes that value; selecting a `✓`-marked value again toggles it off.
* **Multi-Attribute (`AND`) across attributes:** Filtering on both `Sector = North` and `Model = A` selects only series satisfying *both* conditions.
* **High-Cardinality Search:** Whenever an attribute has more than **12** unique values, a live **Search** input automatically appears above its dropdown, and the dropdown displays up to **100** frequency-sorted matches at a time so datasets with thousands of IDs remain responsive.
* **Reset:** Clears all active attribute filters and search boxes at once.
* **Synchronized in `MultivariateKDBox`:** By default (`show_all_attr_controls=False`), only the top child widget displays the Attribute Filters section to save vertical space, but every selection is synced across all stacked variables automatically.

---

## 5. Reading the Information Table

The Information Table below each canvas summarizes the exact line count, total time points, min, max, mean, and variance of every dataset partition. Clicking a row's **`Explore`** radio button switches what is rendered on that widget's density canvas and representative line overlay:

| Table Row | When Visible | What It Shows |
| :--- | :--- | :--- |
| **Global Dataset** | Always | The **true unfiltered dataset** (all `n_lines`), ignoring all local spatial queries, attribute filters, and incoming cross-widget filters. Useful for quick before/after comparison. |
| ***Attribute Filter*** | When any categorical attribute filter is active | All series matching the active attribute filter pills. |
| ***Coordinated Filter*** | In `MultivariateKDBox`, when one or more *other* widgets have an active spatial query | The exact entity subset selected by sibling widgets (intersected with any active attribute filter). |
| **■ BRUSH / RNL / ANG / PIN `k`** | For each drawn spatial query | Series matching that specific shape, intersected with any active attribute and coordinated filters. The currently explored shape renders with a thicker border (`3px`) on the canvas. |
| ***Intersection*** | When 2+ spatial queries exist on this widget | Series matching **all** local spatial queries simultaneously (AND), intersected with attribute/coordinated masks. |
| ***Union*** | When 2+ spatial queries exist on this widget | Series matching **at least one** local spatial query (OR), intersected with attribute/coordinated masks. |

* **Deleting Queries:** Clicking **`Delete`** on a row removes that shape; **`Delete All Queries`** clears all local shapes on that widget. If the deleted row was active, `explore_mode` automatically falls back to `Coordinated Filter` (if active), then `Attribute Filter` (if active), then `Global Dataset`. Dropping from 2 queries to 1 automatically switches `Intersection`/`Union` to the remaining query.

---

## 6. Understanding Coordinated Multiple Views (`MultivariateKDBox`)

When working with `MultivariateKDBox`, all child widgets are linked by **entity row index**:

1. **Cross-Widget Spatial Filtering:** Drawing and exploring a Brush, RNL, Angular, or Pin query on Variable A immediately sends the matching entity indices to Variables B, C, etc., switching any unqueried sibling widgets to ***Coordinated Filter*** mode so you see how those exact entities behave in every other dimension.
2. **Compound Cross-Filtering:** If you draw a second spatial query on Variable B as well, Variable A and Variable B cross-filter each other with the intersection (`AND`) of both queries, while Variable C receives the intersection of both.

### Anatomy of the Top Matplotlib Overlay Plot

Above the stacked widgets sits the **Coordinated Multivariate Representative Lines** plot, which updates automatically whenever any query, attribute filter, representative-line parameter, or `explore_mode` changes. Several subtle visual features are built into this header chart:

* **Color-Coded Multi-Axis Spines:** Each variable is assigned a distinct color from Matplotlib's `tab10` palette.
  * For **1--2 variables**, the left and right Y-axes display bold colored variable titles.
  * For **3+ variables**, additional twin Y-axes alternate outward on the right and left sides (offset by `45 px` per pair) with matching colored spines and tick labels, preventing overlapping axes while keeping the legend compact.
* **Shaded Vertical Time Bands (`axvspan`) & Dashed Borders:** Whenever a child widget has an active local spatial query, the top overlay plot shades the queried time interval $[t_{\min}, t_{\max}]$ with a **semi-transparent vertical band and dashed boundary lines in that variable's exact color**.
  * *Why this matters:* Cross-filtering filters sibling variables across their entire time range, so without these bands it would be hard to tell *when* the triggering anomaly occurred. The shaded band anchors the exact query window across all overlaid representative curves simultaneously (for example, letting you verify whether a pressure drop in Variable 2 occurred *during* or *after* a brushed temperature spike in Variable 1).
* **Dynamic Legend Value Ranges:** Each legend entry displays `{Variable Name} [v_min, v_max]`, reporting the exact min/max value range of that variable's *currently plotted representative lines* --- giving instant scale context even when 4--5 variables share the plot.
* **Empty Selection Handling:** If a strict combination of filters matches `0` lines, the density canvas clears cleanly and the Matplotlib overlay keeps the global X-axis range `[t_min, t_max]` and each variable's Y-axis range `[v_min, v_max]` locked (rather than collapsing to Matplotlib's default `[0.0, 1.0]` frame).

---

## 7. Extracting Data & Programmatic Control in Python

Because all filtering state lives in Python, you can interactively isolate a cluster in the notebook UI and immediately pull its exact rows into the next cell for statistical modeling, anomaly labeling, or export.

### 7.1 Export Methods on `KDBoxWidget`

```python
idx = w.get_selected_indices()           # 1D int32 array of active row indices
sel_t, sel_v = w.get_selected_data()     # (n_sel, n_steps) float32 arrays
sel_attrs = w.get_selected_attributes()  # {attr_name: 1D str array}

rep_idx = w.get_representative_indices()           # 1D int32 array of rep indices
rep_t, rep_v = w.get_representative_data()         # (n_reps, n_steps) float32 arrays
rep_attrs = w.get_representative_attributes()      # {attr_name: 1D str array}
```

### 7.2 Top-Level Export Methods on `MultivariateKDBox`

`MultivariateKDBox` exposes matching top-level methods so you never have to manually intersect child widgets:

```python
# Indices matching the coordinated selection across ALL views
idx = multi.get_selected_indices()            # 1D int32 array

# Map back to original DataFrame entity IDs (when prepared via from_dataframe)
selected_ids = prep.ids[idx]
subset_df = df[df["unique_id"].isin(selected_ids)]

# Extract aligned time matrix and per-variable value dict for selected entities
sel_t, sel_vals = multi.get_selected_data()   # (n_sel, n_steps), {var_name: array}
sel_attrs = multi.get_selected_attributes()   # {attr_name: 1D str array}

# Representative lines per variable
rep_idx_map = multi.get_representative_indices()      # {var_name: int32 array}
rep_data_map = multi.get_representative_data()        # {var_name: (t_arr, v_arr)}
rep_attr_map = multi.get_representative_attributes()  # {var_name: {attr: array}}
```

You can still access any individual child widget directly via `multi.kd_widgets["Fuel Level"]`.

### 7.3 Programmatic Control via Synced Traitlets

All UI controls support two-way binding. Updating a widget attribute in Python immediately updates the browser UI and re-renders:

```python
w.colormap = "viridis"
w.rep_method = "meanshift"
w.rep_count = 4
w.diversity = 0.25
w.strict_mode = True
w.normalize_density = False

# Apply an attribute filter programmatically
import json
w.attr_filters = json.dumps({"Sector": ["North", "East"]})
```

---

## 8. Tips & Troubleshooting

* **First-Run Compilation Delay:** The very first widget instantiation in a fresh Python kernel takes ~2--4 seconds while Numba JIT-compiles the spatial tree and rasterization kernels. All subsequent instantiations and queries within the same kernel execute in milliseconds.
* **Notebook Rendering (`anywidget` & `uv`):** `kdbox` works out of the box in **JupyterLab 4+**, **Jupyter Notebook 7+**, and **VS Code Jupyter Notebooks** without legacy `jupyter labextension` builds.
  * If a widget prints `<kdbox.widget.KDBoxWidget object at 0x...>` instead of rendering interactively, ensure `jupyterlab`, `anywidget`, and `ipywidgets` are installed in the **same** environment (`sys.prefix`) so JupyterLab can discover the frontend extensions (`anywidget` and `@jupyter-widgets/jupyterlab-manager`).
  * **When using `uv`:** Avoid launching JupyterLab via `uv run --with=jupyterlab jupyter lab`, which places JupyterLab in an isolated ephemeral prefix separate from `.venv`. Instead, install or run JupyterLab inside the project environment using:
    ```bash
    uv sync --extra jupyter
    uv run jupyter lab
    ```
    You can confirm the extensions are detected with `uv run jupyter labextension list`.
* **Large Datasets (50K--100K+ series):**
  * Use `n_steps` between `50` and `200` in `from_dataframe` to keep segment counts lean while preserving full visual resolution (canvas width is `1000 px` by default).
  * If a dense overview looks washed out, lower **Contrast (Gamma)** to `0.4--0.6` to reveal low-density trails, or raise it to `1.5--2.0` to isolate the highest-density core.
* **Constant or Flat Variables:** If a variable has identical values across all timestamps (e.g., an inactive sensor stuck at `0.0`), `kdbox` automatically pads its vertical axis (`±0.5` or `±5%`) so the flat line renders cleanly in the center of the canvas without division-by-zero errors.
