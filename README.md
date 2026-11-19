# kdbox

High-performance interactive Jupyter widgets for exploring large-scale univariate and multivariate time-series datasets. Built on a **Numba-accelerated** bounding-volume KD-tree backend and an **`anywidget` HTML5 Canvas** frontend, inspired by *KD-Box* (Zhao et al., IEEE VIS 2021).

---

## Key Capabilities

* **Low-Latency Spatial Querying:** Interactive **Brush (Timebox)** (with toggleable *Box Intersect* and *Strict Timebox* modes), radially resizable **RNL (Radius)**, shaded **Angular** slope wedges with adjustable $\pm\Delta\theta$ tolerance, and **Pin** probes.
* **Dual Representative Line Extraction:** Switch live between the **Greedy Density + Slope Heuristic** and **Density-Weighted Curve Mean-Shift** clustering with medoid snapping.
* **Coordinated Multiple Views (`MultivariateKDBox`):** Cross-filter entity-linked variables simultaneously with an auto-scaling multi-axis Matplotlib overlay and color-coded shaded query time windows.
* **Categorical Attribute Filtering:** Multi-value (`OR`) and multi-attribute (`AND`) metadata filters with live search for high-cardinality categories, automatically synchronized across coordinated views.
* **60 FPS Client-Side Density Controls:** Adjust colormaps (`magma` / `viridis`), local vs. global normalization, gamma contrast, and layer opacities without kernel round-trips.
* **DataFrame Preprocessing (`from_dataframe`):** Resample ragged long-format Pandas DataFrames onto uniform grids with synchronized time, duration, quantile, and value bounding.
* **Two-Way Python Export API:** Extract active entity indices, aligned arrays, representative curves, and metadata directly back into Python for downstream analysis.

---

## Documentation & Examples

| Document | Contents |
| :--- | :--- |
| **[User Guide](docs/user_guide.md)** | Complete walkthrough of data preparation (`from_dataframe`, `align_time_series`), widget creation, interactive tools, sidebar controls, Information Table semantics, coordinated overlay plots, and Python export APIs. |
| **[Architecture & Algorithms](docs/architecture.md)** | Split Python/JS design, simplifications relative to Zhao et al. (2021), in-place midpoint KD-tree partitioning, spatial query equations, formal Curve Mean-Shift derivation and comparison with the Heuristic, and set-theoretic masking rules. |
| **[Interactive Demo Notebook](examples/demo.ipynb)** | Self-contained JupyterLab walkthrough using generated synthetic multivariate data covering every query tool, Mean-Shift vs. Heuristic comparison, attribute filtering, and data export. |

---

## Installation

Requires **Python 3.10+**. Works out of the box in **JupyterLab 4+** and **Jupyter Notebook 7+** with no extra JavaScript build step.

### Using `uv` (Recommended)

```bash
# SSH (recommended for private org repos).
uv add "kdbox[data] @ git+ssh://git@github.com/OSM-Piatnica/kdbox.git@v0.1.0"

# HTTPS (if authenticated via git credential manager / gh CLI).
uv add "kdbox[data] @ git+[https://github.com/](https://github.com/)OSM-Piatnica/kdbox.git@v0.1.0"
```

### Using `pip`

```bash
pip install "kdbox[data] @ git+ssh://git@github.com/OSM-Piatnica/kdbox.git@v0.1.0"
```

### Optional Dependency Groups

* `kdbox` --- Core engine and widgets (`numba`, `numpy`, `anywidget`, `ipywidgets`, `matplotlib`, `pillow`, `traitlets`).
* `kdbox[data]` --- Adds `pandas` and `scipy` for `from_dataframe` and spline interpolation.
* `kdbox[jupyter]` --- Adds `jupyterlab>=4.0.0`.
* `kdbox[all]` --- Installs both `data` and `jupyter` extras.

---

## Quick Start

### Multivariate Exploration from a Pandas DataFrame

```python
import pandas as pd
from kdbox import MultivariateKDBox, from_dataframe

# Clean, bound, align, and resample a long-format DataFrame synchronously.
prep = from_dataframe(
    df,
    id_col="unique_id",
    time_col="timestamp",
    value_cols=["level", "pump"],
    attr_cols=["supplier_id"],
    n_steps=100,
    align="start",
    max_duration_quantile=0.95,
    value_range={"level": (0, 100), "pump": (0, 100)},
    value_out_of_range="drop_series",
    kind={"level": "linear", "pump": "previous"},
)

# Launch coordinated interactive widgets.
multi = MultivariateKDBox(
    {"Elapsed Time (s)": prep.time_data},
    {
        "Fuel Level": prep.value_data["level"],
        "Pump State": prep.value_data["pump"],
    },
    attributes=prep.attributes,
)
multi
```

After brushing a region or filtering attributes in the notebook UI, extract the exact matching entities in the next cell:

```python
selected_idx = multi.get_selected_indices()
selected_ids = prep.ids[selected_idx]

sel_time, sel_values = multi.get_selected_data()
sel_attrs = multi.get_selected_attributes()
```

### Univariate Usage from NumPy Arrays

```python
from kdbox import KDBoxWidget

w = KDBoxWidget(
    {"Time (s)": time_data},
    {"Temperature (°C)": temperature},
    attributes={"Sector": sectors},
    rep_method="meanshift",
)
w
```

---

## Development & Testing

```bash
# Clone and install in editable mode with dev dependencies.
git clone git@github.com:OSM-Piatnica/kdbox.git
cd kdbox
uv sync

# Run the automated test suite.
uv run pytest -v

# Build wheel and source distribution into dist/
uv build
```

---

## Reference & License

Inspired by:
> Y. Zhao, J. Zhang, C.-W. Fu, M. Xu, D. Moritz, and Y. Wang, *"KD-Box: Line-segment-based KD-tree for Interactive Exploration of Large-scale Time-Series Data,"* IEEE Transactions on Visualization and Computer Graphics (VIS 2021).

See [LICENSE](LICENSE) for licensing terms.
