import json
import math

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from kdbox import (
    KDBoxWidget,
    MultivariateKDBox,
    PreparedTimeSeries,
    align_time_series,
    from_dataframe,
)
from kdbox.kd_tree import (
    angular_query,
    build_kd_tree,
    create_tvs_segments,
    get_representative_lines,
    rasterize_density,
    rnl_query,
    timebox_query,
)


@pytest.fixture(scope="module")
def sample_arrays():
    """Deterministic synthetic multivariate dataset with 3 distinct bundles."""
    rng = np.random.default_rng(42)
    n_lines, n_steps = 150, 50
    t_row = np.linspace(0.0, 10.0, n_steps, dtype=np.float32)
    time_data = np.tile(t_row, (n_lines, 1))

    # Three distinct trajectory bundles for Variable 1.
    b1 = 10.0 * np.sin(t_row) + rng.normal(0.0, 1.0, size=(60, n_steps))
    b2 = -15.0 + 3.0 * t_row + rng.normal(0.0, 1.0, size=(50, n_steps))
    b3 = 15.0 - 2.5 * t_row + rng.normal(0.0, 1.0, size=(40, n_steps))
    var1 = np.vstack([b1, b2, b3]).astype(np.float32)

    var2 = (np.cumsum(rng.normal(0.0, 0.5, size=(n_lines, n_steps)), axis=1) + 25.0).astype(
        np.float32
    )
    attrs = {
        "Bundle": np.array(["Sine"] * 60 + ["Up"] * 50 + ["Down"] * 40, dtype=object),
        "Group": np.array(["A", "B"] * (n_lines // 2), dtype=object),
    }
    return time_data, var1, var2, attrs


# =============================================================================
# Numba Spatial Engine Tests (kd_tree.py)
# =============================================================================


def test_kd_tree_build_and_spatial_queries(sample_arrays):
    time_data, var1, _, _ = sample_arrays
    n_lines, n_steps = time_data.shape
    segments = create_tvs_segments(time_data, var1)
    tree = build_kd_tree(segments, max_leaf_size=32, max_depth=15)

    split_dims, _, _, _, node_bounds, seg_starts, _, _ = tree
    internal = np.where(split_dims != -1)[0]
    assert np.all(np.isin(split_dims[internal], [0, 1]))
    used = np.where((split_dims != -1) | (seg_starts != -1))[0]
    assert np.all(np.isfinite(node_bounds[used]))

    t_min, t_max = float(time_data.min()), float(time_data.max())
    v_min, v_max = float(var1.min()), float(var1.max())
    t_scale = 1000.0 / (t_max - t_min)
    v_scale = 500.0 / (v_max - v_min)

    # Brush: intersect mode must be a superset of strict mode.
    tb_inter = timebox_query(tree, segments, 2.0, 4.0, -10.0, 10.0, n_lines, strict=False)
    tb_strict = timebox_query(tree, segments, 2.0, 4.0, -10.0, 10.0, n_lines, strict=True)
    assert len(tb_inter) >= len(tb_strict) > 0
    assert np.all(np.isin(tb_strict, tb_inter))

    # RNL query near center of bundle 1.
    rnl_res = rnl_query(tree, segments, 5.0, 0.0, 40.0, t_scale, v_scale, n_lines)
    assert len(rnl_res) > 0

    # Angular query: net-trend upward slope should recover upward trajectories.
    ang_up = angular_query(
        tree, segments, 2.0, 8.0, -0.35, math.radians(20.0), t_scale, v_scale, n_lines, strict=False
    )
    assert len(ang_up) > 0


def test_representative_lines_heuristic_and_meanshift(sample_arrays):
    time_data, var1, _, _ = sample_arrays
    n_lines, n_steps = time_data.shape
    segments = create_tvs_segments(time_data, var1)
    all_lines = np.arange(n_lines, dtype=np.int32)
    t_min, t_max = float(time_data.min()), float(time_data.max())
    v_min, v_max = float(var1.min()), float(var1.max())
    grid = rasterize_density(segments, all_lines, 400, 200, t_min, t_max, v_min, v_max, n_steps)

    # Regression check: num_reps=1 must not crash or write out of bounds.
    for use_ms in (False, True):
        for k in (1, 3):
            reps = get_representative_lines(
                segments, all_lines, grid, 400, 200,
                t_min, t_max, v_min, v_max, n_steps,
                num_reps=k, diversity_threshold=0.1, use_meanshift=use_ms,
            )
            assert 1 <= len(reps) <= k

    # Mean-Shift should capture all 3 distinct bundles on synthetic 3-bundle data.
    ms_reps = get_representative_lines(
        segments, all_lines, grid, 400, 200,
        t_min, t_max, v_min, v_max, n_steps,
        num_reps=3, diversity_threshold=0.2, use_meanshift=True,
    )
    bundles = {0 if idx < 60 else (1 if idx < 110 else 2) for idx in ms_reps}
    assert len(bundles) == 3


# =============================================================================
# Preprocessing & Alignment Utilities Tests (utils.py)
# =============================================================================


def test_align_time_series_modes():
    raw_t = np.array([[10.0, 15.0, 20.0], [100.0, 120.0, 140.0]], dtype=np.float64)

    t_start = align_time_series(raw_t, align="start")
    assert np.allclose(t_start[:, 0], 0.0)
    assert np.allclose(t_start[:, -1], [10.0, 40.0])

    t_end = align_time_series(raw_t, align="end")
    assert np.allclose(t_end[:, -1], 0.0)
    assert np.allclose(t_end[:, 0], [-10.0, -40.0])

    t_center = align_time_series(raw_t, align="center")
    assert np.allclose(t_center[:, 1], 0.0)

    t_norm = align_time_series(raw_t, align="normalize")
    assert np.allclose(t_norm[:, 0], 0.0)
    assert np.allclose(t_norm[:, -1], 1.0)


def test_from_dataframe_synchronized_bounding():
    df = pd.DataFrame({
        "id": ["s1"] * 5 + ["s2"] * 5 + ["s3"] * 5,
        "t": [0.0, 2.5, 5.0, 7.5, 10.0] * 2 + [0.0, 25.0, 50.0, 75.0, 100.0],
        "level": [0.0, 2.5, 999.0, 7.5, 10.0, 0.0, 2.5, 5.0, 7.5, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "pump": [0.0, 0.0, 50.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        "supplier": ["X"] * 5 + ["Y"] * 5 + ["Z"] * 5,
    })

    # drop_points on `level` must drop the t=5 spike across BOTH `level` and `pump`.
    prep = from_dataframe(
        df,
        id_col="id",
        time_col="t",
        value_cols=["level", "pump"],
        attr_cols=["supplier"],
        n_steps=5,
        duration_range=(0.0, 20.0),
        value_range={"level": (0.0, 20.0)},
        value_out_of_range="drop_points",
    )
    assert isinstance(prep, PreparedTimeSeries)
    assert list(prep.ids) == ["s1", "s2"]
    assert prep.time_data.shape == (2, 5)
    assert np.allclose(prep.value_data["level"][0], [0.0, 2.5, 5.0, 7.5, 10.0])
    assert np.allclose(prep.value_data["pump"][0], 0.0)
    assert list(prep.attributes["supplier"]) == ["X", "Y"]

    # drop_series on `level` removes s1 entirely across all outputs.
    prep_ds = from_dataframe(
        df,
        id_col="id",
        time_col="t",
        value_cols=["level", "pump"],
        n_steps=5,
        value_range={"level": (0.0, 20.0)},
        value_out_of_range="drop_series",
        time_range=(0.0, 20.0),
        time_out_of_range="drop_series",
    )
    assert list(prep_ds.ids) == ["s2"]


# =============================================================================
# Widget State, Validation & Coordination Tests (widget.py)
# =============================================================================


def test_kdbox_widget_validation_and_constant_series():
    t_const = np.tile(np.linspace(0.0, 5.0, 10, dtype=np.float32), (10, 1))
    v_const = np.full((10, 10), 42.0, dtype=np.float32)

    w = KDBoxWidget({"Time (s)": t_const}, {"Const": v_const}, width=200, height=100)
    assert w.x_label == "Time (s)"
    assert w.y_label == "Const"
    assert w.v_min < 42.0 < w.v_max
    assert w.density_image != ""

    with pytest.raises(ValueError):
        KDBoxWidget(t_const, np.full((9, 10), 1.0, dtype=np.float32))
    with pytest.raises(ValueError):
        bad_v = v_const.copy()
        bad_v[0, 0] = np.nan
        KDBoxWidget(t_const, bad_v)


def test_kdbox_widget_queries_and_attributes(sample_arrays):
    time_data, var1, _, attrs = sample_arrays
    w = KDBoxWidget(time_data, var1, attributes=attrs, width=400, height=200)

    # Filter by categorical attribute (OR within attribute, AND across attributes).
    w.attr_filters = json.dumps({"Bundle": ["Sine", "Up"], "Group": ["A"]})
    assert w.explore_mode == "attribute"
    sel_idx = w.get_selected_indices()
    assert len(sel_idx) == 55  # Half of (60 + 50)
    sel_attrs = w.get_selected_attributes()
    assert set(sel_attrs["Group"]) == {"A"}

    # Apply a Brush query via query_data and verify export API.
    brush_q = [{"id": "q_1", "type": "brush", "color": "#e6194B", "x_min": 50, "y_min": 20, "w": 150, "h": 160}]
    w.query_data = json.dumps(brush_q)
    w.explore_mode = "q_1"
    sel_t, sel_v = w.get_selected_data()
    assert sel_t.shape == sel_v.shape
    assert sel_t.shape[0] <= 55


def test_multivariate_kdbox_coordination_and_exports(sample_arrays):
    time_data, var1, var2, attrs = sample_arrays
    multi = MultivariateKDBox(
        {"Time (s)": time_data},
        {"Var1": var1, "Var2": var2},
        attributes=attrs,
    )

    # Attribute sidebar is deduplicated by default (shown on first widget only).
    assert multi.kd_widgets["Var1"].show_attr_controls is True
    assert multi.kd_widgets["Var2"].show_attr_controls is False

    # Attribute filter set on Var1 syncs to Var2 automatically.
    multi.kd_widgets["Var1"].attr_filters = json.dumps({"Bundle": ["Up"]})
    assert multi.kd_widgets["Var2"].attr_filters == json.dumps({"Bundle": ["Up"]})
    assert len(multi.get_selected_indices()) == 50

    # Spatial brush on Var1 propagates as a Coordinated Filter to Var2.
    w1 = multi.kd_widgets["Var1"]
    w2 = multi.kd_widgets["Var2"]
    w1.query_data = json.dumps(
        [{"id": "q_b1", "type": "brush", "color": "#e6194B", "x_min": 100, "y_min": 100, "w": 300, "h": 300}]
    )
    w1.explore_mode = "q_b1"

    assert w2.explore_mode == "coordinated"
    assert np.array_equal(w1.get_selected_indices(), w2.get_selected_indices())
    assert np.array_equal(multi.get_selected_indices(), w1.get_selected_indices())

    sel_t, sel_vals = multi.get_selected_data()
    assert set(sel_vals.keys()) == {"Var1", "Var2"}
    assert sel_t.shape == sel_vals["Var1"].shape == sel_vals["Var2"].shape

    rep_indices = multi.get_representative_indices()
    assert set(rep_indices.keys()) == {"Var1", "Var2"}

def test_kdbox_widget_observed_kwargs_at_init(sample_arrays):
    time_data, var1, _, attrs = sample_arrays
    w = KDBoxWidget(
        {"Time (s)": time_data},
        {"Var1": var1},
        attributes=attrs,
        rep_method="meanshift",
        rep_count=3,
        diversity=0.15,
        strict_mode=True,
        normalize_density=False,
        attr_filters=json.dumps({"Bundle": ["Sine"]}),
    )
    assert w.rep_method == "meanshift"
    assert w.diversity == pytest.approx(0.15)
    assert w.strict_mode is True
    assert w.explore_mode == "attribute"
    assert len(w.get_selected_indices()) == 60
    assert w.density_image != ""
