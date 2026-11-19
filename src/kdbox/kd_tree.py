"""Numba-accelerated spatial indexing and rendering engine for KD-Box.

This module implements the core computational backend inspired by the KD-Box
architecture (Zhao et al., IEEE VIS 2021):

- Line-segment decomposition in Time-Value-Slope (TVS) space.
- Iterative, flat-array KD-tree construction compatible with Numba `@njit`
  compilation (avoiding Python heap objects and recursion overhead).
- Low-latency spatial queries: Radius Nearest Line (RNL), Brush / Timebox
  (both intersection and strict ghost-boundary modes), and Angular queries.
- Digital Differential Analyzer (DDA) line rasterization for 2D density fields.
- Representative line extraction via either a greedy density-plus-slope
  heuristic or density-weighted curve Mean-Shift clustering.

Notes
-----
All functions decorated with `@njit` expect contiguous 32-bit floating-point
(`np.float32`) or 32-bit integer (`np.int32`) NumPy arrays. Runtime PEP 484
type annotations are intentionally omitted from `@njit` signatures to
maintain broad Numba version compatibility; full types and shapes are
documented in each function's NumPy-style docstring.
"""

import math
import numpy as np
from numba import njit


# =============================================================================
# Line Segmentation
# =============================================================================


@njit
def create_tvs_segments(time_data, value_data):
    """Decompose 2D time-series matrices into flat TVS line segments.

    Each pair of consecutive samples `(t_j, v_j) -> (t_{j+1}, v_{j+1})` along
    a curve is stored as one row in the returned segment table.

    Parameters
    ----------
    time_data : np.ndarray of float32, shape (n_lines, n_steps)
        Contiguous matrix of timestamps for each series.
    value_data : np.ndarray of float32, shape (n_lines, n_steps)
        Contiguous matrix of values corresponding to `time_data`.

    Returns
    -------
    segments : np.ndarray of float32, shape (n_lines * (n_steps - 1), 6)
        Segment table with columns
        `[t_start, t_end, v_start, v_end, slope, line_idx]`.
        Segments are ordered contiguously by `line_idx` so that line `i`
        occupies rows `[i * (n_steps - 1), (i + 1) * (n_steps - 1))`.
    """
    n_lines, n_steps = time_data.shape
    n_segments = n_lines * (n_steps - 1)
    
    # Pre-allocate array for speed.
    # Col 0: t_min, Col 1: t_max, Col 2: v_start, Col 3: v_end, Col 4: slope, Col 5: line_idx
    segments = np.empty((n_segments, 6), dtype=np.float32)
    
    idx = 0
    for i in range(n_lines):
        for j in range(n_steps - 1):
            t1, t2 = time_data[i, j], time_data[i, j+1]
            v1, v2 = value_data[i, j], value_data[i, j+1]
            
            # Prevent division by zero.
            slope = (v2 - v1) / (t2 - t1) if t2 != t1 else 0.0
            
            segments[idx, 0] = t1
            segments[idx, 1] = t2
            segments[idx, 2] = v1
            segments[idx, 3] = v2
            segments[idx, 4] = slope
            segments[idx, 5] = i
            idx += 1
            
    return segments


# =============================================================================
# Flat KD-Tree Construction
# =============================================================================


@njit
def compute_segment_bounds(segments):
    """Pre-compute axis-aligned bounding boxes for all TVS segments.

    Parameters
    ----------
    segments : np.ndarray of float32, shape (n_segments, 6)
        Segment table produced by `create_tvs_segments`.

    Returns
    -------
    bounds : np.ndarray of float32, shape (n_segments, 6)
        Bounding intervals `[t_min, t_max, v_min, v_max, s_min, s_max]`
        for each segment.
    """
    n_segments = segments.shape[0]
    bounds = np.empty((n_segments, 6), dtype=np.float32)
    for i in range(n_segments):
        bounds[i, 0] = min(segments[i, 0], segments[i, 1])  # t_min
        bounds[i, 1] = max(segments[i, 0], segments[i, 1])  # t_max
        bounds[i, 2] = min(segments[i, 2], segments[i, 3])  # v_min
        bounds[i, 3] = max(segments[i, 2], segments[i, 3])  # v_max
        bounds[i, 4] = segments[i, 4]                       # s_min
        bounds[i, 5] = segments[i, 4]                       # s_max
    return bounds

@njit
def build_kd_tree(segments, max_leaf_size=32, max_depth=15):
    """Build a flat-array bounding-volume KD-tree over time-series segments.

    Uses an iterative depth-first stack with in-place midpoint partitioning
    over a single ``leaf_seg_indices`` permutation array of exact length
    ``n_segments``. Each segment resides in exactly one leaf (``1.0x`` memory,
    zero duplication), while every internal node stores the tight bounding box
    of all segments in its subtree to guarantee exact spatial pruning.

    Parameters
    ----------
    segments : np.ndarray of float32, shape (n_segments, 6)
        Segment table produced by `create_tvs_segments`.
    max_leaf_size : int, default=32
        Maximum number of segments stored in a leaf before stopping splits.
    max_depth : int, default=15
        Maximum tree depth. Controls pre-allocated node capacity.

    Returns
    -------
    tree_arrays : tuple of np.ndarray
        8-tuple `(split_dims, split_vals, left_children, right_children,
        node_bounds, seg_starts, seg_counts, leaf_seg_indices)` trimmed to
        the exact number of allocated tree nodes.
    """
    n_segments = segments.shape[0]
    seg_bounds = compute_segment_bounds(segments)

    if n_segments == 0:
        return (
            np.full(1, -1, dtype=np.int32),
            np.zeros(1, dtype=np.float32),
            np.full(1, -1, dtype=np.int32),
            np.full(1, -1, dtype=np.int32),
            np.zeros((1, 6), dtype=np.float32),
            np.zeros(1, dtype=np.int32),
            np.zeros(1, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )

    # Pre-allocate flat Structure-of-Arrays (SoA) node storage.
    # A binary tree of height `max_depth` has at most 2^(max_depth + 1) - 1 nodes.
    max_nodes = (1 << (max_depth + 1)) + 2
    split_dims = np.full(max_nodes, -1, dtype=np.int32)
    split_vals = np.zeros(max_nodes, dtype=np.float32)
    left_children = np.full(max_nodes, -1, dtype=np.int32)
    right_children = np.full(max_nodes, -1, dtype=np.int32)
    node_bounds = np.zeros((max_nodes, 6), dtype=np.float32)
    
    # Leaf node properties.
    seg_starts = np.full(max_nodes, -1, dtype=np.int32)
    seg_counts = np.zeros(max_nodes, dtype=np.int32)

    # Pre-compute segment midpoints and global Time/Value spans for scale-invariant splitting.
    seg_mids = np.empty((n_segments, 2), dtype=np.float32)
    g_t_min = np.inf
    g_t_max = -np.inf
    g_v_min = np.inf
    g_v_max = -np.inf
    for i in range(n_segments):
        t0, t1 = seg_bounds[i, 0], seg_bounds[i, 1]
        v0, v1 = seg_bounds[i, 2], seg_bounds[i, 3]
        seg_mids[i, 0] = 0.5 * (t0 + t1)
        seg_mids[i, 1] = 0.5 * (v0 + v1)
        if t0 < g_t_min:
            g_t_min = t0
        if t1 > g_t_max:
            g_t_max = t1
        if v0 < g_v_min:
            g_v_min = v0
        if v1 > g_v_max:
            g_v_max = v1

    global_t_span = max(1e-5, g_t_max - g_t_min)
    global_v_span = max(1e-5, g_v_max - g_v_min)

    # Single in-place segment index array (exact size n_segments, cannot overflow).
    leaf_seg_indices = np.arange(n_segments, dtype=np.int32)

    # Iterative DFS stack (maximum simultaneous depth is max_depth + 1).
    stack_size = max_depth + 4
    stack_node_id = np.zeros(stack_size, dtype=np.int32)
    stack_depth = np.zeros(stack_size, dtype=np.int32)
    stack_seg_start = np.zeros(stack_size, dtype=np.int32)
    stack_seg_count = np.zeros(stack_size, dtype=np.int32)
    
    stack_ptr = 0
    stack_node_id[0] = 0
    stack_depth[0] = 0
    stack_seg_start[0] = 0
    stack_seg_count[0] = n_segments
    stack_ptr += 1
    
    node_counter = 1
    
    while stack_ptr > 0:
        stack_ptr -= 1
        curr_node = stack_node_id[stack_ptr]
        curr_depth = stack_depth[stack_ptr]
        curr_start = stack_seg_start[stack_ptr]
        curr_count = stack_seg_count[stack_ptr]

        # Compute tight bounding volume and midpoint extent for the current slice.
        min_b0 = np.inf
        max_b0 = -np.inf
        min_b1 = np.inf
        max_b1 = -np.inf
        min_b2 = np.inf
        max_b2 = -np.inf
        mid_min0 = np.inf
        mid_max0 = -np.inf
        mid_min1 = np.inf
        mid_max1 = -np.inf

        for i in range(curr_start, curr_start + curr_count):
            s_idx = leaf_seg_indices[i]
            if seg_bounds[s_idx, 0] < min_b0:
                min_b0 = seg_bounds[s_idx, 0]
            if seg_bounds[s_idx, 1] > max_b0:
                max_b0 = seg_bounds[s_idx, 1]
            if seg_bounds[s_idx, 2] < min_b1:
                min_b1 = seg_bounds[s_idx, 2]
            if seg_bounds[s_idx, 3] > max_b1:
                max_b1 = seg_bounds[s_idx, 3]
            if seg_bounds[s_idx, 4] < min_b2:
                min_b2 = seg_bounds[s_idx, 4]
            if seg_bounds[s_idx, 5] > max_b2:
                max_b2 = seg_bounds[s_idx, 5]

            m0 = seg_mids[s_idx, 0]
            m1 = seg_mids[s_idx, 1]
            if m0 < mid_min0:
                mid_min0 = m0
            if m0 > mid_max0:
                mid_max0 = m0
            if m1 < mid_min1:
                mid_min1 = m1
            if m1 > mid_max1:
                mid_max1 = m1

        node_bounds[curr_node, 0] = min_b0
        node_bounds[curr_node, 1] = max_b0
        node_bounds[curr_node, 2] = min_b1
        node_bounds[curr_node, 3] = max_b1
        node_bounds[curr_node, 4] = min_b2
        node_bounds[curr_node, 5] = max_b2

        # Terminate as a leaf if size, depth, or node capacity thresholds are met.
        if curr_count <= max_leaf_size or curr_depth >= max_depth or node_counter + 2 > max_nodes:
            seg_starts[curr_node] = curr_start
            seg_counts[curr_node] = curr_count
            continue

        # Select split axis along the wider normalized midpoint extent.
        t_norm_ext = (mid_max0 - mid_min0) / global_t_span
        v_norm_ext = (mid_max1 - mid_min1) / global_v_span

        if t_norm_ext <= 1e-12 and v_norm_ext <= 1e-12:
            seg_starts[curr_node] = curr_start
            seg_counts[curr_node] = curr_count
            continue

        if t_norm_ext >= v_norm_ext:
            split_dim = 0
            split_val = 0.5 * (mid_min0 + mid_max0)
        else:
            split_dim = 1
            split_val = 0.5 * (mid_min1 + mid_max1)

        # In-place two-pointer partition of leaf_seg_indices[curr_start : curr_start + curr_count].
        left_ptr = curr_start
        right_ptr = curr_start + curr_count - 1
        while left_ptr <= right_ptr:
            if seg_mids[leaf_seg_indices[left_ptr], split_dim] <= split_val:
                left_ptr += 1
            else:
                tmp = leaf_seg_indices[left_ptr]
                leaf_seg_indices[left_ptr] = leaf_seg_indices[right_ptr]
                leaf_seg_indices[right_ptr] = tmp
                right_ptr -= 1

        l_count = left_ptr - curr_start
        r_count = curr_count - l_count

        # Fallback if float32 midpoint ties place all items on one side.
        if l_count == 0 or r_count == 0:
            l_count = curr_count // 2
            r_count = curr_count - l_count

        split_dims[curr_node] = split_dim
        split_vals[curr_node] = split_val
            
        # Create children and push to stack.
        left_child = node_counter
        right_child = node_counter + 1
        node_counter += 2
        
        left_children[curr_node] = left_child
        right_children[curr_node] = right_child

        # Push right child first so left child is processed next (LIFO).
        stack_node_id[stack_ptr] = right_child
        stack_depth[stack_ptr] = curr_depth + 1
        stack_seg_start[stack_ptr] = curr_start + l_count
        stack_seg_count[stack_ptr] = r_count
        stack_ptr += 1
        
        # Push left.
        stack_node_id[stack_ptr] = left_child
        stack_depth[stack_ptr] = curr_depth + 1
        stack_seg_start[stack_ptr] = curr_start
        stack_seg_count[stack_ptr] = l_count
        stack_ptr += 1

    return (
        split_dims[:node_counter].copy(),
        split_vals[:node_counter].copy(),
        left_children[:node_counter].copy(),
        right_children[:node_counter].copy(),
        node_bounds[:node_counter].copy(),
        seg_starts[:node_counter].copy(),
        seg_counts[:node_counter].copy(),
        leaf_seg_indices,
    )


# =============================================================================
# Geometric Primitives
# =============================================================================


@njit
def point_to_segment_dist_sq(px, py, x1, y1, x2, y2):
    """Compute squared Euclidean distance from a point to a 2D line segment.

    Parameters
    ----------
    px, py : float
        Query point coordinates.
    x1, y1, x2, y2 : float
        Segment endpoint coordinates.

    Returns
    -------
    dist_sq : float
        Minimum squared Euclidean distance between `(px, py)` and the
        closed segment `[(x1, y1), (x2, y2)]`.
    """
    l2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if l2 == 0:
        return (px - x1)**2 + (py - y1)**2
    
    # Clamp orthogonal projection parameter t to [0, 1] for segment endpoints.
    t = max(0.0, min(1.0, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)
    return (px - proj_x)**2 + (py - proj_y)**2

@njit
def segment_intersects_aabb(x1, y1, x2, y2, min_x, max_x, min_y, max_y):
    """Test whether a 2D line segment intersects an axis-aligned bounding box.

    Parameters
    ----------
    x1, y1, x2, y2 : float
        Segment endpoint coordinates.
    min_x, max_x, min_y, max_y : float
        Closed AABB boundaries `[min_x, max_x] x [min_y, max_y]`.

    Returns
    -------
    intersects : bool
        `True` if any point of the segment lies inside or on the AABB border.
    """
    seg_min_x, seg_max_x = min(x1, x2), max(x1, x2)
    seg_min_y, seg_max_y = min(y1, y2), max(y1, y2)
    
    # Fast bounding-box rejection.
    if seg_max_x < min_x or seg_min_x > max_x: return False
    if seg_max_y < min_y or seg_min_y > max_y: return False
    
    # Fast acceptance when segment AABB or either endpoint lies inside box.
    if min_x <= seg_min_x and seg_max_x <= max_x and min_y <= seg_min_y and seg_max_y <= max_y:
        return True
    # Check if endpoints are strictly inside.
    if min_x <= x1 <= max_x and min_y <= y1 <= max_y: return True
    if min_x <= x2 <= max_x and min_y <= y2 <= max_y: return True
    
    # Parametric intersection test against each of the four AABB edges.
    dx = x2 - x1
    dy = y2 - y1
    
    if dx != 0:
        t_left = (min_x - x1) / dx
        if 0 <= t_left <= 1 and min_y <= y1 + t_left * dy <= max_y: return True
        
        t_right = (max_x - x1) / dx
        if 0 <= t_right <= 1 and min_y <= y1 + t_right * dy <= max_y: return True
            
    if dy != 0:
        t_bot = (min_y - y1) / dy
        if 0 <= t_bot <= 1 and min_x <= x1 + t_bot * dx <= max_x: return True
        
        t_top = (max_y - y1) / dy
        if 0 <= t_top <= 1 and min_x <= x1 + t_top * dx <= max_x: return True
            
    return False


# =============================================================================
# Spatial Queries (RNL, Angular, Brush / Timebox)
# =============================================================================


@njit
def rnl_query(tree_arrays, segments, q_t, q_v, radius_px, t_scale, v_scale, n_lines):
    """Execute a Radius Nearest Line (RNL) query in screen-pixel distance.

    Finds all curves that pass within `radius_px` pixels of the query center
    `(q_t, q_v)` so circular queries remain visually circular regardless of
    aspect ratio or data units.

    Parameters
    ----------
    tree_arrays : tuple of np.ndarray
        KD-tree arrays returned by `build_kd_tree`.
    segments : np.ndarray of float32, shape (n_segments, 6)
        Segment table produced by `create_tvs_segments`.
    q_t, q_v : float
        Query center in data coordinates (time, value).
    radius_px : float
        Query circle radius in canvas pixels.
    t_scale, v_scale : float
        Pixels-per-data-unit conversion factors along Time and Value axes.
    n_lines : int
        Total number of time series in the dataset.

    Returns
    -------
    matched_indices : np.ndarray of int32, shape (n_matched,)
        Sorted unique line indices intersecting the query circle.
    """
    (
        split_dims, _, left_children, right_children,
        node_bounds, seg_starts, seg_counts, leaf_seg_indices,
    ) = tree_arrays
    matched_lines = np.zeros(n_lines, dtype=np.bool_)
    r_sq = radius_px ** 2
    
    stack = np.zeros(1000, dtype=np.int32)
    stack_ptr = 0
    stack[stack_ptr] = 0
    stack_ptr += 1
    
    while stack_ptr > 0:
        stack_ptr -= 1
        node = stack[stack_ptr]

        # Compute minimum pixel distance from query center to node AABB.
        t_min, t_max = node_bounds[node, 0], node_bounds[node, 1]
        v_min, v_max = node_bounds[node, 2], node_bounds[node, 3]
        closest_t = max(t_min, min(q_t, t_max))
        closest_v = max(v_min, min(q_v, v_max))
        
        # Prune node using distance mapped to pixel space.
        dist_px_sq = ((closest_t - q_t) * t_scale)**2 + ((closest_v - q_v) * v_scale)**2
        if dist_px_sq > r_sq:
            continue
        
        if split_dims[node] == -1: 
            start = seg_starts[node]
            count = seg_counts[node]
            for i in range(count):
                seg_idx = leaf_seg_indices[start + i]
                line_idx = int(segments[seg_idx, 5])
                
                if not matched_lines[line_idx]:
                    # Express segment endpoints in pixel offsets from query center.
                    x1 = (segments[seg_idx, 0] - q_t) * t_scale
                    y1 = (segments[seg_idx, 2] - q_v) * v_scale
                    x2 = (segments[seg_idx, 1] - q_t) * t_scale
                    y2 = (segments[seg_idx, 3] - q_v) * v_scale

                    if point_to_segment_dist_sq(0.0, 0.0, x1, y1, x2, y2) <= r_sq:
                        matched_lines[line_idx] = True
        else: 
            stack[stack_ptr] = right_children[node]
            stack_ptr += 1
            stack[stack_ptr] = left_children[node]
            stack_ptr += 1
            
    result = np.empty(n_lines, dtype=np.int32)
    count = 0
    for i in range(n_lines):
        if matched_lines[i]:
            result[count] = i
            count += 1
    return result[:count].copy()


@njit
def angular_query(
    tree_arrays, segments,
    t_min, t_max,
    target_angle_rad, angle_tol_rad,
    t_scale, v_scale,
    n_lines,
    strict=False
):
    """Select time series matching a visual slope angle over `[t_min, t_max]`.

    Angles are evaluated in canvas-pixel coordinates (where `+y` points
    downward) so the matched slopes visually align with the wedge drawn by the
    user regardless of axis scaling.

    Parameters
    ----------
    tree_arrays : tuple of np.ndarray
        KD-tree arrays returned by `build_kd_tree`.
    segments : np.ndarray of float32, shape (n_segments, 6)
        Segment table produced by `create_tvs_segments`.
    t_min, t_max : float
        Queried time interval in data coordinates.
    target_angle_rad : float
        Center slope angle in radians (canvas pixel coordinates; negative
        values correspond to visually upward slopes).
    angle_tol_rad : float
        Half-width angular tolerance in radians.
    t_scale, v_scale : float
        Pixels-per-data-unit conversion factors along Time and Value axes.
    n_lines : int
        Total number of time series in the dataset.
    strict : bool, default=False
        - `False`: evaluates each curve's net trend angle from its clipped
          entry point to its clipped exit point within `[t_min, t_max]`.
        - `True`: requires every individual micro-segment overlapping
          `(t_min, t_max)` to lie within `target_angle_rad ± angle_tol_rad`.

    Returns
    -------
    matched_indices : np.ndarray of int32, shape (n_matched,)
        Sorted unique line indices satisfying the angular constraint.
    """
    (
        split_dims, _, left_children, right_children,
        node_bounds, seg_starts, seg_counts, leaf_seg_indices,
    ) = tree_arrays
    in_omega = np.zeros(n_lines, dtype=np.bool_)
    is_invalid = np.zeros(n_lines, dtype=np.bool_)

    # Track clipped entry/exit endpoints per curve for net-trend evaluation.
    t_entry = np.full(n_lines, np.inf, dtype=np.float32)
    v_entry = np.zeros(n_lines, dtype=np.float32)
    t_exit = np.full(n_lines, -np.inf, dtype=np.float32)
    v_exit = np.zeros(n_lines, dtype=np.float32)
    
    stack = np.zeros(1000, dtype=np.int32)
    stack_ptr = 0
    stack[stack_ptr] = 0
    stack_ptr += 1
    
    while stack_ptr > 0:
        stack_ptr -= 1
        node = stack[stack_ptr]
        n_t_min, n_t_max = node_bounds[node, 0], node_bounds[node, 1]
        
        if n_t_max < t_min or n_t_min > t_max: 
            continue
            
        if split_dims[node] == -1:
            start = seg_starts[node]
            count = seg_counts[node]
            for i in range(count):
                seg_idx = leaf_seg_indices[start + i]
                line_idx = int(segments[seg_idx, 5])
                
                if strict and is_invalid[line_idx]:
                    continue
                    
                s_t1, s_t2 = segments[seg_idx, 0], segments[seg_idx, 1]
                s_v1, s_v2 = segments[seg_idx, 2], segments[seg_idx, 3]
                s_slope = segments[seg_idx, 4]

                # Require strict interior overlap with (t_min, t_max) so segments
                # merely touching an endpoint outside the window are ignored.
                if s_t2 > t_min and s_t1 < t_max:
                    if strict:
                        dx = (s_t2 - s_t1) * t_scale
                        dy = -(s_v2 - s_v1) * v_scale

                        if dx == 0:
                            seg_angle_rad = math.pi / 2.0 if dy > 0 else -math.pi / 2.0
                        else:
                            seg_angle_rad = math.atan(dy / dx)

                        if abs(seg_angle_rad - target_angle_rad) <= angle_tol_rad:
                            in_omega[line_idx] = True
                        else:
                            is_invalid[line_idx] = True
                    else:
                        # Linearly clip segment endpoints to [t_min, t_max].
                        c_t1 = max(t_min, s_t1)
                        c_v1 = s_v1 + (c_t1 - s_t1) * s_slope
                        c_t2 = min(t_max, s_t2)
                        c_v2 = s_v1 + (c_t2 - s_t1) * s_slope

                        if c_t1 < t_entry[line_idx]:
                            t_entry[line_idx] = c_t1
                            v_entry[line_idx] = c_v1
                        if c_t2 > t_exit[line_idx]:
                            t_exit[line_idx] = c_t2
                            v_exit[line_idx] = c_v2
                        in_omega[line_idx] = True
        else:
            stack[stack_ptr] = right_children[node]
            stack_ptr += 1
            stack[stack_ptr] = left_children[node]
            stack_ptr += 1
            
    result = np.empty(n_lines, dtype=np.int32)
    count = 0
    for i in range(n_lines):
        if not in_omega[i]:
            continue
        if strict:
            if not is_invalid[i]:
                result[count] = i
                count += 1
        else:
            dt = t_exit[i] - t_entry[i]
            if dt > 1e-7:
                dx = dt * t_scale
                dy = -(v_exit[i] - v_entry[i]) * v_scale
                line_angle_rad = math.atan(dy / dx)
                if abs(line_angle_rad - target_angle_rad) <= angle_tol_rad:
                    result[count] = i
                    count += 1
    return result[:count].copy()

@njit
def timebox_query(
    tree_arrays, segments,
    t_min, t_max,
    v_min, v_max,
    n_lines,
    strict=False,
    eps=-1.0,
):
    """Execute a rectangular Brush / Timebox query over `[t_min, t_max] x [v_min, v_max]`.

    Parameters
    ----------
    tree_arrays : tuple of np.ndarray
        KD-tree arrays returned by `build_kd_tree`.
    segments : np.ndarray of float32, shape (n_segments, 6)
        Segment table produced by `create_tvs_segments`.
    t_min, t_max, v_min, v_max : float
        Query rectangle bounds in data coordinates.
    n_lines : int
        Total number of time series in the dataset.
    strict : bool, default=False
        - `False` (Box Intersect): selects every curve that intersects the
          query rectangle at any point.
        - `True` (Strict Timebox): enforces top and bottom ghost boundary
          boxes (Zhao et al., Sec. 4.3) so only curves staying within
          `[v_min, v_max]` across the entire `[t_min, t_max]` span qualify.
    eps : float, default=-1.0
        Margin separating ghost boxes from `v_min`/`v_max`. When negative,
        computed automatically as `max(1e-7, |v_max - v_min| * 1e-5)` for
        scale invariance.

    Returns
    -------
    matched_indices : np.ndarray of int32, shape (n_matched,)
        Sorted unique line indices matching the query.
    """
    (
        split_dims, _, left_children, right_children,
        node_bounds, seg_starts, seg_counts, leaf_seg_indices,
    ) = tree_arrays

    in_omega = np.zeros(n_lines, dtype=np.bool_)
    is_invalid = np.zeros(n_lines, dtype=np.bool_)
    eff_eps = eps if eps >= 0.0 else max(1e-7, abs(v_max - v_min) * 1e-5)

    stack = np.zeros(1000, dtype=np.int32)
    stack_ptr = 0
    stack[stack_ptr] = 0
    stack_ptr += 1
    
    while stack_ptr > 0:
        stack_ptr -= 1
        node = stack[stack_ptr]
        
        n_t_min, n_t_max = node_bounds[node, 0], node_bounds[node, 1]
        
        # Prune if the node's bounding box is strictly outside the time interval:
        # we don't prune on the Value dimension here because lines that are 
        # way above/below the box need to trigger the ghost boundary invalidation.
        if n_t_max < t_min or n_t_min > t_max: 
            continue

        # In non-strict mode we can safely prune on Value as well. In strict
        # mode we must visit nodes above/below the box to detect ghost-box exits.
        if not strict:
            n_v_min, n_v_max = node_bounds[node, 2], node_bounds[node, 3]
            if n_v_max < v_min or n_v_min > v_max:
                continue
        
        if split_dims[node] == -1: # Leaf Node
            start = seg_starts[node]
            count = seg_counts[node]
            for i in range(count):
                seg_idx = leaf_seg_indices[start + i]
                line_idx = int(segments[seg_idx, 5])
                
                if strict and is_invalid[line_idx]: 
                    continue
                if not strict and in_omega[line_idx]:
                    continue
                    
                x1, x2 = segments[seg_idx, 0], segments[seg_idx, 1]
                y1, y2 = segments[seg_idx, 2], segments[seg_idx, 3]
                
                if strict:
                    # Top and bottom ghost boxes extend to +/-1e30 (float32-safe).
                    if segment_intersects_aabb(x1, y1, x2, y2, t_min, t_max, v_max + eff_eps, 1e30):
                        is_invalid[line_idx] = True
                        continue
                    if segment_intersects_aabb(x1, y1, x2, y2, t_min, t_max, -1e30, v_min - eff_eps):
                        is_invalid[line_idx] = True
                        continue
                    
                # Check Main Box.
                if not in_omega[line_idx]:
                    if segment_intersects_aabb(x1, y1, x2, y2, t_min, t_max, v_min, v_max):
                        in_omega[line_idx] = True
                        
        else: # Internal Node
            stack[stack_ptr] = right_children[node]
            stack_ptr += 1
            stack[stack_ptr] = left_children[node]
            stack_ptr += 1

    result = np.empty(n_lines, dtype=np.int32)
    count = 0
    for i in range(n_lines):
        if in_omega[i] and (not strict or not is_invalid[i]):
            result[count] = i
            count += 1
    return result[:count].copy()


# =============================================================================
# Density Rasterization & Representative Line Extraction
# =============================================================================


@njit
def rasterize_density(
    segments,
    active_lines,
    width, height,
    t_min, t_max,
    v_min, v_max,
    n_steps,
):
    """Rasterize active time-series curves onto a 2D pixel density grid.

    Uses a Digital Differential Analyzer (DDA) line-drawing pass over every
    segment belonging to `active_lines`.

    Parameters
    ----------
    segments : np.ndarray of float32, shape (n_segments, 6)
        Segment table produced by `create_tvs_segments`.
    active_lines : np.ndarray of int32, shape (n_active,)
        Indices of series to include in the density accumulation.
    width, height : int
        Output image dimensions in pixels.
    t_min, t_max, v_min, v_max : float
        Global data bounds mapped onto the canvas.
    n_steps : int
        Number of sample points per series.

    Returns
    -------
    grid : np.ndarray of int32, shape (height, width)
        Per-pixel line-crossing counts in top-down image coordinates (row 0
        corresponds to `v_max`).
    """
    grid = np.zeros((height, width), dtype=np.int32)
    
    # Prevent division by zero.
    t_diff = max(1e-5, t_max - t_min)
    v_diff = max(1e-5, v_max - v_min)
    
    t_scale = width / t_diff
    v_scale = height / v_diff
    
    for i in range(len(active_lines)):
        line_idx = active_lines[i]
        seg_start = line_idx * (n_steps - 1)
        for s in range(n_steps - 1):
            seg = segments[seg_start + s]

            # Clamp coordinates into [0, width-1] / [0, height-1] so exact
            # boundary samples (t == t_max or v == v_max) are never dropped.
            x1 = min(width - 1, max(0, int((seg[0] - t_min) * t_scale)))
            y1 = min(height - 1, max(0, height - 1 - int((seg[2] - v_min) * v_scale)))
            x2 = min(width - 1, max(0, int((seg[1] - t_min) * t_scale)))
            y2 = min(height - 1, max(0, height - 1 - int((seg[3] - v_min) * v_scale)))

            dx = x2 - x1
            dy = y2 - y1
            steps = max(abs(dx), abs(dy))
            
            if steps == 0:
                grid[y1, x1] += 1
                continue
                
            x_inc = dx / steps
            y_inc = dy / steps
            x, y = float(x1), float(y1)
            for _ in range(steps + 1):
                ix, iy = int(x), int(y)
                if 0 <= ix < width and 0 <= iy < height:
                    grid[iy, ix] += 1
                x += x_inc
                y += y_inc
    return grid

@njit
def get_representative_lines(
    segments, active_lines, grid,
    width, height,
    t_min, t_max, v_min, v_max,
    n_steps,
    num_reps=3,
    diversity_threshold=0.1,
    use_meanshift=False
):
    """Select representative time series capturing major trends in `active_lines`.

    Every active series is first scored by its mean pixel density along `grid`.
    Representative curves are then selected via one of two methods:

    1. **Greedy Heuristic (`use_meanshift=False`):** Picks the densest curve
       first, then iteratively accepts the next densest candidate whose mean
       absolute slope difference from all previously chosen curves is at least
       `diversity_threshold`.
    2. **Density-Weighted Curve Mean-Shift (`use_meanshift=True`):** Samples
       a density-stratified candidate pool (up to 250 curves), runs Gaussian
       Mean-Shift in normalized value space from farthest-point seeds, merges
       converged modes within a diversity-controlled radius, and snaps each
       unique mode to its nearest real dataset curve (medoid).

    Parameters
    ----------
    segments : np.ndarray of float32, shape (n_segments, 6)
        Segment table produced by `create_tvs_segments`.
    active_lines : np.ndarray of int32, shape (n_active,)
        Indices of currently active series.
    grid : np.ndarray of int32, shape (height, width)
        Density grid produced by `rasterize_density`.
    width, height : int
        Canvas dimensions in pixels.
    t_min, t_max, v_min, v_max : float
        Global data bounds mapped onto the canvas.
    n_steps : int
        Number of sample points per series.
    num_reps : int, default=3
        Maximum number of representative curves to return.
    diversity_threshold : float, default=0.1
        Minimum separation between chosen representative curves (mean slope
        difference for Heuristic; bandwidth and mode-merge radius control for
        Mean-Shift).
    use_meanshift : bool, default=False
        If `True`, use Curve Mean-Shift clustering; otherwise use the greedy
        density-plus-slope heuristic.

    Returns
    -------
    reps : np.ndarray of int32, shape (n_selected,)
        Indices of the chosen representative series (`n_selected <= num_reps`).
    """
    n_active = len(active_lines)
    if n_active == 0 or num_reps <= 0:
        return np.zeros(0, dtype=np.int32)
        
    scores = np.zeros(n_active, dtype=np.float32)
    
    t_diff = max(1e-5, t_max - t_min)
    v_diff = max(1e-5, v_max - v_min)
    t_scale = width / t_diff
    v_scale = height / v_diff

    # Score each active curve by average pixel density along its path.
    for idx in range(n_active):
        line_idx = active_lines[idx]
        seg_start = line_idx * (n_steps - 1)
        
        total_density = 0.0
        total_pixels = 0
        
        for s in range(n_steps - 1):
            seg = segments[seg_start + s]
            x1 = min(width - 1, max(0, int((seg[0] - t_min) * t_scale)))
            y1 = min(height - 1, max(0, height - 1 - int((seg[2] - v_min) * v_scale)))
            x2 = min(width - 1, max(0, int((seg[1] - t_min) * t_scale)))
            y2 = min(height - 1, max(0, height - 1 - int((seg[3] - v_min) * v_scale)))

            dx = x2 - x1
            dy = y2 - y1
            steps = max(abs(dx), abs(dy))
            
            if steps == 0:
                total_density += grid[y1, x1]
                total_pixels += 1
                continue
                
            x_inc = dx / steps
            y_inc = dy / steps
            x, y = float(x1), float(y1)
            for _ in range(steps + 1):
                ix, iy = int(x), int(y)
                if 0 <= ix < width and 0 <= iy < height:
                    total_density += grid[iy, ix]
                    total_pixels += 1
                x += x_inc
                y += y_inc
        if total_pixels > 0:
            scores[idx] = total_density / total_pixels
            
    # Sort lines by density score descending.
    sorted_indices = np.argsort(-scores)

    # =========================================================================
    # Method 1: Greedy Density + Slope Heuristic
    # =========================================================================
    if not use_meanshift:
        reps = np.empty(num_reps, dtype=np.int32)
        rep_count = 0

        for idx in range(n_active):
            candidate_idx = active_lines[sorted_indices[idx]]

            if rep_count == 0:
                reps[0] = candidate_idx
                rep_count += 1
                if rep_count >= num_reps:
                    break
                continue

            is_diverse = True
            c_seg_start = candidate_idx * (n_steps - 1)

            for r in range(rep_count):
                rep_idx = reps[r]
                r_seg_start = rep_idx * (n_steps - 1)
                shape_diff = 0.0

                for s in range(n_steps - 1):
                    c_slope = segments[c_seg_start + s, 4]
                    r_slope = segments[r_seg_start + s, 4]
                    shape_diff += abs(c_slope - r_slope)
                shape_diff /= n_steps - 1

                if shape_diff < diversity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                reps[rep_count] = candidate_idx
                rep_count += 1

            if rep_count >= num_reps:
                break

        return reps[:rep_count]

    # =========================================================================
    # Method 2: Density-Weighted Curve Mean-Shift (Medoid Snapping)
    # =========================================================================
    n_segs = n_steps - 1
    # Stratify across the top 95% of density-ranked series (capped at K=250) so
    # secondary trajectory bundles remain represented while keeping latency <3ms.
    pool_limit = max(num_reps, int(n_active * 0.95))
    if pool_limit > n_active:
        pool_limit = n_active
    K = min(pool_limit, 250)
    
    cand_indices = np.empty(K, dtype=np.int32)
    cand_weights = np.empty(K, dtype=np.float32)
    cand_features = np.empty((K, n_segs), dtype=np.float32)
    
    for k in range(K):
        s_pos = int((k * (pool_limit - 1)) / max(1, K - 1)) if K > 1 else 0
        act_pos = sorted_indices[s_pos]
        line_idx = active_lines[act_pos]
        cand_indices[k] = line_idx
        cand_weights[k] = max(1e-3, scores[act_pos])
        seg_start = line_idx * n_segs
        for s in range(n_segs):
            v_mid = 0.5 * (segments[seg_start + s, 2] + segments[seg_start + s, 3])
            cand_features[k, s] = (v_mid - v_min) / v_diff

    # Map user diversity_threshold to Gaussian kernel bandwidth h and merge radius.
    div_clamped = max(0.0, diversity_threshold)
    h = max(0.02, min(0.18, 0.035 + 0.02 * div_clamped))
    merge_dist = 0.075 * math.sqrt(div_clamped)

    # Farthest-point seeding ensures seeds cover distinct regions of curve space.
    M = min(K, max(num_reps * 5, 30))
    seed_pool_idx = np.empty(M, dtype=np.int32)
    seed_pool_idx[0] = 0
    min_dists = np.full(K, np.inf, dtype=np.float32)
    
    for m in range(1, M):
        last_idx = seed_pool_idx[m - 1]
        best_k = 0
        best_val = -1.0
        for k in range(K):
            d = 0.0
            for s in range(n_segs):
                d += abs(cand_features[k, s] - cand_features[last_idx, s])
            d /= n_segs
            if d < min_dists[k]:
                min_dists[k] = d
            if min_dists[k] > best_val:
                best_val = min_dists[k]
                best_k = k
        seed_pool_idx[m] = best_k

    # Iteratively shift each seed toward its density-weighted Gaussian mean.
    modes = np.empty((M, n_segs), dtype=np.float32)
    mode_mass = np.zeros(M, dtype=np.float32)
    inv_2h2 = 1.0 / (2.0 * h * h)
    
    for m in range(M):
        s_idx = seed_pool_idx[m]
        for s in range(n_segs):
            modes[m, s] = cand_features[s_idx, s]
            
        for _ in range(10):
            new_mode = np.zeros(n_segs, dtype=np.float32)
            weight_sum = 0.0
            for k in range(K):
                dist_sq = 0.0
                for s in range(n_segs):
                    diff = modes[m, s] - cand_features[k, s]
                    dist_sq += diff * diff
                dist_sq /= n_segs
                
                kern = cand_weights[k] * math.exp(-dist_sq * inv_2h2)
                weight_sum += kern
                for s in range(n_segs):
                    new_mode[s] += kern * cand_features[k, s]
                    
            if weight_sum > 1e-9:
                shift = 0.0
                for s in range(n_segs):
                    updated = new_mode[s] / weight_sum
                    shift += abs(updated - modes[m, s])
                    modes[m, s] = updated
                mode_mass[m] = weight_sum
                if (shift / n_segs) < 1e-4:
                    break

    # Deduplicate converged modes by kernel mass and snap each to its medoid.
    mode_order = np.argsort(-mode_mass)
    reps = np.empty(num_reps, dtype=np.int32)
    accepted_modes = np.empty((num_reps, n_segs), dtype=np.float32)
    used_cand = np.zeros(K, dtype=np.bool_)
    rep_count = 0
    
    for idx in range(M):
        m = mode_order[idx]
        
        best_k = -1
        best_dist = np.inf
        for k in range(K):
            if used_cand[k]:
                continue
            d = 0.0
            for s in range(n_segs):
                d += abs(cand_features[k, s] - modes[m, s])
            d /= n_segs
            if d < best_dist:
                best_dist = d
                best_k = k
                
        if best_k == -1:
            continue
            
        is_distinct = True
        for r in range(rep_count):
            d_mode = 0.0
            for s in range(n_segs):
                d_mode += abs(modes[m, s] - accepted_modes[r, s])
            d_mode /= n_segs
            if d_mode < merge_dist:
                is_distinct = False
                break
                
        if not is_distinct:
            continue
            
        used_cand[best_k] = True
        reps[rep_count] = cand_indices[best_k]
        for s in range(n_segs):
            accepted_modes[rep_count, s] = modes[m, s]
        rep_count += 1
        if rep_count >= num_reps:
            break
            
    return reps[:rep_count]
