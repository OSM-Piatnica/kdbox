/**
 * @fileoverview Interactive HTML5 Canvas frontend for KD-Box (anywidget ESM).
 *
 * Rendering uses four stacked HTML5 Canvas layers to separate update frequencies:
 *   axes-layer        --- Static coordinate frame, ticks, and optional X/Y labels.
 *   density-layer     --- Client-side colormapped density field (gamma + opacity).
 *   rep-layer         --- Representative curves overlaid on top of the density field.
 *   interaction-layer --- Draggable/resizable query shapes (Brush, RNL, Angular, Pin)
 *                         and live coordinate crosshair with white text halo.
 */

/** Perceptually uniform RGB colormap stops sampled linearly in [0, 255]. */
const COLORMAPS = {
    viridis: [
        [68, 1, 84], [72, 37, 117], [65, 68, 135], [53, 96, 141],
        [42, 120, 142], [33, 145, 141], [34, 168, 132], [67, 191, 113],
        [122, 209, 81], [188, 223, 39], [253, 231, 37],
    ],
    magma: [
        [0, 0, 4], [21, 14, 55], [59, 15, 112], [101, 26, 128],
        [140, 41, 129], [182, 55, 122], [222, 73, 104], [247, 111, 92],
        [254, 159, 109], [254, 206, 145], [252, 253, 191],
    ]
};

/** Distinct palette cycled across simultaneous spatial query shapes. */
const QUERY_COLORS = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4"];

/** High-contrast palette for representative line overlays. */
const REP_LINE_COLORS = ["#00FF00", "#00FFFF", "#FF00FF", "#FFFF00", "#FF0000"];

/**
 * Interpolate an RGBA color for a grayscale pixel intensity.
 *
 * @param {number} intensity - Pixel intensity in [0, 255] after gamma adjustment.
 * @param {string} cmapName  - Key in {@link COLORMAPS} ("magma" or "viridis").
 * @returns {[number, number, number, number]} RGBA tuple in [0, 255].
 */
function getColor(intensity, cmapName) {
    if (intensity === 0) return [0, 0, 0, 0]; // Transparent background
    const cmap = COLORMAPS[cmapName] || COLORMAPS.magma;
    const t = (intensity / 255) * (cmap.length - 1);
    const idx = Math.floor(t);
    if (idx >= cmap.length - 1) return [...cmap[cmap.length - 1], 255];
    
    const c1 = cmap[idx];
    const c2 = cmap[idx + 1];
    const f = t - idx;
    return [
        Math.round(c1[0] + (c2[0] - c1[0]) * f),
        Math.round(c1[1] + (c2[1] - c1[1]) * f),
        Math.round(c1[2] + (c2[2] - c1[2]) * f),
        255
    ];
}

/**
 * Parse an opacity input safely into [0.0, 1.0], defaulting to 1.0 if blank/NaN.
 *
 * @param {string|number} rawVal - Raw value from a slider or numeric input.
 * @returns {number} Clamped opacity in [0.0, 1.0].
 */
function safeOpacity(rawVal) {
    const parsed = parseFloat(rawVal);
    return Number.isFinite(parsed) ? Math.min(1.0, Math.max(0.0, parsed)) : 1.0;
}

/**
 * anywidget entry point invoked when a KDBoxWidget view is mounted in Jupyter.
 *
 * @param {{ model: any, el: HTMLElement }} context - Synced model and root DOM element.
 */
export function render({ model, el }) {
    const width = model.get("width");
    const height = model.get("height");

    // Isolate radio button groups per widget instance so stacked views in
    // MultivariateKDBox never interfere with one another.
    const widgetGroupId = "grp_" + Math.random().toString(36).substr(2, 9);
    
    const numInputStyle = "width: 68px; padding: 2px 4px; border: 1px solid #ccc; border-radius: 3px; text-align: right; font-size: 13px;";
    const labelRowStyle = "display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;";

    // Margins reserved around the data canvas for tick labels and optional axis titles.
    const PAD_LEFT = 65;
    const PAD_TOP = 24;
    const PAD_RIGHT = 20;
    const PAD_BOTTOM = 45;

    el.innerHTML = `
        <div style="display: flex; flex-direction: column; font-family: sans-serif; background: #f5f7f9; border: 1px solid #d7dde4; border-radius: 4px; padding: 10px; overflow-x: auto; max-width: 100%;">
            
            <!-- Controls and Canvas Row -->
            <div style="display: flex; min-width: max-content;">

                <!-- Controls Sidebar -->
                <div style="width: 250px; padding-right: 20px; display: flex; flex-direction: column; gap: 10px;">
                    <h3 style="margin: 0; color: #515a6e;">KD-Box Controls</h3>
                    
                    <label><b>Tool:</b>
                        <select id="tool-select" style="width: 100%; padding: 4px; margin-top: 2px;">
                            <option value="brush">Brush (Timebox)</option>
                            <option value="rnl">RNL (Radius)</option>
                            <option value="ang">Angular</option>
                            <option value="pin">Pin (Hover)</option>
                        </select>
                    </label>

                    <label><b>Colormap:</b>
                        <select id="cmap-select" style="width: 100%; padding: 4px; margin-top: 2px;">
                            <option value="magma">Magma</option>
                            <option value="viridis">Viridis</option>
                        </select>
                    </label>
                    
                    <label style="display: flex; align-items: center; gap: 8px; cursor: pointer;">
                        <input type="checkbox" id="norm-check" checked> <b>Normalize density</b>
                    </label>
                    
                    <label id="strict-label" style="display: flex; align-items: center; gap: 8px; cursor: pointer;" title="When unchecked: Brush selects any line intersecting the box. When checked: lines must remain inside the box for the entire brushed time range.">
                        <input type="checkbox" id="strict-check"> <b>Strict query mode</b>
                    </label>
                    
                    <!-- Dynamic Categorical Attribute Filters (hidden if no attributes passed) -->
                    <div id="attr-section" style="display: none; flex-direction: column; gap: 8px;">
                        <hr style="width: 100%; border: 0; border-top: 1px solid #ccc; margin: 0;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <b style="color: #515a6e; font-size: 13px;">Attribute Filters</b>
                            <button id="reset-attr-btn" style="cursor: pointer; padding: 1px 6px; font-size: 11px; color: #515a6e;">Reset</button>
                        </div>
                        <div id="attr-controls-list" style="display: flex; flex-direction: column; gap: 8px; max-height: 165px; overflow-y: auto; padding-right: 2px;"></div>
                    </div>
                    
                    <hr style="width: 100%; border: 0; border-top: 1px solid #ccc; margin: 0;">

                    <div>
                        <div style="${labelRowStyle}">
                            <b>Contrast (Gamma):</b>
                            <input type="number" id="gamma-num" min="0.05" max="10.0" step="0.05" value="1.0" style="${numInputStyle}">
                        </div>
                        <input type="range" id="gamma-slider" min="0.1" max="3.0" step="0.05" value="1.0" style="width: 100%;">
                    </div>

                    <div>
                        <div style="${labelRowStyle}">
                            <b>Data Opacity:</b>
                            <input type="number" id="data-op-num" min="0" max="1" step="0.05" value="1.0" style="${numInputStyle}">
                        </div>
                        <input type="range" id="data-op-slider" min="0" max="1" step="0.01" value="1.0" style="width: 100%;">
                    </div>

                    <div>
                        <div style="${labelRowStyle}">
                            <b>Rep Lines Opacity:</b>
                            <input type="number" id="rep-op-num" min="0" max="1" step="0.05" value="1.0" style="${numInputStyle}">
                        </div>
                        <input type="range" id="rep-op-slider" min="0" max="1" step="0.01" value="1.0" style="width: 100%;">
                    </div>
                    
                    <hr style="width: 100%; border: 0; border-top: 1px solid #ccc; margin: 0;">

                    <label><b>Rep Lines Method:</b>
                        <select id="rep-method-select" style="width: 100%; padding: 4px; margin-top: 2px;">
                            <option value="heuristic">Heuristic (Density + Slope)</option>
                            <option value="meanshift">Mean-Shift (Curve Modes)</option>
                        </select>
                    </label>
                    
                    <div>
                        <div style="${labelRowStyle}">
                            <b>Rep Lines Count:</b>
                            <input type="number" id="rep-num" min="1" max="50" step="1" value="3" style="${numInputStyle}">
                        </div>
                        <input type="range" id="rep-slider" min="1" max="10" step="1" value="3" style="width: 100%;">
                    </div>
                    
                    <div>
                        <div style="${labelRowStyle}">
                            <b>Diversity:</b>
                            <input type="number" id="div-num" min="0" max="100" step="0.01" value="0.1" style="${numInputStyle}">
                        </div>
                        <input type="range" id="div-slider" min="0" max="10" step="0.05" value="0.1" style="width: 100%;">
                    </div>
                </div> <!-- End Controls Sidebar -->

                <!-- Canvas Column -->
                <div style="padding-left: ${PAD_LEFT}px; padding-top: ${PAD_TOP}px; padding-bottom: ${PAD_BOTTOM}px; padding-right: ${PAD_RIGHT}px; position: relative;">
                    <canvas id="axes-layer" width="${width + PAD_LEFT + PAD_RIGHT}" height="${height + PAD_TOP + PAD_BOTTOM}" style="position: absolute; left: 0; top: 0; pointer-events: none;"></canvas>
                    <div id="canvas-container" style="position: relative; width: ${width}px; height: ${height}px; background: white; border: 1px solid #ccc; cursor: crosshair;">
                        <canvas id="density-layer" width="${width}" height="${height}" style="position: absolute; top: 0; left: 0;"></canvas>
                        <canvas id="rep-layer" width="${width}" height="${height}" style="position: absolute; top: 0; left: 0; pointer-events: none;"></canvas>
                        <canvas id="interaction-layer" width="${width}" height="${height}" style="position: absolute; top: 0; left: 0;"></canvas>
                    </div>
                </div> <!-- End Canvas Column -->
                
            </div> <!-- End Controls and Canvas Row -->
            
            <!-- Dynamic Information Table Row -->
            <div style="margin-top: 15px; background: white; padding: 10px; border: 1px solid #ccc; border-radius: 4px; min-width: max-content; clear: both;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <b style="color: #515a6e;">Information Table</b>
                    <button id="clear-all-btn" style="cursor: pointer; padding: 4px 8px; color: red;">Delete All Queries</button>
                </div>
                <table style="width: 100%; text-align: center; border-collapse: collapse; font-size: 14px; color: #515a6e;">
                    <thead>
                        <tr style="border-bottom: 2px solid #e8eaec;">
                            <th style="padding: 8px;">Explore</th>
                            <th style="padding: 8px; text-align: left;">Query</th>
                            <th style="padding: 8px;">Line Count</th>
                            <th style="padding: 8px;">Time Points</th>
                            <th style="padding: 8px;">Min Value</th>
                            <th style="padding: 8px;">Max Value</th>
                            <th style="padding: 8px;">Mean</th>
                            <th style="padding: 8px;">Variance</th>
                            <th style="padding: 8px;">Action</th>
                        </tr>
                    </thead>
                    <tbody id="table-body"></tbody>
                </table>
            </div> <!-- End Information Table Row -->
            
        </div>
    `;

    // Acquire 2D contexts for the four visible layers plus an offscreen buffer
    // holding the raw grayscale density image for fast client-side re-colormapping.
    const densityCtx = el.querySelector("#density-layer").getContext("2d");
    const repCtx = el.querySelector("#rep-layer").getContext("2d");
    const interactionCtx = el.querySelector("#interaction-layer").getContext("2d");
    const axesCtx = el.querySelector("#axes-layer").getContext("2d");
    
    const hiddenCanvas = document.createElement("canvas");
    hiddenCanvas.width = width;
    hiddenCanvas.height = height;
    const hiddenCtx = hiddenCanvas.getContext("2d", { willReadFrequently: true });

    // Restore persisted spatial queries if the widget view is re-mounted.
    let queries = [];
    try {
        const initialQueries = JSON.parse(model.get("query_data") || "[]");
        if (Array.isArray(initialQueries)) queries = initialQueries;
    } catch (e) {
        queries = [];
    }

    // =========================================================================
    // Categorical Attribute Filters Manager
    // =========================================================================

    const attrSection = el.querySelector("#attr-section");
    const attrListEl = el.querySelector("#attr-controls-list");
    const resetAttrBtn = el.querySelector("#reset-attr-btn");

    /** Per-attribute search strings retained across re-renders. */
    const attrSearchTerms = {};
    /** Show a live search input when an attribute exceeds this cardinality. */
    const SEARCH_THRESHOLD = 12;
    /** Cap rendered <option> elements to keep high-cardinality dropdowns snappy. */
    const MAX_RENDERED_OPTIONS = 100;

    /**
     * Parse the currently synced JSON attribute filter dictionary.
     * @returns {Record<string, string[]>}
     */
    function getActiveAttrFilters() {
        try {
            return JSON.parse(model.get("attr_filters") || "{}");
        } catch (e) {
            return {};
        }
    }

    /**
     * Prune empty filter lists and sync updated attribute filters to Python.
     * @param {Record<string, string[]>} filtersObj
     */
    function saveAttrFilters(filtersObj) {
        const cleaned = {};
        Object.keys(filtersObj).forEach(k => {
            if (Array.isArray(filtersObj[k]) && filtersObj[k].length > 0) {
                cleaned[k] = filtersObj[k];
            }
        });
        model.set("attr_filters", JSON.stringify(cleaned));
        model.save_changes();
        renderAttributeControls();
    }

    /**
     * Populate an attribute <select> dropdown with search-filtered options.
     *
     * @param {HTMLSelectElement} selectEl
     * @param {string} attrName
     * @param {Array<{value: string, count: number}>} opts
     * @param {string[]} selectedVals
     * @param {string} searchTerm
     */
    function populateAttrSelect(selectEl, attrName, opts, selectedVals, searchTerm) {
        selectEl.innerHTML = "";
        const totalCount = opts.reduce((acc, o) => acc + o.count, 0);
        const term = (searchTerm || "").trim().toLowerCase();
        const filtered = term
            ? opts.filter(o => o.value.toLowerCase().includes(term))
            : opts;

        if (selectedVals.length === 0) {
            const allOpt = document.createElement("option");
            allOpt.value = "__ALL__";
            allOpt.innerText = term
                ? `Matches: ${filtered.length} / ${opts.length} (select to filter)`
                : `All (${totalCount})`;
            allOpt.selected = true;
            selectEl.appendChild(allOpt);
        } else {
            const phOpt = document.createElement("option");
            phOpt.value = "";
            phOpt.innerText = term
                ? `+ Add (${filtered.length} matches, ${selectedVals.length} active)...`
                : `+ Add ${attrName} (${selectedVals.length} active)...`;
            phOpt.selected = true;
            selectEl.appendChild(phOpt);

            const resetOpt = document.createElement("option");
            resetOpt.value = "__ALL__";
            resetOpt.innerText = `All / Clear (${totalCount})`;
            selectEl.appendChild(resetOpt);
        }

        const visibleOpts = filtered.slice(0, MAX_RENDERED_OPTIONS);
        visibleOpts.forEach(o => {
            const optEl = document.createElement("option");
            optEl.value = o.value;
            const isSel = selectedVals.includes(o.value);
            optEl.innerText = `${isSel ? "✓ " : ""}${o.value} (${o.count})`;
            selectEl.appendChild(optEl);
        });

        const remaining = filtered.length - visibleOpts.length;
        if (remaining > 0) {
            const moreOpt = document.createElement("option");
            moreOpt.value = "";
            moreOpt.disabled = true;
            moreOpt.innerText = `... and ${remaining} more (refine search)`;
            selectEl.appendChild(moreOpt);
        }
    }

    /** Render (or hide) the dynamic Attribute Filters section in the sidebar. */
    function renderAttributeControls() {
        const attrOptions = model.get("attr_options") || {};
        const attrNames = Object.keys(attrOptions);
        const showControls = model.get("show_attr_controls") !== false;

        if (!showControls || attrNames.length === 0) {
            attrSection.style.display = "none";
            return;
        }
        attrSection.style.display = "flex";
        const activeFilters = getActiveAttrFilters();
        attrListEl.innerHTML = "";

        attrNames.forEach(attrName => {
            const opts = attrOptions[attrName] || [];
            const selectedVals = activeFilters[attrName] || [];

            const wrapper = document.createElement("div");
            wrapper.style.cssText = "display: flex; flex-direction: column; gap: 4px;";

            const labelEl = document.createElement("div");
            labelEl.style.cssText = "font-size: 12px; font-weight: bold;";
            labelEl.innerText = `${attrName}:`;
            wrapper.appendChild(labelEl);

            const selectEl = document.createElement("select");
            selectEl.style.cssText = "width: 100%; padding: 3px; font-size: 12px;";

            // Render search filter box for high-cardinality attributes.
            if (opts.length > SEARCH_THRESHOLD) {
                const searchInput = document.createElement("input");
                searchInput.type = "text";
                searchInput.placeholder = `Search ${attrName} (${opts.length})...`;
                searchInput.value = attrSearchTerms[attrName] || "";
                searchInput.style.cssText = "width: 100%; box-sizing: border-box; padding: 3px 5px; font-size: 11px; border: 1px solid #cbd5e1; border-radius: 3px;";
                searchInput.addEventListener("input", (e) => {
                    attrSearchTerms[attrName] = e.target.value;
                    populateAttrSelect(selectEl, attrName, opts, selectedVals, e.target.value);
                });
                wrapper.appendChild(searchInput);
            }

            populateAttrSelect(selectEl, attrName, opts, selectedVals, attrSearchTerms[attrName] || "");

            selectEl.addEventListener("change", (e) => {
                const val = e.target.value;
                const curr = getActiveAttrFilters();
                const list = curr[attrName] ? [...curr[attrName]] : [];
                if (val === "__ALL__") {
                    delete curr[attrName];
                    delete attrSearchTerms[attrName];
                } else if (val !== "") {
                    const idx = list.indexOf(val);
                    if (idx >= 0) {
                        list.splice(idx, 1);
                    } else {
                        list.push(val);
                    }
                    curr[attrName] = list;
                }
                saveAttrFilters(curr);
            });
            wrapper.appendChild(selectEl);

            // Render removable pill badges for active selections.
            if (selectedVals.length > 0) {
                const pillsDiv = document.createElement("div");
                pillsDiv.style.cssText = "display: flex; flex-wrap: wrap; gap: 4px;";

                selectedVals.forEach(val => {
                    const pill = document.createElement("span");
                    pill.style.cssText = "background: #e1e8f0; color: #334155; font-size: 11px; padding: 2px 6px; border-radius: 10px; cursor: pointer; display: inline-flex; align-items: center; gap: 4px; border: 1px solid #cbd5e1;";
                    pill.title = "Click to remove filter";
                    pill.innerText = `${val} ×`;
                    pill.addEventListener("click", () => {
                        const curr = getActiveAttrFilters();
                        curr[attrName] = (curr[attrName] || []).filter(v => v !== val);
                        saveAttrFilters(curr);
                    });
                    pillsDiv.appendChild(pill);
                });
                wrapper.appendChild(pillsDiv);
            }

            attrListEl.appendChild(wrapper);
        });
    }

    resetAttrBtn.addEventListener("click", () => {
        Object.keys(attrSearchTerms).forEach(k => delete attrSearchTerms[k]);
        saveAttrFilters({});
    });

    model.on("change:attr_options", renderAttributeControls);
    model.on("change:attr_filters", renderAttributeControls);
    model.on("change:show_attr_controls", renderAttributeControls);

    // =========================================================================
    // Sidebar Controls and Two-Way Model Synchronization
    // =========================================================================

    const toolSelect = el.querySelector("#tool-select");
    const cmapSelect = el.querySelector("#cmap-select");
    const normCheck = el.querySelector("#norm-check");
    const strictLabel = el.querySelector("#strict-label");
    const strictCheck = el.querySelector("#strict-check");
    const repMethodSelect = el.querySelector("#rep-method-select");
    const repSlider = el.querySelector("#rep-slider");
    const repNum = el.querySelector("#rep-num");
    const divSlider = el.querySelector("#div-slider");
    const divNum = el.querySelector("#div-num");
    const gammaSlider = el.querySelector("#gamma-slider");
    const gammaNum = el.querySelector("#gamma-num");
    const dataOpSlider = el.querySelector("#data-op-slider");
    const dataOpNum = el.querySelector("#data-op-num");
    const repOpSlider = el.querySelector("#rep-op-slider");
    const repOpNum = el.querySelector("#rep-op-num");

    function updateStrictVisibility() {
        strictLabel.style.display = toolSelect.value === "brush" ? "flex" : "none";
    }

    function syncControlsFromModel() {
        toolSelect.value = model.get("tool_mode") || "brush";
        cmapSelect.value = model.get("colormap") || "magma";
        normCheck.checked = Boolean(model.get("normalize_density"));
        strictCheck.checked = Boolean(model.get("strict_mode"));
        repMethodSelect.value = model.get("rep_method") || "heuristic";

        const rc = Math.max(1, parseInt(model.get("rep_count"), 10) || 3);
        repNum.value = rc;
        repSlider.value = Math.min(10, rc);

        const dv = Math.max(0.0, parseFloat(model.get("diversity")) || 0.0);
        divNum.value = dv;
        divSlider.value = Math.min(10.0, dv);

        updateStrictVisibility();
    }

    syncControlsFromModel();

    cmapSelect.addEventListener("change", (e) => {
        model.set("colormap", e.target.value);
        model.save_changes();
        applyColormap(); 
    });
    
    toolSelect.addEventListener("change", (e) => {
        model.set("tool_mode", e.target.value);
        model.save_changes();
        updateStrictVisibility();
    });

    normCheck.addEventListener("change", (e) => {
        model.set("normalize_density", e.target.checked);
        model.save_changes();
    });

    strictCheck.addEventListener("change", (e) => {
        model.set("strict_mode", e.target.checked);
        model.save_changes();
    });

    // Contrast (Gamma)
    gammaSlider.addEventListener("input", (e) => {
        gammaNum.value = e.target.value;
        applyColormap();
    });
    gammaNum.addEventListener("change", (e) => {
        const val = Math.max(0.05, parseFloat(e.target.value) || 1.0);
        gammaNum.value = val;
        gammaSlider.value = Math.min(3.0, val);
        applyColormap();
    });

    // Data Opacity
    dataOpSlider.addEventListener("input", (e) => {
        dataOpNum.value = e.target.value;
        applyColormap();
    });
    dataOpNum.addEventListener("change", (e) => {
        const val = safeOpacity(e.target.value);
        dataOpNum.value = val;
        dataOpSlider.value = val;
        applyColormap();
    });

    // Rep Lines Opacity
    repOpSlider.addEventListener("input", (e) => {
        repOpNum.value = e.target.value;
        updateRepLines();
    });
    repOpNum.addEventListener("change", (e) => {
        const val = safeOpacity(e.target.value);
        repOpNum.value = val;
        repOpSlider.value = val;
        updateRepLines();
    });

    // Rep Lines Count
    // Backend-driven sliders sync on "change" (release) to avoid flooding the ZMQ queue.
    repSlider.addEventListener("input", (e) => {
        repNum.value = e.target.value;
    });
    repSlider.addEventListener("change", (e) => {
        model.set("rep_count", parseInt(e.target.value, 10));
        model.save_changes();
    });
    repNum.addEventListener("change", (e) => {
        const val = Math.max(1, parseInt(e.target.value, 10) || 1);
        repNum.value = val;
        repSlider.value = Math.min(10, val);
        model.set("rep_count", val);
        model.save_changes();
    });

    // Diversity
    divSlider.addEventListener("input", (e) => {
        divNum.value = e.target.value;
    });
    divSlider.addEventListener("change", (e) => {
        model.set("diversity", parseFloat(e.target.value));
        model.save_changes();
    });
    divNum.addEventListener("change", (e) => {
        const val = Math.max(0.0, parseFloat(e.target.value) || 0.0);
        divNum.value = val;
        divSlider.value = Math.min(10.0, val);
        model.set("diversity", val);
        model.save_changes();
    });
    
    // Method/Algorithm Selection
    repMethodSelect.addEventListener("change", (e) => {
        model.set("rep_method", e.target.value);
        model.save_changes();
    });

    // Reflect programmatic Python traitlet updates in the UI.
    model.on("change:colormap", () => { syncControlsFromModel(); applyColormap(); });
    model.on("change:tool_mode", syncControlsFromModel);
    model.on("change:normalize_density", syncControlsFromModel);
    model.on("change:strict_mode", syncControlsFromModel);
    model.on("change:rep_method", syncControlsFromModel);
    model.on("change:rep_count", syncControlsFromModel);
    model.on("change:diversity", syncControlsFromModel);

    /**
     * Determine the appropriate fallback explore_mode after deleting spatial queries.
     * @returns {string}
     */
    function getFallbackExploreMode() {
        const stats = model.get("stats_data");
        if (stats && stats.coordinated) return "coordinated";
        if (stats && stats.attribute) return "attribute";
        return "global";
    }

    // Query Clear-up
    el.querySelector("#clear-all-btn").addEventListener("click", () => {
        queries = [];
        model.set("explore_mode", getFallbackExploreMode());
        syncQueries();
    });

    // =========================================================================
    // Canvas Rendering Layers (Axes, Density, Rep Lines, Table)
    // =========================================================================

    /** Render L-shaped axes, 5-interval numeric ticks, and optional X/Y titles. */
    function drawAxes() {
        const bounds = model.get("bounds");
        if (!bounds) return;
        const xLabel = (model.get("x_label") || "").trim();
        const yLabel = (model.get("y_label") || "").trim();
        
        axesCtx.clearRect(0, 0, width + PAD_LEFT + PAD_RIGHT, height + PAD_TOP + PAD_BOTTOM);
        axesCtx.strokeStyle = "black";
        axesCtx.fillStyle = "black";
        axesCtx.font = "12px sans-serif";
        axesCtx.lineWidth = 1;

        // Draw Axis.
        axesCtx.beginPath();
        axesCtx.moveTo(PAD_LEFT, PAD_TOP);
        axesCtx.lineTo(PAD_LEFT, PAD_TOP + height);
        axesCtx.lineTo(PAD_LEFT + width, PAD_TOP + height);
        axesCtx.stroke();

        // Y-Ticks.
        for(let i=0; i<=5; i++) {
            const py = PAD_TOP + height - (i/5 * height);
            const val = bounds.v_min + (i/5 * (bounds.v_max - bounds.v_min));
            axesCtx.beginPath();
            axesCtx.moveTo(PAD_LEFT - 5, py);
            axesCtx.lineTo(PAD_LEFT, py);
            axesCtx.stroke();
            axesCtx.textAlign = "right";
            axesCtx.textBaseline = "middle";
            axesCtx.fillText(val.toFixed(1), PAD_LEFT - 8, py);
        }

        // X-Ticks.
        for(let i=0; i<=5; i++) {
            const px = PAD_LEFT + (i/5 * width);
            const val = bounds.t_min + (i/5 * (bounds.t_max - bounds.t_min));
            axesCtx.beginPath();
            axesCtx.moveTo(px, PAD_TOP + height);
            axesCtx.lineTo(px, PAD_TOP + height + 5);
            axesCtx.stroke();
            axesCtx.textAlign = "center";
            axesCtx.textBaseline = "top";
            axesCtx.fillText(val.toFixed(1), px, PAD_TOP + height + 8);
        }

        // Optional Y-Axis Label and Plot Header Badge
        if (yLabel) {
            // Header badge above canvas
            axesCtx.save();
            axesCtx.font = "bold 14px sans-serif";
            axesCtx.fillStyle = "#515a6e";
            axesCtx.textAlign = "center";
            axesCtx.textBaseline = "middle";
            axesCtx.fillText(yLabel, PAD_LEFT + width / 2, PAD_TOP / 2);
            axesCtx.restore();

            // Rotated Y-axis label
            axesCtx.save();
            axesCtx.font = "bold 12px sans-serif";
            axesCtx.fillStyle = "#515a6e";
            axesCtx.translate(14, PAD_TOP + height / 2);
            axesCtx.rotate(-Math.PI / 2);
            axesCtx.textAlign = "center";
            axesCtx.textBaseline = "middle";
            axesCtx.fillText(yLabel, 0, 0);
            axesCtx.restore();
        }

        // Optional X-Axis Label
        if (xLabel) {
            axesCtx.save();
            axesCtx.font = "bold 12px sans-serif";
            axesCtx.fillStyle = "#515a6e";
            axesCtx.textAlign = "center";
            axesCtx.textBaseline = "bottom";
            axesCtx.fillText(xLabel, PAD_LEFT + width / 2, PAD_TOP + height + PAD_BOTTOM - 2);
            axesCtx.restore();
        }
    }

    /** Apply colormap, gamma correction, and opacity to the offscreen density buffer. */
    function applyColormap() {
        const cmapName = model.get("colormap");
        const parsedGamma = parseFloat(gammaNum.value);
        const gamma = Number.isFinite(parsedGamma) && parsedGamma > 0 ? parsedGamma : 1.0;
        const opacity = safeOpacity(dataOpNum.value);
        
        const imgData = hiddenCtx.getImageData(0, 0, width, height);
        const data = imgData.data;
        const outData = densityCtx.createImageData(width, height);
        
        for (let i = 0; i < data.length; i += 4) {
            // Apply Gamma correction to intensity.
            const intensity = Math.pow(data[i] / 255, gamma) * 255;
            const color = getColor(intensity, cmapName);
            outData.data[i] = color[0];
            outData.data[i+1] = color[1];
            outData.data[i+2] = color[2];
            outData.data[i+3] = intensity <= 0 ? 0 : 255 * opacity; // Handle Data Opacity
        }
        densityCtx.putImageData(outData, 0, 0);
    }

    /** Decode the base64 grayscale PNG sent by Python and re-render the density layer. */
    function updateDensity() {
        const base64Image = model.get("density_image");
        if (!base64Image) {
            hiddenCtx.clearRect(0, 0, width, height);
            densityCtx.clearRect(0, 0, width, height);
            return;
        }
        const img = new Image();
        img.onload = () => {
            hiddenCtx.clearRect(0, 0, width, height);
            hiddenCtx.drawImage(img, 0, 0);
            applyColormap();
        };
        img.src = "data:image/png;base64," + base64Image;
    }

    function updateRepLines() {
        repCtx.clearRect(0, 0, width, height);
        const lines = model.get("rep_lines_data");
        if (!lines || lines.length === 0) return;

        repCtx.globalAlpha = safeOpacity(repOpNum.value);
        lines.forEach((line, i) => {
            repCtx.beginPath();
            repCtx.strokeStyle = REP_LINE_COLORS[i % REP_LINE_COLORS.length];
            repCtx.lineWidth = 3;
            repCtx.moveTo(line[0][0], line[0][1]);
            for(let j=1; j<line.length; j++) {
                repCtx.lineTo(line[j][0], line[j][1]);
            }
            repCtx.stroke();
        });
        repCtx.globalAlpha = 1.0;
    }

    /**
     * Derive angular wedge geometry and handle coordinates for an Angular query.
     * Clamps wedge bounds strictly inside the forward-time half-plane (-pi/2, +pi/2).
     *
     * @param {{sx: number, sy: number, ex: number, ey: number, tol_deg?: number}} q
     * @returns {{R: number, theta: number, displayDeg: number, tolDeg: number, tolRad: number, theta1: number, theta2: number, ux: number, uy: number, lx: number, ly: number}}
     */
    function getAngularGeometry(q) {
        const dx = q.ex - q.sx;
        const dy = q.ey - q.sy;
        const R = Math.hypot(dx, dy);
        const theta = Math.atan2(dy, dx);
        // Invert dy for user-facing angle so upward slopes are positive (+deg°) 
        // and downward slopes are negative (-deg°).
        const safeDx = Math.abs(dx) < 1e-9 ? 1e-9 : Math.abs(dx);
        const normDy = dx < 0 ? dy : -dy;
        const displayDeg = (Math.atan2(normDy, safeDx) * 180) / Math.PI;
        const tolDeg = q.tol_deg || 15;
        const tolRad = (tolDeg * Math.PI) / 180.0;
        // Clamp wedge edges strictly inside forward-time half-plane (-pi/2, +pi/2).
        const maxHalfAngle = Math.PI / 2 - 0.02;
        const theta1 = Math.max(-maxHalfAngle, theta - tolRad);
        const theta2 = Math.min(maxHalfAngle, theta + tolRad);
        return {
            R, theta, displayDeg, tolDeg, tolRad, theta1, theta2,
            ux: q.sx + R * Math.cos(theta1),
            uy: q.sy + R * Math.sin(theta1),
            lx: q.sx + R * Math.cos(theta2),
            ly: q.sy + R * Math.sin(theta2)
        };
    }

    /**
     * Format a single HTML table row for the Information Table.
     *
     * @param {string} name
     * @param {{count: number, points: number, min: number, max: number, mean: number, var: number}|undefined} data
     * @param {string} id
     * @param {boolean} isExplore
     * @returns {string}
     */
    function formatRow(name, data, id, isExplore) {
        if (!data) {
            return `<tr><td colspan="9" style="text-align: left; padding: 8px;"><b>${name}</b>: No data</td></tr>`;
        }
        const exploreCheck = isExplore ? "checked" : "";
        const actionBtn = id.startsWith("q")
            ? `<button class="del-btn" data-id="${id}" style="color:red; cursor:pointer;">Delete</button>`
            : "";

        // Use the unique `widgetGroupId` for the radio button name.
        return `
            <tr>
                <td><input type="radio" name="${widgetGroupId}" class="exp-radio" value="${id}" ${exploreCheck} style="cursor:pointer;"></td>
                <td style="text-align: left; font-weight: bold; padding: 8px;">${name}</td>
                <td>${data.count}</td><td>${data.points}</td>
                <td>${data.min.toFixed(2)}</td><td>${data.max.toFixed(2)}</td>
                <td>${data.mean.toFixed(2)}</td><td>${data.var.toFixed(2)}</td>
                <td>${actionBtn}</td>
            </tr>`;
    }

    /** Re-render the Information Table from synced Python statistics. */
    function updateStats() {
        const stats = model.get("stats_data");
        const mode = model.get("explore_mode");
        const tbody = el.querySelector("#table-body");
        
        if (!stats || !stats.global) {
            tbody.innerHTML = `<tr><td colspan="9" style="padding: 8px;">No data</td></tr>`;
            return;
        }

        let html = formatRow("Global Dataset", stats.global, "global", mode === "global");
        
        // Show Filters from other widgets when active.
        if (stats.attribute) {
            html += formatRow("<i>Attribute Filter</i>", stats.attribute, "attribute", mode === "attribute");
        }
        if (stats.coordinated) {
            html += formatRow("<i>Coordinated Filter</i>", stats.coordinated, "coordinated", mode === "coordinated");
        }
        
        queries.forEach((q, i) => {
            let extra = "";
            if (q.type === "ang") {
                const geom = getAngularGeometry(q);
                const sign = geom.displayDeg >= 0 ? "+" : "";
                extra = ` (${sign}${geom.displayDeg.toFixed(1)}° ± ${Math.round(geom.tolDeg)}°)`;
            }
            const qName = `<span style="color:${q.color}">&#9632;</span> ${q.type.toUpperCase()} ${i+1}${extra}`;
            html += formatRow(qName, stats.queries[q.id], q.id, mode === q.id);
        });

        if (queries.length > 1) {
            html += formatRow("<i>Intersection</i>", stats.intersection, "intersection", mode === "intersection");
            html += formatRow("<i>Union</i>", stats.union, "union", mode === "union");
        }

        tbody.innerHTML = html;

        tbody.querySelectorAll(".exp-radio").forEach(r => {
            r.addEventListener("change", (e) => {
                model.set("explore_mode", e.target.value);
                model.save_changes();
                renderShapes();
            });
        });

        tbody.querySelectorAll(".del-btn").forEach(b => {
            b.addEventListener("click", (e) => {
                const delId = e.target.getAttribute("data-id");
                queries = queries.filter(q => q.id !== delId);
                
                const fallbackMode = getFallbackExploreMode();
                
                let nextMode = model.get("explore_mode");
                if (nextMode === delId) nextMode = fallbackMode;
                    
                // If user drops to a single query, `intersection`/`union`
                // modes are invalid. Switch to the remaining query.
                if (queries.length === 1 && (nextMode === "intersection" || nextMode === "union")) {
                    nextMode = queries[0].id;
                } else if (queries.length === 0) {
                    nextMode = fallbackMode;
                }
                
                model.set("explore_mode", nextMode);
                syncQueries();
            });
        });
    }

    model.on("change:bounds", drawAxes);
    model.on("change:x_label", drawAxes);
    model.on("change:y_label", drawAxes);
    model.on("change:density_image", updateDensity);
    model.on("change:rep_lines_data", updateRepLines);
    model.on("change:stats_data", updateStats);
    model.on("change:explore_mode", () => { updateStats(); renderShapes(); });
    model.on("change:query_data", () => {
        try {
            const parsed = JSON.parse(model.get("query_data") || "[]");
            if (Array.isArray(parsed)) {
                queries = parsed;
                renderShapes();
                updateStats();
            }
        } catch (e) {}
    });

    // =========================================================================
    // Stateful Multi-Shape Interaction Manager
    // =========================================================================

    let isDragging = false;
    let dragMode = "none";
    let activeId = null;
    let startX = 0;
    let startY = 0;
    let activeOriginal = null;
    const EDGE_TOLERANCE = 8;

    /** Persist the local query list to Python and redraw the interaction layer. */
    function syncQueries() {
        model.set("query_data", JSON.stringify(queries));
        model.save_changes();
        renderShapes();
    }

    /** Draw all active Brush, RNL, Angular, and Pin shapes on the interaction canvas. */
    function renderShapes() {
        interactionCtx.clearRect(0, 0, width, height);
        const mode = model.get("explore_mode");
        
        queries.forEach(q => {
            const isFocused = (q.id === mode || mode === "union" || mode === "intersection");
            interactionCtx.strokeStyle = q.color;
            interactionCtx.lineWidth = isFocused ? 3 : 2;
            interactionCtx.beginPath();
            
            if (q.type === "brush") {
                interactionCtx.save();
                interactionCtx.fillStyle = q.color;
                interactionCtx.globalAlpha = 0.10;
                interactionCtx.fillRect(q.x_min, q.y_min, q.w, q.h);
                interactionCtx.restore();
                interactionCtx.strokeRect(q.x_min, q.y_min, q.w, q.h);
            } else if (q.type === "rnl") {
                interactionCtx.save();
                interactionCtx.fillStyle = q.color;
                interactionCtx.globalAlpha = 0.10;
                interactionCtx.beginPath();
                interactionCtx.arc(q.cx, q.cy, q.r, 0, 2 * Math.PI);
                interactionCtx.fill();
                interactionCtx.restore();

                interactionCtx.beginPath();
                interactionCtx.arc(q.cx, q.cy, q.r, 0, 2 * Math.PI);
                interactionCtx.stroke();
            } else if (q.type === "ang") {
                const geom = getAngularGeometry(q);
                if (geom.R > 2) {
                    // Shaded angular tolerance wedge.
                    interactionCtx.save();
                    interactionCtx.fillStyle = q.color;
                    interactionCtx.globalAlpha = 0.22;
                    interactionCtx.beginPath();
                    interactionCtx.moveTo(q.sx, q.sy);
                    interactionCtx.arc(q.sx, q.sy, geom.R, geom.theta1, geom.theta2);
                    interactionCtx.closePath();
                    interactionCtx.fill();
                    interactionCtx.restore();

                    // Outer wedge perimeter.
                    interactionCtx.beginPath();
                    interactionCtx.moveTo(q.sx, q.sy);
                    interactionCtx.arc(q.sx, q.sy, geom.R, geom.theta1, geom.theta2);
                    interactionCtx.closePath();
                    interactionCtx.stroke();

                    // Central target-slope ray (dashed).
                    interactionCtx.save();
                    interactionCtx.setLineDash([4, 4]);
                    interactionCtx.beginPath();
                    interactionCtx.moveTo(q.sx, q.sy);
                    interactionCtx.lineTo(q.ex, q.ey);
                    interactionCtx.stroke();
                    interactionCtx.restore();

                    // Draggable handles: vertex, center tip, upper/lower wedge corners.
                    interactionCtx.fillStyle = q.color;
                    [
                        [q.sx, q.sy],
                        [q.ex, q.ey],
                        [geom.ux, geom.uy],
                        [geom.lx, geom.ly]
                    ].forEach(([hx, hy]) => {
                        interactionCtx.beginPath();
                        interactionCtx.arc(hx, hy, 4, 0, 2 * Math.PI);
                        interactionCtx.fill();
                    });

                    // Live angle readout beside tip while actively adjusting.
                    if (isDragging && activeId === q.id) {
                        const sign = geom.displayDeg >= 0 ? "+" : "";
                        const angText = `θ: ${sign}${geom.displayDeg.toFixed(1)}° (±${Math.round(geom.tolDeg)}°)`;
                        const tx = Math.min(width - 145, Math.max(10, q.ex + 12));
                        const ty = Math.min(height - 10, Math.max(20, q.ey + 18));
                        interactionCtx.save();
                        interactionCtx.font = "bold 13px sans-serif";
                        interactionCtx.lineWidth = 3;
                        interactionCtx.strokeStyle = "white";
                        interactionCtx.strokeText(angText, tx, ty);
                        interactionCtx.fillStyle = q.color;
                        interactionCtx.fillText(angText, tx, ty);
                        interactionCtx.restore();
                    }
                }
            } else if (q.type === "pin") {
                interactionCtx.fillStyle = q.color;
                interactionCtx.arc(q.cx, q.cy, 4, 0, 2 * Math.PI);
                interactionCtx.fill();
            }
        });
    }

    /** Draw dashed hover crosshair with a white-haloed data coordinate readout. */
    function drawCrosshairAndText(x, y) {
        const bounds = model.get("bounds");
        if (!bounds) return;
        const tVal = bounds.t_min + (x / width) * (bounds.t_max - bounds.t_min);
        const vVal = bounds.v_max - (y / height) * (bounds.v_max - bounds.v_min);

        interactionCtx.beginPath();
        interactionCtx.setLineDash([5, 5]);
        interactionCtx.strokeStyle = "rgba(0,0,0,0.5)";
        interactionCtx.lineWidth = 1;
        interactionCtx.moveTo(x, 0);
        interactionCtx.lineTo(x, height);
        interactionCtx.moveTo(0, y);
        interactionCtx.lineTo(width, y);
        interactionCtx.stroke();
        interactionCtx.setLineDash([]);

        const txt1 = `T: ${tVal.toFixed(2)}`;
        const txt2 = `V: ${vVal.toFixed(2)}`;
        interactionCtx.font = "bold 13px sans-serif";
        interactionCtx.lineWidth = 3;
        interactionCtx.strokeStyle = "white";
        
        interactionCtx.strokeText(txt1, x + 8, y - 20);
        interactionCtx.strokeText(txt2, x + 8, y - 5);
        
        interactionCtx.fillStyle = "black";
        interactionCtx.fillText(txt1, x + 8, y - 20);
        interactionCtx.fillText(txt2, x + 8, y - 5);
    }

    /**
     * Map cursor angle around an RNL circle center to the matching CSS resize cursor.
     * @param {number} cx
     * @param {number} cy
     * @param {number} x
     * @param {number} y
     * @returns {string}
     */
    function getRadialCursor(cx, cy, x, y) {
        // Determine resize cursor orientation based on angle around circle center.
        const angle = ((Math.atan2(y - cy, x - cx) * 180) / Math.PI + 360) % 180;
        if (angle < 22.5 || angle >= 157.5) return "ew-resize";
        if (angle >= 22.5 && angle < 67.5) return "nwse-resize";
        if (angle >= 67.5 && angle < 112.5) return "ns-resize";
        return "nesw-resize";
    }

    /**
     * Hit-test cursor position against existing query shapes (top-most first).
     *
     * @param {number} x - Canvas X pixel coordinate.
     * @param {number} y - Canvas Y pixel coordinate.
     * @returns {{ mode: string, qid?: string, cursor?: string }}
     */
    function getHitTest(x, y) {
        for (let i = queries.length - 1; i >= 0; i--) {
            const q = queries[i];
            if (q.type === "brush") {
                const inX = x >= q.x_min - EDGE_TOLERANCE && x <= q.x_min + q.w + EDGE_TOLERANCE;
                const inY = y >= q.y_min - EDGE_TOLERANCE && y <= q.y_min + q.h + EDGE_TOLERANCE;
                if (!inX || !inY) continue;
                
                const onL = Math.abs(x - q.x_min) <= EDGE_TOLERANCE;
                const onR = Math.abs(x - (q.x_min + q.w)) <= EDGE_TOLERANCE;
                const onT = Math.abs(y - q.y_min) <= EDGE_TOLERANCE;
                const onB = Math.abs(y - (q.y_min + q.h)) <= EDGE_TOLERANCE;

                if (onL && onT) return { qid: q.id, mode: "resize-nw" };
                if (onR && onT) return { qid: q.id, mode: "resize-ne" };
                if (onL && onB) return { qid: q.id, mode: "resize-sw" };
                if (onR && onB) return { qid: q.id, mode: "resize-se" };
                if (onL) return { qid: q.id, mode: "resize-w" };
                if (onR) return { qid: q.id, mode: "resize-e" };
                if (onT) return { qid: q.id, mode: "resize-n" };
                if (onB) return { qid: q.id, mode: "resize-s" };
                return { qid: q.id, mode: "move" };
            } else if (q.type === "rnl") {
                const dist = Math.hypot(x - q.cx, y - q.cy);
                const rnlTol = Math.min(EDGE_TOLERANCE, Math.max(3, q.r * 0.3));
                if (Math.abs(dist - q.r) <= rnlTol) {
                    return { qid: q.id, mode: "resize-rnl", cursor: getRadialCursor(q.cx, q.cy, x, y) };
                }
                if (dist < q.r - rnlTol) {
                    return { qid: q.id, mode: "move" };
                }
            } else if (q.type === "ang") {
                const geom = getAngularGeometry(q);
                // Center tip handle -> resize direction & length
                if (Math.hypot(x - q.ex, y - q.ey) <= 10) {
                    return { qid: q.id, mode: "resize-end" };
                }
                // Upper or lower wedge corner handles -> resize angular tolerance
                if (Math.hypot(x - geom.ux, y - geom.uy) <= 10 || Math.hypot(x - geom.lx, y - geom.ly) <= 10) {
                    return { qid: q.id, mode: "resize-ang-tol" };
                }
                // Vertex or inside shaded sector -> move entire wedge
                const distFromVertex = Math.hypot(x - q.sx, y - q.sy);
                if (distFromVertex <= 10) {
                    return { qid: q.id, mode: "move" };
                }
                if (distFromVertex <= geom.R) {
                    const ptAngle = Math.atan2(y - q.sy, x - q.sx);
                    if (ptAngle >= geom.theta1 && ptAngle <= geom.theta2) {
                        return { qid: q.id, mode: "move" };
                    }
                }
            } else if (q.type === "pin") {
                if (Math.hypot(x - q.cx, y - q.cy) < 10) return { qid: q.id, mode: "move" };
            }
        }
        return { mode: "none" };
    }

    const interactionCanvas = el.querySelector("#interaction-layer");

    interactionCanvas.addEventListener("mousedown", (e) => {
        isDragging = true;
        startX = e.offsetX;
        startY = e.offsetY;
        const tool = model.get("tool_mode");
        const hit = getHitTest(startX, startY);

        if (hit.mode !== "none") {
            dragMode = hit.mode;
            activeId = hit.qid;
            activeOriginal = { ...queries.find(q => q.id === activeId) };
        } else {
            dragMode = "draw";
            activeId = "q_" + Date.now();
            const color = QUERY_COLORS[queries.length % QUERY_COLORS.length];
            if (tool === "brush") queries.push({ id: activeId, type: "brush", color, x_min: startX, y_min: startY, w: 0, h: 0 });
            else if (tool === "rnl") queries.push({ id: activeId, type: "rnl", color, cx: startX, cy: startY, r: 0 });
            else if (tool === "ang") queries.push({ id: activeId, type: "ang", color, sx: startX, sy: startY, ex: startX, ey: startY, tol_deg: 15 });
            else if (tool === "pin") queries.push({ id: activeId, type: "pin", color, cx: startX, cy: startY });
        }
    });

    interactionCanvas.addEventListener("mousemove", (e) => {
        const x = e.offsetX;
        const y = e.offsetY;
        const dx = x - startX;
        const dy = y - startY;

        if (!isDragging) {
            const hit = getHitTest(x, y);
            if (hit.mode === "move") interactionCanvas.style.cursor = "move";
            else if (hit.mode === "resize-rnl") interactionCanvas.style.cursor = hit.cursor;
            else if (hit.mode === "resize-ang-tol") interactionCanvas.style.cursor = "ns-resize";
            else if (hit.mode.includes("resize")) {
                if (hit.mode === "resize-e" || hit.mode === "resize-w") interactionCanvas.style.cursor = "ew-resize";
                else if (hit.mode === "resize-n" || hit.mode === "resize-s") interactionCanvas.style.cursor = "ns-resize";
                else if (hit.mode === "resize-nw" || hit.mode === "resize-se") interactionCanvas.style.cursor = "nwse-resize";
                else interactionCanvas.style.cursor = "nesw-resize";
            } else interactionCanvas.style.cursor = "crosshair";
        }

        if (isDragging) {
            const q = queries.find(item => item.id === activeId);
            if (dragMode === "draw") {
                if (q.type === "brush") {
                    q.x_min = Math.min(startX, x);
                    q.y_min = Math.min(startY, y);
                    q.w = Math.abs(x - startX);
                    q.h = Math.abs(y - startY);
                } else if (q.type === "rnl") {
                    q.r = Math.hypot(x - startX, y - startY);
                } else if (q.type === "ang") {
                    q.ex = Math.max(q.sx + 5, x);
                    q.ey = y;
                } else if (q.type === "pin") {
                    q.cx = x;
                    q.cy = y;
                }
            } else if (dragMode === "move") {
                if (q.type === "brush") {
                    q.x_min = activeOriginal.x_min + dx;
                    q.y_min = activeOriginal.y_min + dy;
                } else if (q.type === "rnl" || q.type === "pin") {
                    q.cx = activeOriginal.cx + dx;
                    q.cy = activeOriginal.cy + dy;
                } else if (q.type === "ang") {
                    q.sx = activeOriginal.sx + dx;
                    q.sy = activeOriginal.sy + dy;
                    q.ex = activeOriginal.ex + dx;
                    q.ey = activeOriginal.ey + dy;
                }
            } else if (dragMode === "resize-rnl" && q.type === "rnl") {
                q.r = Math.max(3, Math.hypot(x - q.cx, y - q.cy));
            } else if (dragMode.includes("resize") && q.type === "brush") {
                const dir = dragMode.split("-")[1];
                if (dir.includes("e")) q.w = Math.max(5, activeOriginal.w + dx);
                if (dir.includes("s")) q.h = Math.max(5, activeOriginal.h + dy);
                if (dir.includes("w")) {
                    const shift = Math.min(dx, activeOriginal.w - 5);
                    q.x_min = activeOriginal.x_min + shift;
                    q.w = activeOriginal.w - shift;
                }
                if (dir.includes("n")) {
                    const shift = Math.min(dy, activeOriginal.h - 5);
                    q.y_min = activeOriginal.y_min + shift;
                    q.h = activeOriginal.h - shift;
                }
            } else if (dragMode === "resize-end" && q.type === "ang") {
                q.ex = Math.max(q.sx + 5, activeOriginal.ex + dx);
                q.ey = activeOriginal.ey + dy;
            } else if (dragMode === "resize-ang-tol" && q.type === "ang") {
                const centerTheta = Math.atan2(q.ey - q.sy, q.ex - q.sx);
                const mouseTheta = Math.atan2(y - q.sy, x - q.sx);
                const diffDeg = (Math.abs(Math.atan2(Math.sin(mouseTheta - centerTheta), Math.cos(mouseTheta - centerTheta))) * 180) / Math.PI;
                q.tol_deg = Math.min(85, Math.max(1, diffDeg));
            }
        }
        
        renderShapes();
        drawCrosshairAndText(x, y);
    });

    interactionCanvas.addEventListener("mouseup", () => {
        if (!isDragging) return;
        isDragging = false;
        
        // Track if a new shape is created.
        const wasDrawing = (dragMode === "draw"); 
        
        // Discard accidental micro-clicks that do not form valid query regions.
        queries = queries.filter(q => {
            if (q.type === "brush") return q.w > 2 && q.h > 2;
            if (q.type === "rnl") return q.r > 2;
            if (q.type === "ang") return Math.hypot(q.ex - q.sx, q.ey - q.sy) > 2;
            return true; // Pins are always valid
        });
        
        // Auto-switch explore mode for newly drawn shapes.
        if (wasDrawing && queries.some(q => q.id === activeId)) {
            if (queries.length === 1) {
                model.set("explore_mode", activeId);
            } else if (queries.length > 1) {
                model.set("explore_mode", "intersection");
            }
        }
        
        syncQueries();
        activeId = null;
        dragMode = "none";
    });

    interactionCanvas.addEventListener("mouseleave", () => {
        if (isDragging) interactionCanvas.dispatchEvent(new Event("mouseup"));
        renderShapes();
    });

    // Initial renders.
    renderAttributeControls();
    drawAxes();
    updateDensity();
    updateRepLines();
    updateStats();
    renderShapes();
}
