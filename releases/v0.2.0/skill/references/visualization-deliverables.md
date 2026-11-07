# Visual evidence is part of the deliverable

The report must let a reader see what was simulated, where its derivative acts,
and whether the evidence supports the numerical claims. Choose a few coherent
figure plates; do not fill the README with repeated metrics or decorate it with
synthetic-looking wavefronts. Export readable PNG previews and editable PDF/SVG,
plus a native wavefield animation when saved states make one feasible.

| Evidence | Required content | Important distinction |
|---|---|---|
| Model | Actual target, actual starting model and their difference in declared coordinates | Call an initial model smoothed/blurred only if a real smoothing operation produced the simulated input; record kernel, widths, grid spacing and fixed masks. A layered reference is not a blurred model. |
| Native state | Several physically labelled time snapshots, source/receiver geometry, and a short animation when feasible | Extract original states, with component, time convention, coordinates and MPI identity. Label a meridional or modal view; do not imply full 3-D coverage. |
| Data boundary | Observed/predicted gather or representative traces, residual and actual AD data cotangent | Residual and adjoint source coincide only for the corresponding objective/conventions. Include the processed-objective case. |
| Model derivative | Unprocessed coefficient/model gradient and its coordinates/units | Separate quadrature-weighted coefficient derivatives, continuous kernel densities and regional parameter derivatives. Use bars for a few regional parameters; do not invent a dense kernel by broadcasting them. |
| Correctness | Full Taylor sweeps, model/propagation dots, native-versus-extension discrepancies, or matched gradient comparison | Keep failed steps and roundoff regions. Show raw signed/absolute differences with their own scale; do not independently normalize two gradients and claim equality. |
| Resources, if measured | Matched time/memory with repeat points and measured uncertainty | Keep failed capacity attempts separate; do not infer a controlled speedup from single shared-host observations. |

Use common scales for comparable models, methods and animation frames. Use a
zero-centred diverging scale for signed residuals and gradients. If display
clipping is necessary, disclose limits and clipped count/fraction; preserve raw
arrays and full-range statistics. No gradient smoothing, illumination division,
or fitted scaling may be silently introduced for appearance. If a true-model
gradient is zero because the objective exactly matches its target, display and
explain that fact rather than manufacture a nonzero kernel.

For native unstructured states, preserve topology or clearly label the display
interpolation. Merge MPI state replicas only after verifying physical identity
and numerical agreement. Element-local coefficient derivatives are separate
contributions: sum only according to the advertised parameter map, not an
arbitrary spatial averaging rule. After native repartition, align saved states
by physical/global identity rather than local rank index.

Temporal sampling for snapshots/GIFs is allowed with a recorded rule, the full
declared spatial field or section, selected native indices and physical timestamps.
For a 3-D section or spatial aggregation, disclose its extent, selection and interpolation;
do not silently choose a favourable crop. A newly run
visualization case must have its own configuration/provenance; it does not
inherit a historical benchmark's time, memory or error measurement.

## Reproducible output

Provide plotting source, an exact command and a figure manifest containing input
hashes, array keys, model/coordinate/component definitions, case/parameter vector,
time indices, transforms/aggregation, scale/clipping policy and output files.
Publish only permitted source data; link restricted inputs to their origin and
reconstruction recipe. Keep workstation-specific paths out of public captions.

Put representative model, wavefield and gradient figures near the README top,
with short captions and a table of tested physics/parameters/ranks. Link a visual
report for the full gallery, animation and verification/resource figures. Open
every rendered plate to check labels, units, colourbars and overlays. Record
rendered axes alignment and review PDF text/collisions when the tools are
available; never call an uninspected output publication-ready.

## Optional portable array renderer

`python scripts/visualize.py --arrays case.npz --spec figures.json --output figures`
needs NumPy, Matplotlib and Pillow (for GIF). It does not run or differentiate a
solver. `figures.json` contains `figures`, each with `name`, `title`, `caption`,
`columns` and a list of panels:

```json
{"figures": [{"name": "models", "title": "Target and starting model",
  "caption": "Actual simulated arrays; same velocity scale.", "columns": 2,
  "panels": [
    {"kind": "map", "key": "target", "label": "Target", "units": "km/s",
     "extent": [0, 10, 5, 0], "xlabel": "Distance (km)", "ylabel": "Depth (km)", "scale_group": "velocity"},
    {"kind": "map", "key": "initial", "label": "Starting model", "units": "km/s",
     "extent": [0, 10, 5, 0], "xlabel": "Distance (km)", "ylabel": "Depth (km)", "scale_group": "velocity"}
  ]}]}
```

`map` uses a real 2-D array and physical `extent`. `points` uses one-dimensional
values and `x_key`/`y_key`; no spatial interpolation is performed. Either accepts
`frame` to select a history slice and `symmetric: true` for signed scales. Shared
`scale_group` requires identical units and clipping policy. Optional `percentile`
sets an explicitly recorded display limit; the default uses the full range.
`bars` uses `key`, `categories` and `ylabel`. `curves` uses `series` entries
(`x`, `y`, `label`) and optional `xscale`/`yscale`; logarithmic data must be
positive. Labels and axis units are supplied explicitly by the application.

An optional `animations` list uses a `map` or `points` definition with a history
`key`, `time_key`, explicit `frames`, `label`, `units` and `fps`. All selected
frames use the same scale. The manifest records selection and clipping; it is
not a substitute for the source/discrete derivative contract. Other plotting
libraries or exports are equally valid when the required evidence is clear.
