# First-Stage-SVG

Pure-Python bitmap-to-SVG vectorizer suite. Three independent pipelines, no C bindings, no pypotrace.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Pipelines Overview](#pipelines-overview)
- [tracer.py — Potrace pipeline](#tracerpy--potrace-pipeline-recommended)
- [ypotrace.py — Schneider fitter](#ypotracepy--schneider-least-squares-fitter)
- [yd_vector/tracer.py — Catmull-Rom library](#yd_vectortracerpy--catmull-rom-library)
- [Configuration](#configuration)
- [ML Pre-processing](#ml-pre-processing-real-esrgan--sam-21)
- [Architecture](#architecture)
- [SVG Output](#svg-output)
- [File Reference](#file-reference)
- [Tests](#tests)

---

## Quick Start

```bash
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

python tracer.py inputs/1.jpg outputs/1.svg --otsu
```

---

## Installation

```bash
git clone <repo>
cd First-Stage-SVG

python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

**Core dependencies** (`numpy`, `opencv-python`, `Pillow`, `PyYAML`) are all in `requirements.txt`.

**Optional — ML pre-processing** (Real-ESRGAN + SAM 2.1, requires CUDA 12.x):

```bash
# Install torch first with the correct CUDA index
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Then the ML packages
pip install git+https://github.com/XPixelGroup/BasicSR.git
pip install realesrgan segment-anything-2 scikit-image
```

> Tested on NVIDIA RTX 5090 32 GB. Without these packages the `--preprocess` flag
> falls back to Lanczos4 upscaling + k-means segmentation automatically.

---

## Pipelines Overview

| Entry point | Algorithm | Output SVG commands | CLI |
|---|---|---|---|
| `tracer.py` | Potrace 8-stage | M C L A Z | yes |
| `ypotrace.py` | Schneider least-squares | M C L Z | yes |
| `yd_vector/tracer.py` | Catmull-Rom | M C L A Z | library only |

---

## `tracer.py` — Potrace pipeline (recommended)

Faithful Python translation of Peter Selinger's Potrace algorithm (sourced from kilobtye/potrace.js).

### Binary mode

```bash
# Dark shapes on white background (default threshold 128)
python tracer.py inputs/1.jpg outputs/1.svg

# OTSU auto-threshold — best for photos
python tracer.py inputs/1.jpg outputs/1.svg --otsu

# Trace dark shapes on light background
python tracer.py inputs/1.jpg outputs/1.svg --otsu --invert

# Custom colour + transparent background
python tracer.py inputs/1.jpg outputs/1.svg --otsu --fill "#1a1a2e"

# Bridge thin white gaps before tracing (morphological close)
python tracer.py inputs/1.jpg outputs/1.svg --otsu --close 3

# Scale output 2× and add white background
python tracer.py inputs/1.jpg outputs/1.svg --otsu --size 2.0 --bg white
```

### Multi-colour mode

K-means quantizes the image to N colours, then runs the full Potrace pipeline on each colour mask independently. Layers are stacked lightest → darkest in the SVG.

```bash
python tracer.py inputs/1.jpg outputs/1.svg --colors 8
python tracer.py inputs/1.jpg outputs/1.svg --colors 12 --size 1.5
```

### ML pre-processing mode

Real-ESRGAN ×4 upscale → SAM 2.1 segmentation → LAB colour merge → per-layer Potrace.
Model weights are downloaded automatically to `models/` on first run.

```bash
python tracer.py inputs/1.jpg outputs/1.svg --preprocess
python tracer.py inputs/1.jpg outputs/1.svg --preprocess --colors 8
python tracer.py inputs/1.jpg outputs/1.svg --preprocess --no-upscale  # skip ESRGAN
```

### scripts/trace.py wrapper

Same flags, same pipeline — convenience wrapper for scripting:

```bash
python scripts/trace.py inputs/1.jpg outputs/1.svg --otsu --invert --debug
```

### Debug mode

Dumps one JSON object per pipeline stage per path to stdout:

```bash
python tracer.py inputs/1.jpg outputs/1.svg --debug 2>nul
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--otsu` | off | OTSU automatic threshold (ignores `--threshold`) |
| `--threshold N` | `128` | Manual grayscale threshold 0–255 |
| `--invert` | off | Invert bitmap — trace dark on light |
| `--close N` | `0` | Morphological close kernel in pixels (bridges thin gaps) |
| `--turdsize N` | `2` | Drop shapes with area ≤ N px² |
| `--alphamax F` | `1.0` | Corner threshold: lower = more corners kept |
| `--opttolerance F` | `0.2` | Curve-merge tolerance in pixels |
| `--no-optcurve` | off | Disable optiCurve merging pass (Stage 7) |
| `--turnpolicy P` | `minority` | Ambiguous-turn policy: `minority\|majority\|right\|black\|white` |
| `--colors N` | `0` (binary) | Multi-colour k-means mode — N colour layers |
| `--fill COLOR` | `#000000` | SVG fill colour (binary mode) |
| `--bg COLOR` | transparent | Background rect colour |
| `--size F` / `--scale F` | `1.0` | Output scale factor |
| `--preprocess` | off | ML pre-processing: Real-ESRGAN + SAM 2.1 |
| `--no-upscale` | off | Skip ESRGAN upscale when `--preprocess` is set |
| `--debug` | off | Print per-stage JSON state to stdout |

---

## `ypotrace.py` — Schneider least-squares fitter

Standalone vectorizer using Philip Schneider's least-squares Bézier fitting algorithm on `cv2.findContours` output. Independent of the Potrace pipeline.

```bash
python ypotrace.py inputs/1.jpg outputs/1.svg
python ypotrace.py inputs/1.jpg outputs/1.svg --threshold 100 --tolerance 1.5
python ypotrace.py inputs/1.jpg outputs/1.svg --alphamax 45
python ypotrace.py inputs/1.jpg outputs/1.svg --no-invert   # trace white shapes
```

| Flag | Default | Description |
|---|---|---|
| `--threshold N` | `128` | Binarization threshold |
| `--tolerance F` | `1.0` | Polygon approximation tolerance (opttolerance) |
| `--alphamax F` | `60.0` | Corner penalty angle in degrees |
| `--no-invert` | off | Do not invert — trace white shapes instead of black |

---

## `yd_vector/tracer.py` — Catmull-Rom library

Catmull-Rom → cubic Bézier path builder with micro-arc corner transitions at straight segments. This is a **library, not a CLI** — import it in your own code.

```python
from yd_vector.tracer import trace, export_svg, Params

# Load all params from configs/default.yaml
params = Params.from_yaml()

# Or construct manually
params = Params(
    threshold=100,
    alpha_max=0.8,
    scale=2.0,
    corner_arc_radius_sharp=0.42,
    corner_arc_radius_soft=0.92,
)

paths, width, height = trace("inputs/1.jpg", params)

export_svg(paths, width, height, "outputs/1.svg",
           foreground="#000000",
           background="white")
```

### `Params` fields

| Field | Default | Description |
|---|---|---|
| `threshold` | `128` | Grayscale binarization cutoff |
| `turd_size` | `2` | Minimum contour area (px²) |
| `alpha_max` | `1.0` | Smoothness: 0 = all corners, 1.333 = fully smooth |
| `opti_curve` | `True` | Enable RDP simplification before fitting |
| `opti_tolerance` | `0.2` | RDP epsilon × 3 |
| `scale` | `1.0` | Output scale factor |
| `invert` | `False` | Invert bitmap |
| `corner_arc_radius_sharp` | `0.42` | Arc radius at sharp corners (dot < 0.5) |
| `corner_arc_radius_soft` | `0.92` | Arc radius at soft corners (dot 0.5–0.85) |

---

## Configuration

All default parameters live in `configs/default.yaml`. Both `tracer.py` and `yd_vector/tracer.py` read from this file via `Params.from_yaml()`.

```yaml
preprocessing:
  threshold: 128        # ignored when --otsu is used
  otsu: false
  invert: false
  close: 0              # morphological close kernel (pixels)

pipeline:
  turdsize: 2           # suppress speckles <= this area (px^2)
  alphamax: 1.0         # corner threshold: 0=all corners, 4=all curves
  opttolerance: 0.2     # curve-merge tolerance in pixels
  optcurve: true        # enable optiCurve pass
  turnpolicy: minority  # minority|majority|right|black|white
  tension: 0.55         # bezier control-arm scale (higher = fuller curves)
  corner_arc_radius_sharp: 0.42   # arc radius at sharp corners (dot < 0.5)
  corner_arc_radius_soft:  0.92   # arc radius at soft  corners (dot 0.5–0.85)

output:
  fill: "#000000"
  bg: null              # null = transparent
  scale: 1.0

preprocess:
  preprocess_enabled: false
  upscale_scale: 4
  sam_model: sam2_hiera_large
  sam_points_per_side: 64
  sam_iou_thresh: 0.90
  color_merge_threshold: 12
  n_colors: 8
```

Edit `configs/default.yaml` to change defaults globally. CLI flags always override YAML values.

---

## ML Pre-processing (Real-ESRGAN + SAM 2.1)

Implemented in `ml_preprocess.py`, invoked by `tracer.py --preprocess`.

```
input image
    │
    ▼  Real-ESRGAN x4plus  (fp16, CUDA:0)
4× upscale — reveals boundary detail hidden at original resolution
    │
    ▼  SAM 2.1 hiera_large  (auto-mask generation)
segments image into per-region binary masks
    │
    ▼  LAB colour merge  (euclidean distance < color_merge_threshold)
merges visually similar regions into fewer layers
    │
    ▼  list of (hex_color, binary_mask) tuples  (up to n_colors layers)
    │
    ▼  PurePythonTracer.trace_layers()
full Potrace pipeline on each layer independently
    │
    ▼  layered SVG  (one <path> per colour, light→dark stacking order)
```

Model weights are downloaded automatically on first `--preprocess` run:

```
models/RealESRGAN_x4plus.pth    ~67 MB
models/sam2.1_hiera_large.pt   ~898 MB
```

**Graceful fallback:** if torch / realesrgan / SAM are not installed, upscaling falls back to Lanczos4 and segmentation falls back to k-means — both with a `RuntimeWarning` showing the install command.

---

## Architecture

### Repository layout

```
First-Stage-SVG/
│
├── tracer.py              # PRIMARY: 8-stage Potrace pipeline + CLI
├── ypotrace.py            # ALT: Schneider least-squares Bézier fitter + CLI
├── ml_preprocess.py       # ML front-end: Real-ESRGAN + SAM 2.1
│
├── configs/
│   └── default.yaml       # shared parameters for all pipelines
│
├── yd_vector/             # package — now delegates core tracing to tracer.py
│   ├── tracer.py          # Catmull-Rom smooth path library (smooth_path)
│   ├── bezier.py          # corner detection, cubic Bézier fitting, micro-arc SVG
│   ├── svg_builder.py     # multi-contour SVG with winding + fill-rule=evenodd
│   ├── pipeline.py        # run_trace(TraceOptions) → delegates to PurePythonTracer
│   ├── fit.py             # DEPRECATED: straight-line fitting (kept for compat)
│   ├── contours.py        # cv2.findContours wrapper
│   ├── simplify.py        # Douglas-Peucker contour simplification
│   ├── preprocess.py      # image load / threshold / morphological clean
│   ├── svg_writer.py      # legacy L-only SVG writer (backward compat)
│   ├── models.py          # TraceOptions, TraceResult, ContourData dataclasses
│   ├── config.py          # YAML loader utility
│   ├── utils.py           # ensure_parent_dir, clamp_int
│   └── cli.py             # argument parser for yd_vector CLI
│
├── scripts/
│   └── trace.py           # thin CLI wrapper around PurePythonTracer
│
├── tests/
│   ├── test_contours.py   # bitmap path-walk + legacy contour tests
│   └── test_svg_writer.py # end-to-end SVG output tests
│
├── inputs/                # sample input images
├── outputs/               # generated SVG files
└── models/                # auto-downloaded ML weights (gitignored)
```

### `tracer.py` — stage-by-stage pipeline

```
cv2.imread / cv2.threshold
          │
          ▼  internal Bitmap class (NOT cv2.findContours)
   Stage 1 — bm_to_pathlist
   pixel-walk boundary tracing → list of closed Path objects
   XOR fill to classify outer shapes vs. holes (sign = "+" or "−")
          │
          ▼
   Stage 2 — calc_sums
   prefix-sum tables: x, y, xy, x², y²
   enables O(1) evaluation of the quadratic line-fit penalty
          │
          ▼
   Stage 3 — calc_lon
   longest-monotone-run table per vertex
   bounds the DP search window in Stage 4
          │
          ▼
   Stage 4 — best_polygon
   dynamic-programming shortest polygon via penalty3()
   finds the fewest polygon vertices that faithfully approximate the boundary
          │
          ▼
   Stage 5 — adjust_vertices
   least-squares sub-pixel vertex optimisation
   3×3 quadratic forms + PCA eigen-direction → vertices snap to pixel edges
          │
          ▼
   Stage 6 — smooth  (tension=0.55)
   alpha-penalty corner detection → tag each segment CURVE or CORNER
   CURVE: assign Bézier control points using tension-scaled arm length
          arm = 0.5 + min(alpha/1.333, 1.0) × 0.55
   CORNER: record vertex for micro-arc in Stage 8
          │
          ▼
   Stage 7 — opti_curve  [optional, --no-optcurve to skip]
   merge adjacent CURVE segments into a single Bézier
   where the deviation stays within opttolerance pixels
          │
          ▼
   Stage 8 — get_svg
   CURVE  segment → C cp1x cp1y, cp2x cp2y, ex ey
   CORNER segment → L entry  A rx ry 0 0 sweep exit  L anchor
   all paths joined into one <path fill-rule="evenodd"> per colour
```

### `yd_vector/tracer.py` — Catmull-Rom path builder

```
cv2.findContours (RETR_CCOMP)
          │
          ▼  cv2.approxPolyDP (RDP simplification)
simplified polygon points
          │
          ▼  smooth_path()
   for each vertex i, compute Catmull-Rom control points from p0..p3
   if control-point deviation d1,d2 < 0.5px → straight segment branch:
       compute incoming (p2−p1) and outgoing (p3−p2) unit vectors
       dot product → choose arc radius (0.42 sharp / 0.92 soft / L if ≥0.85)
       emit:  L pre_point  A rx ry 0 0 sweep post_point
   else → emit C cubic Bézier
          │
          ▼  export_svg()
   <path fill-rule="evenodd"> per path (outer + holes merged)
```

### `ypotrace.py` — Schneider pipeline

```
PIL.Image / cv2.threshold
          │
          ▼  cv2.findContours (RETR_TREE)
raw contour pixel points
          │
          ▼  find_optimal_polygon()
greedy longest-valid-segment approximation (Potrace-style)
          │
          ▼  is_corner()  per polygon vertex
angle-change > alphamax_deg → CORNER (use L)
          │
          ▼  fit_bezier()  per segment
Schneider chord-length parameterisation + C matrix least-squares
→ alpha, beta arm lengths for P1 = P0 + alpha·T_out,  P2 = P3 − beta·T_in
          │
          ▼  SVG serialisation
L for corner–corner segments, C for all others
single <path fill-rule="evenodd">
```

---

## SVG Output

### Commands produced

| Command | When emitted |
|---|---|
| `M x y` | Start of each sub-path |
| `C cp1x cp1y, cp2x cp2y, x y` | Smooth CURVE segment (Potrace Stage 8 / Catmull-Rom) |
| `L x y` | CORNER approach, degenerate, or nearly-straight segment |
| `A rx ry 0 0 sweep x y` | Micro-arc at CORNER (replaces hard angle) |
| `Z` | Close each sub-path |

### Corner micro-arc geometry

At every CORNER the path uses a micro-arc instead of a sharp angle:

```
L (corner − 0.8px × incoming_unit)           ← entry point
A r r 0 0 sweep (corner + 0.8px × outgoing_unit)  ← arc
L next_anchor
```

| Condition | Arc radius |
|---|---|
| dot product < 0.5 (sharp) | r = 0.42 px |
| dot product 0.5–0.85 (soft) | r = 0.92 px |
| dot product ≥ 0.85 (straight) | plain `L`, no arc |

Sweep direction is determined by the sign of the cross product of the two direction vectors.
Both radii are configurable in `configs/default.yaml` and scale with `--size`.

---

## File Reference

| File | Role |
|---|---|
| `tracer.py` | `PurePythonTracer` class, `Params` dataclass, all 8 Potrace stages, CLI |
| `ypotrace.py` | Standalone CLI — Schneider least-squares + Potrace polygon approximation |
| `ml_preprocess.py` | `upscale()`, `segment_with_sam()`, `preprocess_pipeline()`, graceful CPU fallback |
| `configs/default.yaml` | All tunable parameters shared by both `tracer.py` and `yd_vector/tracer.py` |
| `yd_vector/tracer.py` | `smooth_path()` Catmull-Rom library, `Params.from_yaml()` |
| `yd_vector/bezier.py` | `contour_to_svg_path()`, `detect_corners()`, `fit_cubic_bezier()`, circle detection |
| `yd_vector/svg_builder.py` | `build_svg()` / `write_svg()` — winding enforcement + evenodd compound path |
| `yd_vector/pipeline.py` | `run_trace(TraceOptions)` — delegates to `PurePythonTracer` |
| `yd_vector/fit.py` | **Deprecated.** Straight-line fitting, kept for backward compatibility |
| `scripts/trace.py` | CLI wrapper: `python scripts/trace.py input.png output.svg [options]` |

---

## Tests

```bash
python -m pytest tests/ -v
```

Expected output:

```
tests/test_contours.py::test_extract_contours_runs               PASSED
tests/test_contours.py::test_bitmap_pathwalk                     PASSED
tests/test_contours.py::test_bitmap_pathwalk_no_contours_on_blank PASSED
tests/test_svg_writer.py::test_contour_to_path_d                 PASSED
tests/test_svg_writer.py::test_tracer_end_to_end_circle          PASSED
tests/test_svg_writer.py::test_tracer_end_to_end_square          PASSED
tests/test_svg_writer.py::test_tracer_debug_mode_produces_json   PASSED

7 passed
```
