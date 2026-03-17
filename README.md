# First-Stage-SVG

> **Potrace-style bitmap→SVG tracer in pure Python.**
> No C-bindings. No `pypotrace`. No Douglas-Peucker.

---

## Architecture

The tracer implements the pipeline from Peter Selinger's 2003 paper
*"Potrace: a polygon-based tracing algorithm"*, faithfully translated from
the open-source JavaScript port [kilobtye/potrace](https://github.com/kilobtye/potrace).

```
raster image
    │
    ▼  cv2.imread / cv2.threshold / optional morphologyEx
binary bitmap (numpy array)
    │
    ▼  Stage 1 — bm_to_pathlist
pixel-walk path decomposition  (pure Python Bitmap class, ≠ cv2.findContours)
    │
    ▼  Stage 2 — calc_sums
prefix-sum tables  (x, y, x², xy, y² for O(1) line-fit penalty)
    │
    ▼  Stage 3 — calc_lon
longest-monotone-run table per vertex
    │
    ▼  Stage 4 — best_polygon
DP shortest polygon  (penalty3 quadratic form)
    │
    ▼  Stage 5 — adjust_vertices
sub-pixel vertex optimisation  (3×3 quadratic forms + PCA eigen-direction)
    │
    ▼  Stage 6 — smooth
corner detection (alpha penalty) + cubic Bézier control-point assignment
    │
    ▼  Stage 7 — opti_curve  [optional]
curve merging  (reduce segment count within opttolerance)
    │
    ▼  Stage 8 — get_svg
SVG serialisation  (M / L / C / Z, fill-rule=evenodd)
```

> **OpenCV usage:** `imread`, `threshold`, `bitwise_not`, `morphologyEx` (optional),
> `kmeans` (multi-color mode only).
> **All tracing geometry is pure Python.** `cv2.findContours` is never called by the tracer.

---

## Install

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

### Main tracer (`tracer.py`)

```bash
# Simplest — dark shapes on light background
python tracer.py input.png output.svg

# Use OTSU auto-threshold + invert + morph-close to bridge gaps
python tracer.py inputs/1.jpg outputs/1.svg --otsu --invert --close 5

# Fine-tune curve quality
python tracer.py logo.png out.svg --alphamax 0.8 --opttolerance 0.1

# Debug: print intermediate pipeline state as JSON to stdout
python tracer.py input.png out.svg --debug

# Multi-color mode (k-means quantization, one layer per color)
python tracer.py photo.jpg out.svg --colors 8
```

### Script wrapper (`scripts/trace.py`)

```bash
# Same flags, same pipeline — convenience wrapper
python scripts/trace.py inputs/1.jpg outputs/1.svg --otsu --invert --close 5 --debug
```

---

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--threshold N` | `128` | Grayscale binarization threshold (0–255) |
| `--otsu` | off | OTSU automatic threshold (overrides `--threshold`) |
| `--invert` | off | Invert bitmap (trace dark shapes on light background) |
| `--close N` | `0` | Morphological close kernel size (pixels); bridges thin gaps |
| `--turdsize N` | `2` | Suppress speckles ≤ N px² |
| `--alphamax F` | `1.0` | Corner threshold: lower → more corners; higher → smoother |
| `--opttolerance F` | `0.2` | Curve-merge tolerance (pixels) |
| `--no-optcurve` | off | Disable optiCurve pass |
| `--turnpolicy P` | `minority` | Ambiguous-turn policy: minority/majority/right/black/white |
| `--fill COLOR` | `#000000` | SVG foreground fill color |
| `--bg COLOR` | _(transparent)_ | Optional SVG background rect color |
| `--size F` / `--scale F` | `1.0` | Output scale factor |
| `--colors N` | `0` | Multi-color mode: k-means to N colors, one Potrace layer each |
| `--debug` | off | Print JSON intermediate state (paths, polygons, vertices, tags) |

---

## Legacy note

The `yd_vector/` package contains the original `cv2.findContours`-based tracer.
It is retained for test coverage only. **Do not use it for new work.**

---

## Tests

```bash
python -m pytest tests/ -v
```