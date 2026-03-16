# YD-Vector

Minimal CLI-first bitmap-to-SVG tracer.

## Features
- Load raster image
- Convert to grayscale
- Threshold to binary
- Extract contours
- Simplify contour points
- Export SVG

## Install

```bash
python -m venv .venv
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt