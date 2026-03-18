from __future__ import annotations
import sys
import os
import numpy as np
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracer import PurePythonTracer
from yd_vector.layers import bgr_to_hex

def trace_layer(
    mask: np.ndarray,
    alphamax: float = 1.0,
    opttolerance: float = 0.2,
    turdsize: int = 2,
    opticurve: bool = True,
) -> str:
    """
    Run PurePythonTracer on a single binary mask.

    Args:
        mask: uint8 numpy array (H x W), 255 = foreground
        alphamax: corner smoothness (passed to PurePythonTracer)
        opttolerance: curve merge tolerance (passed to PurePythonTracer)
        turdsize: speckle suppression (passed to PurePythonTracer)
        opticurve: enable optiCurve pass (passed to PurePythonTracer)

    Returns:
        SVG path d= string with M, C, L, Z commands only.
        Returns empty string "" if no paths found.
    """
    tracer = PurePythonTracer(
        turdsize=turdsize,
        alphamax=alphamax,
        opttolerance=opttolerance,
        optcurve=opticurve,
        turnpolicy="minority"
    )
    # The mask is uint8 255/0, we need to pass it
    svg_str = tracer.trace(mask)

    start = svg_str.find('d="')
    if start == -1:
        return ""
    start += 3
    end = svg_str.find('"', start)
    if end == -1:
        return ""
    return svg_str[start:end]

def build_color_svg(
    layers: List[Tuple[Tuple[int,int,int], np.ndarray]],
    width: int,
    height: int,
    alphamax: float = 1.0,
    opttolerance: float = 0.2,
    turdsize: int = 2,
    opticurve: bool = True,
    scale: float = 1.0,
    background: Optional[str] = None,
) -> str:
    """
    Trace all layers and compose into a single multi-layer SVG.

    Args:
        layers: list of (color_bgr, binary_mask) from layers.py
        width, height: original image dimensions
        alphamax, opttolerance, turdsize, opticurve: tracer parameters
        scale: output scale factor
        background: optional hex color for background rect (e.g. "#ffffff")

    Returns:
        Complete SVG string with one <path> element per color layer.
        Layers are ordered bottom to top (most frequent color at bottom).
        Each path has fill=hex_color and fill-rule="evenodd".
    """
    sw_w, sw_h = int(width * scale), int(height * scale)
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{sw_w}" height="{sw_h}" viewBox="0 0 {sw_w} {sw_h}">'
    ]
    if background:
        out.append(f'  <rect width="100%" height="100%" fill="{background}"/>')

    for i, (color_bgr, mask) in enumerate(layers):
        hex_color = bgr_to_hex(color_bgr)
        print(f"[compositor] Tracing layer {i+1}/{len(layers)} (color {hex_color})")
        d_str = trace_layer(mask, alphamax, opttolerance, turdsize, opticurve)
        if not d_str:
            continue
        
        # Apply scale using g transform if scale != 1.0
        scale_transform = f' transform="scale({scale})"' if scale != 1.0 else ""
        out.append(f'  <g id="layer-{i}" fill="{hex_color}"{scale_transform}>')
        out.append(f'    <path d="{d_str}" fill-rule="evenodd" stroke="none"/>')
        out.append('  </g>')

    out.append('</svg>')
    return '\n'.join(out)
