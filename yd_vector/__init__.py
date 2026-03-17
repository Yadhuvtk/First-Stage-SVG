# LEGACY: This package was the original cv2.findContours-based tracer.
# It has been superseded by the Potrace-style bitmap-walk pipeline in tracer.py.
# Use: python tracer.py input.png output.svg
#
# This package is retained for backward-compatibility and test coverage only.
# All new tracing should go through PurePythonTracer in tracer.py.

import warnings as _warnings
_warnings.warn(
    "yd_vector is a legacy package. Use 'tracer.py' (PurePythonTracer) instead.",
    DeprecationWarning,
    stacklevel=2,
)