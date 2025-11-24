"""Driver drowsiness detection package.

This package contains a lightweight CV-based drowsiness detection engine
that uses MediaPipe Face Mesh when available and falls back to simple
heuristics when not. The external reference implementation is not bundled
— see README for instructions to clone an external repo if desired.
"""

__all__ = ["DrowsinessDetector", "CameraInterface"]
