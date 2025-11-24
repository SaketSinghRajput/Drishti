# Driver Drowsiness Detection (integration)

This directory contains integration code and a lightweight detection engine
for driver drowsiness detection. The implementation prefers MediaPipe
Face Mesh for landmark extraction and can fall back to simpler heuristics
if MediaPipe is not available in the environment.

If you want to use the external research code referenced in the project,
clone it into the `external_repo/` directory here:

```bash
cd data_collection/sensors/driver_drowsiness
git clone https://github.com/neelanjan00/Driver-Drowsiness-Detection.git external_repo
```

Notes:
- This repository is not included by default; cloning is optional.
- The detection modules here use MediaPipe when present; for CI we mock
  MediaPipe to keep tests lightweight.

See `detection_engine.py` and `camera_interface.py` for details.
