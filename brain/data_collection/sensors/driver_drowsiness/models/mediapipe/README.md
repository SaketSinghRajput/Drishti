MediaPipe TFLite models for driver drowsiness detection

Place the following MediaPipe TFLite models here (not included in this repository):

- face_detection_short_range.tflite  # fast face detector (input ~192x192)
- face_landmark_lite.tflite          # lite face landmark model (468 landmarks)
- iris_landmark.tflite               # iris landmark model (per-eye)

These models can be downloaded from the official MediaPipe releases or exported using the MediaPipe Tasks tooling. The helper `brain/utils/mediapipe_loaders.py` expects normalized output tensors matching MediaPipe task exports.

Quick download helper:

- Create `scripts/mediapipe_model_urls.json` (copy `scripts/mediapipe_model_urls.sample.json`) and verify URLs.
- Run the helper: `python scripts/download_mediapipe_models.py` or in PowerShell: `.\scripts\download_mediapipe_models.ps1`.

Model I/O specification

- face_detection_short_range.tflite
	- Input: `float32` tensor shape `[1, 192, 192, 3]`, values normalized to `[0.0, 1.0]` (RGB)
	- Output: commonly two tensors: boxes `[1, N, 4]` where each box is `[ymin, xmin, ymax, xmax]` normalized (0..1) and scores `[1, N]` (0..1). Some exports combine score in the last dim as `[1,N,5]`.

- face_landmark_lite.tflite
	- Input: `float32` tensor shape `[1, 192, 192, 3]`, values normalized to `[0.0, 1.0]` (RGB) on a crop of the face.
	- Output: typically `[1, 468, 3]` with `(x,y,z)` per-landmark normalized to the face crop (x,y in 0..1). This loader maps crop-relative coords back to full-frame normalized coords.

- iris_landmark.tflite
	- Input: `float32` tensor shape `[1, H, W, 3]` (common exports use 64x64 eye crops), normalized to `[0.0, 1.0]`.
	- Output: small set of precise iris landmarks `[1, M, 2]` or `[1, M, 3]` from which the loader computes an iris center normalized to the full frame.

Checksum verification

Add expected `sha256` values here after you download canonical models. Run `scripts/verify_mediapipe_models.py` (added in the repo) to compute and display the SHA256 for each model; paste the values below for reproducible verification.

- face_detection_short_range.tflite: <sha256-placeholder>
- face_landmark_lite.tflite: <sha256-placeholder>
- iris_landmark.tflite: <sha256-placeholder>

Notes

- The project helper `brain/utils/mediapipe_loaders.py` includes `preprocess_for_mediapipe()` (resize + normalize) used by the loaders; ensure model inputs are `[0..1]` floats and RGB ordering.
- If your export produces different tensor shapes or names, adapt `brain/utils/mediapipe_loaders.py` parsing heuristics.
face_detection_short_range.tflite: BBFF11CEBD1EB27A1E004CAE0B0E63EC8C551CBF34A4451148B4908B8DB3ECA8
face_landmark_lite.tflite: 1055CB9D4A9CA8B8C688902A3A5194311138BA256BCC94E336D8373A5F30C814
iris_landmark.tflite: D1744D2A09C25F501D39EBA4FAFF47E53ECCA8852C5CE19BCE8EEAC39357521F
face_detection_short_range.tflite: BBFF11CEBD1EB27A1E004CAE0B0E63EC8C551CBF34A4451148B4908B8DB3ECA8
face_landmark_lite.tflite: 1055CB9D4A9CA8B8C688902A3A5194311138BA256BCC94E336D8373A5F30C814
iris_landmark.tflite: D1744D2A09C25F501D39EBA4FAFF47E53ECCA8852C5CE19BCE8EEAC39357521F
