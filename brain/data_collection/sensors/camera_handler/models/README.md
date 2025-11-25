MediaPipe TFLite models for external camera and pedestrian pose

Place the following models here (not included in this repository):

- face_detection_full_range.tflite    # full-range face detector for longer distances
- pose_detection.tflite              # person detection model (bbox)
- pose_landmark_full.tflite          # pose landmark model (33 keypoints)

These models are used by `brain/utils/mediapipe_loaders.py` and the camera handler when `enable_pose_detection` is enabled.

Quick download helper:

- Create `scripts/mediapipe_model_urls.json` (copy `scripts/mediapipe_model_urls.sample.json`) and verify URLs.
- Run the helper: `python scripts/download_mediapipe_models.py` or in PowerShell: `.\scripts\download_mediapipe_models.ps1`.

Model I/O specification

- pose_detection.tflite
	- Input: `float32` tensor shape `[1, 256, 256, 3]` (RGB), values normalized to `[0.0, 1.0]`.
	- Output: typically `[1, N, 5]` or `[1, N, 4]` rows where rows are `[ymin, xmin, ymax, xmax, score]` normalized to the frame.

- pose_landmark_full.tflite
	- Input: `float32` tensor shape `[1, H, W, 3]` (crop around person), normalized to `[0.0, 1.0]`.
	- Output: typically `[1, 33, 3]` (x,y,z) keypoints normalized to the crop; `brain/utils/mediapipe_loaders.py` maps these back to full-frame normalized coords.

Checksum verification

Run `scripts/verify_mediapipe_models.py` to compute SHA256 checksums for downloaded models and paste the canonical checksums here once verified.

- pose_detection.tflite: <sha256-placeholder>
- pose_landmark_full.tflite: <sha256-placeholder>

Notes

- `camera_handler` will only initialize pose loaders when `mediapipe.pose.enabled` (or `sensors.camera.enable_pose_detection`) is set to true in `config/default_config.yaml`.
- If model outputs differ from the shapes above, update `brain/utils/mediapipe_loaders.py` parsing heuristics.
pose_detection.tflite: 9BA9DD3D42EFAABA86B4FF0122B06F29C4122E756B329D89DCA1E297FD8F866C
pose_landmark_full.tflite: E9A5C5CB17F736FAFD4C2EC1DA3B3D331D6EDBE8A0D32395855AEB2CDFD64B9F
pose_detection.tflite: 9BA9DD3D42EFAABA86B4FF0122B06F29C4122E756B329D89DCA1E297FD8F866C
pose_landmark_full.tflite: E9A5C5CB17F736FAFD4C2EC1DA3B3D331D6EDBE8A0D32395855AEB2CDFD64B9F
