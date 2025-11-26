"""
MediaPipe Hand Landmarker wrapper for TFLite models.

Provides MediaPipeHandLandmarker class which mirrors the style of other
mediapipe loaders in the codebase (face/pose). The loader expects a
TFLite model that outputs (1, 21, 2) keypoints normalized to crop.

Keypoint indices (common MediaPipe hand layout):
 0: wrist
 1-4: thumb (1..4 with 4 = thumb tip)
 5-8: index (8 = index tip)
 9-12: middle (12 = middle tip)
 13-16: ring (16 = ring tip)
 17-20: pinky (20 = pinky tip)

The loader exposes:
  - detect_hand_landmarks(frame, hand_bbox) -> List[Tuple[x,y]]

If the model or interpreter fails to initialize, methods return empty lists
and log warnings — callers should handle missing results gracefully.
"""

from typing import List, Tuple, Optional
import logging
import numpy as np
import cv2
import os

logger = logging.getLogger(__name__)

try:
    # Lazy import to keep runtime flexible
    import tflite_runtime.interpreter as tflite
except Exception:
    try:
        import tensorflow as tf
        tflite = tf.lite
    except Exception:
        tflite = None


class MediaPipeHandLandmarker:
    """Lightweight TFLite hand landmarker wrapper.

    Args:
        model_path: path to the TFLite hand landmark model (expects 21 keypoints)
        config: optional dict for inference settings (input_size etc.)
    """

    def __init__(self, model_path: str, config: Optional[dict] = None):
        self.model_path = model_path
        self._config = config or {}
        self._interp = None
        self._input_details = None
        self._output_details = None
        self._input_size = tuple(self._config.get("input_size", (224, 224)))

        if not os.path.isfile(self.model_path):
            logger.warning("Hand landmarker model not found: %s", self.model_path)
            return

        if tflite is None:
            logger.warning("TFLite runtime not available; hand landmarker disabled")
            return

        try:
            self._interp = tflite.Interpreter(model_path=self.model_path)
            self._interp.allocate_tensors()
            self._input_details = self._interp.get_input_details()
            self._output_details = self._interp.get_output_details()
            # Try to infer input size
            try:
                s = self._input_details[0]["shape"]
                if len(s) >= 3:
                    self._input_size = (int(s[-2]), int(s[-1]))
            except Exception:
                pass
        except Exception as e:
            logger.exception("Failed to load hand landmarker model: %s", e)
            self._interp = None

    def is_available(self) -> bool:
        return self._interp is not None

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Resize and normalize crop to model input size.

        MediaPipe TFLite tasks generally expect RGB float32 in 0..1 range.
        """
        target_w, target_h = self._input_size
        crop_resized = cv2.resize(crop, (target_w, target_h))
        # Ensure RGB ordering
        if crop_resized.shape[2] == 3:
            img = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        else:
            img = crop_resized
        arr = img.astype(np.float32) / 255.0
        # Expand batch
        return np.expand_dims(arr, axis=0)

    def _safe_invoke(self):
        try:
            self._interp.invoke()
            return True
        except Exception as e:
            logger.exception("Hand landmarker invocation failed: %s", e)
            return False

    def detect_hand_landmarks(self, frame: np.ndarray, hand_bbox: Tuple[float, float, float, float]) -> List[Tuple[float, float]]:
        """Detect hand landmarks in a specified normalized bbox.

        Args:
            frame: HxWxC BGR image (numpy)
            hand_bbox: normalized (xmin, ymin, w, h) relative to frame

        Returns:
            List of (x, y) tuples normalized to full-frame coordinates (0..1).
            Returns empty list on failure or if no keypoints detected.
        """
        if not self.is_available():
            return []

        h, w = frame.shape[:2]
        xmin, ymin, bw, bh = hand_bbox
        # Convert normalized bbox to pixel crop coords
        x0 = int(max(0, xmin * w))
        y0 = int(max(0, ymin * h))
        x1 = int(min(w, (xmin + bw) * w))
        y1 = int(min(h, (ymin + bh) * h))

        if x1 <= x0 or y1 <= y0:
            return []

        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return []

        inp = self._preprocess(crop)
        # Set input tensor
        try:
            self._interp.set_tensor(self._input_details[0]['index'], inp)
        except Exception:
            # Some runtimes expose set_tensor differently; try fallback
            try:
                self._interp.tensor(self._input_details[0]['index'])()[...] = inp
            except Exception as e:
                logger.exception("Failed to set input tensor for hand landmarker: %s", e)
                return []

        if not self._safe_invoke():
            return []

        # Read output - support a few possible output naming/ordering patterns
        out = None
        try:
            # Primary: single output (1, 21, 2)
            od = self._output_details[0]
            out = self._interp.get_tensor(od['index'])
        except Exception:
            # Try to gather outputs across different output indices
            outs = []
            for od in self._output_details:
                try:
                    outs.append(self._interp.get_tensor(od['index']))
                except Exception:
                    pass
            if len(outs) == 1:
                out = outs[0]

        if out is None:
            logger.warning("Hand landmarker returned no output tensors")
            return []

        # Expect out shape (1, 21, 2) or (21, 2)
        if out.ndim == 3 and out.shape[0] == 1:
            kp_arr = out[0]
        elif out.ndim == 2 and out.shape[0] == 21:
            kp_arr = out
        else:
            # Try to flatten or reshape if possible
            try:
                kp_arr = out.reshape(-1, 2)
            except Exception:
                logger.warning("Unexpected hand output shape: %s", out.shape)
                return []

        # Map crop-relative normalized coords to full-frame normalized coords
        kp_list = []
        crop_h = (y1 - y0)
        crop_w = (x1 - x0)
        if crop_w == 0 or crop_h == 0:
            return []

        for pt in kp_arr:
            # Some models provide (x,y) in 0..1 relative to crop
            try:
                nx = float(pt[0])
                ny = float(pt[1])
            except Exception:
                continue
            fx = (x0 + nx * crop_w) / w
            fy = (y0 + ny * crop_h) / h
            kp_list.append((fx, fy))

        return kp_list


# Simple helper for grip heuristic (exported for reuse)

def compute_grip_confidence_from_keypoints(kps: List[Tuple[float, float]], frame_size: Tuple[int, int]=None) -> float:
    """Compute a heuristic grip confidence from hand keypoints.

    Heuristic:
      - If no keypoints -> 0.0
      - Ensure wrist present (kp 0) and at least 3 fingertips visible (4,8,12,16,20)
      - Fingertips cluster vertically (y stddev small) and horizontal spread indicates wrapping

    Returns value in [0,1].
    """
    if not kps:
        return 0.0
    try:
        # fingertips indices
        fingertips = [4, 8, 12, 16, 20]
        wrist = kps[0]
        fingertip_pts = [kps[i] for i in fingertips if i < len(kps)]
        if len(fingertip_pts) < 1:
            return 0.0
        ys = np.array([p[1] for p in fingertip_pts])
        xs = np.array([p[0] for p in fingertip_pts])

        y_std = float(np.std(ys))
        x_spread = float(np.max(xs) - np.min(xs)) if len(xs) > 0 else 0.0

        # base score for detection
        score = 0.5
        if len(fingertip_pts) >= 3:
            score += 0.3
        # tighter vertical clustering increases score
        if y_std < 0.08:
            score += 0.1
        # wider horizontal spread suggests wrapping wheel
        if x_spread > 0.12:
            score += 0.1

        return min(1.0, max(0.0, score))
    except Exception:
        return 0.0
