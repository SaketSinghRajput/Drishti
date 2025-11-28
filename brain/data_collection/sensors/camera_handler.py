"""Camera sensor handler.

Simulation-first implementation: generates synthetic detections when no model_path is provided
or when enable_simulation=True in config. Designed to never return raw frames—only metadata.
"""
from __future__ import annotations

import logging
import threading
import time
import random
from typing import Any, Dict, List, Optional

from config.config_loader import ConfigLoader
from data_collection.sensors.base import SensorHandler
from utils import camera_utils
from utils.model_loader import load_model
import importlib

_cv2 = None

def _ensure_cv2():
    global _cv2
    if _cv2 is None:
        try:
            import cv2 as _c

            _cv2 = _c
        except Exception:
            _cv2 = None
    return _cv2

logger = logging.getLogger(__name__)


class CameraHandler(SensorHandler):
    """Camera sensor handler. Simulation by default.

    Returns metadata-only dictionaries suitable for WindowManager.
    """

    def __init__(self, config: ConfigLoader):
        self.config = config
        self.cfg = config.get("sensors.camera", {})

        # helpers for robust numeric parsing from config (tolerant of strings/comments)
        def _safe_int(v, default: int = 15) -> int:
            try:
                return int(float(v))
            except Exception:
                try:
                    s = str(v)
                    import re

                    m = re.match(r"\s*([0-9]+)", s)
                    if m:
                        return int(m.group(1))
                except Exception:
                    pass
                return int(default)

        def _safe_float(v, default: float = 0.5) -> float:
            try:
                return float(v)
            except Exception:
                try:
                    s = str(v)
                    import re

                    m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", s)
                    if m:
                        return float(m.group(1))
                except Exception:
                    pass
                return float(default)

        # primary parameters
        self.fps = _safe_int(self.cfg.get("fps", 15))
        res = self.cfg.get("resolution", {})
        self.width = int(res.get("width", 640))
        self.height = int(res.get("height", 480))
        self.detection_confidence_threshold = _safe_float(self.cfg.get("detection_confidence_threshold", 0.5))
        # expose safe float helper for other methods
        self._safe_float = _safe_float

        self.model_type = self.cfg.get("model_type", "MobileNet-SSD")
        self.model_path = self.cfg.get("model_path", "")
        self.enable_simulation = bool(self.cfg.get("enable_simulation", True))
        self.enable_lane_detection = bool(self.cfg.get("enable_lane_detection", False))
        self.focal_length_px = _safe_float(self.cfg.get("focal_length_px", 500.0))
        self.known_object_sizes = self.cfg.get("known_object_sizes", None)

        self._model = None
        if not self.enable_simulation and self.model_path:
            try:
                # pass a plain dict (camera-specific config) to backends
                self._model = load_model(self.model_path, self.model_type, config=self.cfg)
                self._model.load()
            except Exception as e:
                logger.warning("Failed to load camera model, switching to simulation: %s", e)
                self._model = None
                self.enable_simulation = True

        # Optional MediaPipe TFLite pose backend initialization (config-gated)
        self.enable_pose_detection = bool(self.cfg.get("enable_pose_detection", False))
        self._mp_pose_detector = None
        # optional pose enrichment toggles
        self.enable_pedestrian_intent = bool(self.cfg.get("enable_pedestrian_intent", False))
        self.enable_cyclist_lean = bool(self.cfg.get("enable_cyclist_lean", False))
        self.pedestrian_intent_window = int(self.cfg.get("pedestrian_intent_velocity_window_frames", 5))
        # store previous keypoints per object index for velocity/intent computation
        self._prev_pose_keypoints = {}
        try:
            from data_collection.sensors.camera_handler_pose_enrichment import compute_pedestrian_intent, compute_cyclist_lean_angle
        except Exception:
            try:
                from .camera_handler_pose_enrichment import compute_pedestrian_intent, compute_cyclist_lean_angle
            except Exception:
                compute_pedestrian_intent = None
                compute_cyclist_lean_angle = None
            if self.enable_pose_detection:
                try:
                    from brain.utils.mediapipe_loaders import MediaPipePoseDetector

                    # Support nested mediapipe config under top-level `mediapipe.pose`
                    try:
                        mp_pose_cfg = self.config.get("mediapipe", {}).get("pose", {}) if hasattr(self.config, "get") else {}
                    except Exception:
                        mp_pose_cfg = {}

                    det_path = str(self.cfg.get("pose_detection_model_path", "")) or mp_pose_cfg.get("detection_model_path") or mp_pose_cfg.get("detection_model_path")
                    lm_path = str(self.cfg.get("pose_landmark_model_path", "")) or mp_pose_cfg.get("landmark_model_path") or mp_pose_cfg.get("landmark_model_path")
                    if det_path:
                        self._mp_pose_detector = MediaPipePoseDetector(det_path, landmark_path=lm_path, config=mp_pose_cfg or self.cfg)
                        logger.info("CameraHandler: initialized MediaPipePoseDetector (%s)", det_path)
                    else:
                        logger.warning("CameraHandler: pose_detection enabled but model path not provided")
                except Exception:
                    logger.exception("Failed to initialize MediaPipePoseDetector; pose detection disabled")
                    self._mp_pose_detector = None

        # persistent video capture (avoid opening/closing per-frame)
        self._cap = None

        # new backend/hardware accel options
        self.backend = str(self.cfg.get("backend", "auto"))
        self.enable_hardware_accel = bool(self.cfg.get("enable_hardware_accel", False))
        self.preferred_pixel_format = self.cfg.get("preferred_pixel_format", None)
        self.gstreamer_pipeline = self.cfg.get("gstreamer_pipeline", None)

        self._prev_frame_detections: Dict[str, Any] = {}
        self._frame_count = 0
        self._total_reads = 0
        self._successful_reads = 0
        self._inference_times: List[float] = []
        # retention counter to enforce max_retained_frames_per_trip
        self._retained_frames_count = 0
        self._lock = threading.Lock()
        # sample_rate attribute used by DataLogger sensor loop
        self.sample_rate = int(self.fps)

    def get_name(self) -> str:
        return "camera"

    def _ensure_capture(self):
        """Lazily open and configure a persistent cv2.VideoCapture instance.

        Returns the capture object or None if unavailable.
        """
        cv2 = _ensure_cv2()
        if cv2 is None:
            return None
        try:
            if self._cap is not None:
                try:
                    if self._cap.isOpened():
                        return self._cap
                except Exception:
                    # fallthrough to re-open
                    pass
            # open device by preferred backend
            device_index = int(self.cfg.get("device_index", 0))
            cap = None
            # prefer gstreamer pipeline if provided
            if self.gstreamer_pipeline:
                try:
                    cap = cv2.VideoCapture(self.gstreamer_pipeline, cv2.CAP_GSTREAMER)
                except Exception:
                    cap = None

            # choose V4L2 on linux when requested or when hardware accel enabled
            if cap is None:
                try:
                    if self.backend == "v4l2" or (self.backend == "auto" and self.enable_hardware_accel):
                        cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
                    elif self.backend == "gstreamer":
                        cap = cv2.VideoCapture(self.gstreamer_pipeline or device_index, cv2.CAP_GSTREAMER)
                    else:
                        cap = cv2.VideoCapture(device_index)
                except Exception:
                    cap = cv2.VideoCapture(device_index)

            if cap is None:
                return None

            # set resolution
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
            except Exception:
                pass

            # set preferred pixel format (e.g., MJPG) when requested
            if self.preferred_pixel_format and self.preferred_pixel_format.upper() == "MJPEG":
                try:
                    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                except Exception:
                    pass

            self._cap = cap
            return self._cap
        except Exception:
            logger.exception("Failed to initialize video capture")
            try:
                if self._cap is not None:
                    self._cap.release()
            except Exception:
                pass
            self._cap = None
            return None

    def close(self) -> None:
        """Release capture resources."""
        try:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
        finally:
            self._cap = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def is_available(self) -> bool:
        return self.enable_simulation or (self._model is not None)

    def get_health(self) -> Dict[str, Any]:
        with self._lock:
            avg_inf = float(sum(self._inference_times) / len(self._inference_times)) if self._inference_times else 0.0
            availability = float(self._successful_reads) / max(1, self._total_reads)
            # degraded if repeated fallback to simulation
            status = "ok"
            if getattr(self, "_fallback_to_simulation", False):
                status = "degraded"
            elif availability <= 0.5:
                status = "degraded"
            return {
                "status": status,
                "availability": availability,
                "avg_inference_time_ms": avg_inf * 1000.0,
                "frame_count": self._frame_count,
                "retained_frames_count": int(self._retained_frames_count),
            }

    def read(self) -> Optional[Dict[str, Any]]:
        ts = time.time()
        start = time.time()
        self._total_reads += 1
        try:
            if self.enable_simulation or self._model is None:
                data = self._simulate_frame(ts)
            else:
                data = self._infer_frame(ts)
            self._successful_reads += 1
            return data
        except Exception as e:
            logger.exception("Camera read failed: %s", e)
            return None
        finally:
            inf_time = time.time() - start
            with self._lock:
                self._inference_times.append(inf_time)
                if len(self._inference_times) > 500:
                    self._inference_times = self._inference_times[-500:]

    def _simulate_frame(self, ts: float) -> Dict[str, Any]:
        # create 0-5 synthetic detections
        num = random.randint(0, 5)
        objects: List[Dict[str, Any]] = []
        classes = ["car", "person", "bicycle", "truck"]

        for i in range(num):
            cls = random.choice(classes)
            w = random.randint(int(self.width * 0.05), int(self.width * 0.4))
            h = random.randint(int(self.height * 0.05), int(self.height * 0.4))
            x1 = random.randint(0, max(0, self.width - w))
            y1 = random.randint(int(self.height * 0.3), max(0, self.height - h))
            x2 = x1 + w
            y2 = y1 + h
            conf = round(random.uniform(0.4, 0.99), 2)
            bbox = [x1, y1, x2, y2]
            dist = camera_utils.estimate_distance_from_bbox(
                bbox, cls, self.focal_length_px, self.known_object_sizes
            )
            prev = self._prev_frame_detections.get(i, {}).get("distance_m", dist + random.uniform(-1.0, 1.0))
            rel_speed = camera_utils.estimate_relative_speed(dist, prev, 1.0 / max(1, self.fps))
            heading = random.uniform(-30.0, 30.0)
            obj = {
                "class": cls,
                "confidence": conf,
                "bbox_pixels": bbox,
                "distance_m": dist,
                "rel_speed_mps": rel_speed,
                "heading": heading,
            }
            objects.append(obj)
            # store for next frame
            self._prev_frame_detections[i] = {"distance_m": dist}

        # filter synthetic detections to match production filtering (apply before aggregates)
        objects = camera_utils.filter_low_confidence_detections(objects, self._safe_float(self.detection_confidence_threshold))
        objects = camera_utils.nms_detections(objects, self._safe_float(self.cfg.get("nms_iou_threshold", 0.5)))

        # compute lane offset based on in-path objects
        lane_center_offset = 0.0
        in_path_objs = [o for o in objects if camera_utils.is_in_path(o.get("bbox_pixels", []), self.width, self.height, self._safe_float(self.cfg.get("path_center_tolerance", 0.3)))]
        if in_path_objs:
            avg_cx = sum(((o["bbox_pixels"][0] + o["bbox_pixels"][2]) / 2) for o in in_path_objs) / len(in_path_objs)
            lane_center_offset = camera_utils.compute_lane_offset(int(avg_cx), self.width, self._safe_float(self.cfg.get("lane_width_m", 3.5)))

        # aggregates derived from filtered objects
        # aggregates derived from filtered objects (use the final, filtered objects list)
        objects_in_path = len(in_path_objs)
        closest = min((o["distance_m"] for o in objects), default=999.0)
        obstacle_flag = objects_in_path > 0
        pedestrian_near = any(o["class"] == "person" and o["distance_m"] < 10.0 for o in objects)

        self._frame_count += 1
        # determine retention flag via helper
        agg = {"closest_obj_dist_m": closest, "objects_in_path": objects_in_path}
        flagged = self.should_flag_frame(objects, agg)

        # lane occupancy: left/right based on center split
        tol = self._safe_float(self.cfg.get("path_center_tolerance", 0.3))
        left_bound = 0.5 - tol * 0.5
        right_bound = 0.5 + tol * 0.5
        left_blocking = any(((o["bbox_pixels"][0] + o["bbox_pixels"][2]) / 2.0) / float(self.width) < left_bound for o in objects)
        right_blocking = any(((o["bbox_pixels"][0] + o["bbox_pixels"][2]) / 2.0) / float(self.width) > right_bound for o in objects)

        metadata = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
            "objects": objects,
            "lane_center_offset_m": lane_center_offset,
            "is_left_lane_clear": not left_blocking,
            "is_right_lane_clear": not right_blocking,
            "obstacle_in_path_flag": obstacle_flag,
            "pedestrian_near_crossing_flag": pedestrian_near,
            "visibility_estimate_m": 1000.0,
            "traffic_light_state": "unknown",
            "closest_obj_dist_m": closest,
            "objects_in_path": int(objects_in_path),
            "frame_flagged_for_retention": bool(flagged),
        }

        if flagged:
            # call hook: simulation path remains metadata-only. For production
            # inference we pass the raw frame (see _infer_frame); here we keep
            # metadata-only behaviour to avoid accidental raw-frame storage.
            try:
                self.on_frame_flagged(metadata.copy())
            except Exception:
                logger.exception("on_frame_flagged hook raised an exception")

        return metadata

    def _infer_frame(self, ts: float) -> Dict[str, Any]:
        # Production inference: capture frame and run model.predict(frame)
        # Returns metadata-only dict identical in shape to _simulate_frame output.
        cv2 = _ensure_cv2()
        if cv2 is None:
            # OpenCV not available; fallback
            self._fallback_to_simulation = True
            return self._simulate_frame(ts)
        # ensure persistent capture device
        try:
            cap = self._ensure_capture()
            if cap is None:
                self._fallback_to_simulation = True
                return self._simulate_frame(ts)
            read_ok, frame = cap.read()
            if not read_ok or frame is None:
                # read failed -> fallback
                self._fallback_to_simulation = True
                return self._simulate_frame(ts)

            # run model inference
            try:
                preds = self._model.predict(frame)
            except Exception as e:
                logger.exception("Model prediction failed: %s", e)
                # mark fallback and simulate
                self._fallback_to_simulation = True
                return self._simulate_frame(ts)

            # preds expected to be list of dicts with keys 'class','confidence','bbox_pixels'
            # apply confidence filtering and NMS before computing derived metrics
            try:
                filtered = camera_utils.filter_low_confidence_detections(preds or [], float(self.detection_confidence_threshold))
                filtered = camera_utils.nms_detections(filtered, self._safe_float(self.cfg.get("nms_iou_threshold", 0.5)))
            except Exception:
                filtered = preds or []

            objects = []
            for i, p in enumerate(filtered):
                try:
                    cls = p.get("class")
                    conf = float(p.get("confidence", p.get("score", 0.0)))
                    bbox = p.get("bbox_pixels") or p.get("bbox")
                    if not bbox or len(bbox) < 4:
                        continue
                    # compute derived metrics
                    dist = camera_utils.estimate_distance_from_bbox(bbox, cls, self.focal_length_px, self.known_object_sizes)
                    prev = self._prev_frame_detections.get(i, {}).get("distance_m", dist)
                    rel_speed = camera_utils.estimate_relative_speed(dist, prev, 1.0 / max(1, self.fps))
                    heading = 0.0
                    obj = {
                        "class": cls,
                        "confidence": conf,
                        "bbox_pixels": [int(x) for x in bbox],
                        "distance_m": dist,
                        "rel_speed_mps": rel_speed,
                        "heading": heading,
                    }
                    objects.append(obj)
                    self._prev_frame_detections[i] = {"distance_m": dist}
                except Exception:
                    continue

                # If MediaPipe pose detector available and enabled, enrich person objects
                try:
                    if self._mp_pose_detector is not None:
                        for idx, obj in enumerate(objects):
                            try:
                                if obj.get("class") != "person":
                                    continue
                                # convert pixel bbox to normalized bbox
                                x1, y1, x2, y2 = obj.get("bbox_pixels", [0, 0, 0, 0])
                                fw = float(self.width)
                                fh = float(self.height)
                                xmin = max(0.0, min(1.0, float(x1) / fw))
                                ymin = max(0.0, min(1.0, float(y1) / fh))
                                w = max(0.0, min(1.0, float(x2 - x1) / fw))
                                h = max(0.0, min(1.0, float(y2 - y1) / fh))
                                bbox_norm = (xmin, ymin, w, h)
                                kps = self._mp_pose_detector.detect_landmarks(frame, bbox_norm)
                                if not kps:
                                    continue
                                # compute simple pose metrics: average visibility/confidence (if available)
                                try:
                                    # landmark format is list of (x,y) normalized
                                    avg_conf = 1.0
                                    # walking_towards heuristic: check if hip y < knee y (approaching camera -> smaller y)
                                    walking_towards = False
                                    rel_heading = 0.0
                                    # indices: shoulders 11/12, hips 23/24, knees 25/26
                                    def _safe(idx):
                                        return kps[idx] if idx < len(kps) else (0.0, 0.0)

                                    lh = _safe(23)
                                    rh = _safe(24)
                                    lk = _safe(25)
                                    rk = _safe(26)
                                    # average hips and knees y coordinate in normalized frame
                                    hip_y = ((lh[1] if lh else 0.0) + (rh[1] if rh else 0.0)) / 2.0
                                    knee_y = ((lk[1] if lk else 0.0) + (rk[1] if rk else 0.0)) / 2.0
                                    # if hip is higher (smaller y) than knee substantially, assume upright; walking_towards if hip_y < knee_y - small_margin
                                    walking_towards = hip_y < (knee_y - 0.02)
                                    rel_heading = 0.0
                                except Exception:
                                    walking_towards = False
                                    rel_heading = 0.0
                                # populate pose_data and optionally compute intent/lean
                                pd = {
                                    "keypoints": kps,
                                    "walking_towards": bool(walking_towards),
                                    "relative_heading": float(rel_heading),
                                    "confidence": float(avg_conf),
                                }
                                # pedestrian intent
                                try:
                                    if self.enable_pedestrian_intent and compute_pedestrian_intent is not None:
                                        prev_kps = self._prev_pose_keypoints.get(idx)
                                        intent = compute_pedestrian_intent(prev_kps, kps, camera_velocity_window=int(self.pedestrian_intent_window))
                                        pd['pedestrian_intent_score'] = float(intent)
                                except Exception:
                                    logger.exception('Failed to compute pedestrian intent')
                                # cyclist lean
                                try:
                                    if obj.get('class') in ('bicycle', 'bike', 'cyclist') and self.enable_cyclist_lean and compute_cyclist_lean_angle is not None:
                                        lean = compute_cyclist_lean_angle(kps)
                                        pd['cyclist_lean_angle'] = float(lean)
                                        pd['cyclist_maneuvering_flag'] = bool(lean > 15.0)
                                except Exception:
                                    logger.exception('Failed to compute cyclist lean')

                                obj['pose_data'] = pd
                                # update prev keypoints map
                                try:
                                    self._prev_pose_keypoints[idx] = kps
                                except Exception:
                                    pass
                            except Exception:
                                continue
                except Exception:
                    logger.exception("Error while enriching objects with pose data")

            # compute lane center offset from average of in-path objects
            lane_center_offset = 0.0
            in_path_objs = [o for o in objects if camera_utils.is_in_path(o.get("bbox_pixels", []), self.width, self.height, self._safe_float(self.cfg.get("path_center_tolerance", 0.3)))]
            if in_path_objs:
                avg_cx = sum(((o["bbox_pixels"][0] + o["bbox_pixels"][2]) / 2) for o in in_path_objs) / len(in_path_objs)
                lane_center_offset = camera_utils.compute_lane_offset(int(avg_cx), self.width, self._safe_float(self.cfg.get("lane_width_m", 3.5)))

            objects_in_path = len(in_path_objs)
            closest = min((o["distance_m"] for o in objects), default=999.0)
            obstacle_flag = objects_in_path > 0
            pedestrian_near = any(o["class"] == "person" and o["distance_m"] < 10.0 for o in objects)

            self._frame_count += 1
            # never return raw frames
            agg = {"closest_obj_dist_m": closest, "objects_in_path": objects_in_path}
            flagged = self.should_flag_frame(objects, agg)

            # lane occupancy: left/right based on center split
            tol = self._safe_float(self.cfg.get("path_center_tolerance", 0.3))
            left_bound = 0.5 - tol * 0.5
            right_bound = 0.5 + tol * 0.5
            left_blocking = any(((o["bbox_pixels"][0] + o["bbox_pixels"][2]) / 2.0) / float(self.width) < left_bound for o in objects)
            right_blocking = any(((o["bbox_pixels"][0] + o["bbox_pixels"][2]) / 2.0) / float(self.width) > right_bound for o in objects)

            metadata = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
                "objects": objects,
                "lane_center_offset_m": lane_center_offset,
                "is_left_lane_clear": not left_blocking,
                "is_right_lane_clear": not right_blocking,
                "obstacle_in_path_flag": obstacle_flag,
                "pedestrian_near_crossing_flag": pedestrian_near,
                "visibility_estimate_m": 1000.0,
                "traffic_light_state": "unknown",
                "closest_obj_dist_m": closest,
                "objects_in_path": int(objects_in_path),
                "frame_flagged_for_retention": bool(flagged),
            }

            if flagged:
                try:
                    # When a frame is flagged during production inference, pass
                    # the captured `frame` to the hook so implementations can
                    # persist raw images for flagged events. Honor the
                    # configuration guard to avoid unintentional frame writes.
                    if bool(self.cfg.get("enable_frame_retention", True)):
                        # enforce per-trip cap before passing raw frames
                        cap = int(self.cfg.get("max_retained_frames_per_trip", 100))
                        if int(self._retained_frames_count) < int(cap):
                            try:
                                self.on_frame_flagged(metadata.copy(), frame)
                                self._retained_frames_count += 1
                            except Exception:
                                # if hook fails after increment, don't lose the counter
                                logger.exception("on_frame_flagged hook raised an exception while saving frame")
                        else:
                            # cap reached: still call metadata-only hook so higher layers
                            # are aware of the flagged event but raw frames are not passed
                            try:
                                self.on_frame_flagged(metadata.copy())
                            except Exception:
                                logger.exception("on_frame_flagged metadata-only hook raised an exception")
                    else:
                        # explicit metadata-only call if retention disabled
                        self.on_frame_flagged(metadata.copy())
                except Exception:
                    logger.exception("on_frame_flagged hook raised an exception")

            return metadata
        except Exception:
            # any issues while inferring -> fallback to simulation
            logger.exception("Error during _infer_frame, falling back to simulation")
            self._fallback_to_simulation = True
            return self._simulate_frame(ts)

    def should_flag_frame(self, objects: List[Dict[str, Any]], aggregate_fields: Optional[Dict[str, Any]] = None) -> bool:
        """Decide whether a frame should be flagged for retention.

        This does not store frames; it only returns True/False. Future integrations can
        attach a callback to CameraHandler to receive flagged frames for storage.
        """
        if not bool(self.cfg.get("enable_frame_retention", True)):
            return False
        triggers = self.cfg.get("frame_retention_triggers", ["pedestrian_near_crossing", "obstacle_in_path", "closest_distance_below_5m"]) or []
        # derive fields
        closest = float(aggregate_fields.get("closest_obj_dist_m", 999.0)) if aggregate_fields else 999.0
        objects_in_path = int(aggregate_fields.get("objects_in_path", 0)) if aggregate_fields else 0
        # check triggers
        if "obstacle_in_path" in triggers and objects_in_path > 0:
            return True
        # high risk prediction trigger (if upstream provides risk_score in aggregate_fields)
        if "high_risk_prediction" in triggers:
            try:
                risk = float(aggregate_fields.get("risk_score", 0.0)) if aggregate_fields else 0.0
                threshold = float(self.cfg.get("high_risk_threshold", 0.6))
                if risk > threshold:
                    return True
            except Exception:
                pass
        if "pedestrian_near_crossing" in triggers:
            for o in objects:
                if o.get("class") == "person" and float(o.get("distance_m", 999.0)) <= float(self.cfg.get("critical_distance_thresholds_m", [5, 10, 20])[0]):
                    return True
        if "closest_distance_below_5m" in triggers and closest < 5.0:
            return True
        # sudden_braking requires vehicle controls integration; skip here
        return False

    def on_frame_flagged(self, metadata: Dict[str, Any], frame: Optional[Any] = None) -> None:
        """Hook called when a frame is flagged for retention.

        Default implementation is a no-op. Implementations may override this
        method to receive the metadata and optionally the raw `frame` (e.g., a
        numpy array) for storage or encoding. The system's default output is
        metadata-only; the frame is provided solely via this hook.
        """
        # no-op by default. Subclasses or runtime code may override this method
        # to receive metadata and the optional raw frame from _infer_frame.
        return
