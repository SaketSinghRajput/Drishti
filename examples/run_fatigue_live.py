#!/usr/bin/env python3
"""
Live driver fatigue demo:
- Uses MediaPipe FaceMesh (fast) when available to compute EAR (eye aspect ratio)
- Displays live video, EAR, blink count, fatigue percentage and animated graphics
- Falls back to a static image if camera not available (uploaded image path used)
"""

import time
import collections
import argparse
import math
import sys
import os

import cv2
import numpy as np

# Try to import mediapipe; fall back to haar cascades if not installed
USE_MEDIAPIPE = False
try:
    import mediapipe as mp
    USE_MEDIAPIPE = True
except Exception:
    USE_MEDIAPIPE = False

# --- YOLO (ONNX) helpers using OpenCV DNN ---
def yolo_load_model(onnx_path, input_size=320):
    net = cv2.dnn.readNet(onnx_path)
    return {"net": net, "input_size": int(input_size)}

def yolo_preprocess(frame, input_size):
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
    return blob

def yolo_postprocess(outputs, conf_thresh, nms_thresh, input_size, frame_w, frame_h):
    # Generic YOLO postprocess supporting common ONNX outputs like (1, N, 85)
    dets = []
    out = outputs
    if isinstance(outputs, list) and len(outputs) > 0:
        out = outputs[0]
    # ensure 2D
    arr = out.reshape(-1, out.shape[-1])
    for row in arr:
        conf = float(row[4])
        if conf < conf_thresh:
            continue
        # class scores after the 5th element
        scores = row[5:]
        if scores.size == 0:
            cls_id = 0
            cls_conf = conf
        else:
            cls_id = int(np.argmax(scores))
            cls_conf = float(scores[cls_id]) * conf
            if cls_conf < conf_thresh:
                continue
        cx, cy, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        # xywh are typically relative to input_size
        x = int((cx - w/2.0) * frame_w / input_size)
        y = int((cy - h/2.0) * frame_h / input_size)
        bw = int(w * frame_w / input_size)
        bh = int(h * frame_h / input_size)
        dets.append((x, y, bw, bh, cls_conf, cls_id))
    # apply NMS
    if len(dets) == 0:
        return []
    boxes = np.array([[d[0], d[1], d[0]+d[2], d[1]+d[3]] for d in dets])
    scores = np.array([d[4] for d in dets])
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thresh, nms_thresh)
    out_dets = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh, conf, cid = dets[i]
            out_dets.append({"box": (x, y, bw, bh), "conf": conf, "class_id": int(cid)})
    return out_dets

def yolo_infer(model, frame, conf_thresh=0.4, nms_thresh=0.45):
    net = model["net"]
    input_size = model["input_size"]
    blob = yolo_preprocess(frame, input_size)
    net.setInput(blob)
    outs = net.forward()
    h, w = frame.shape[:2]
    dets = yolo_postprocess(outs, conf_thresh, nms_thresh, input_size, w, h)
    return dets

# Mediapipe face mesh landmark indices for eyes (common mapping)
# left eye: 33 (p1), 160 (p2), 158 (p3), 133 (p4), 153 (p5), 144 (p6)
# right eye: 362 (p1), 385 (p2), 387 (p3), 263 (p4), 373 (p5), 380 (p6)
LM_LEFT = [33, 160, 158, 133, 153, 144]
LM_RIGHT = [362, 385, 387, 263, 373, 380]

def eucl(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def eye_aspect_ratio(pts):
    # pts: [p1,p2,p3,p4,p5,p6] in (x,y)
    # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    A = eucl(pts[1], pts[5])
    B = eucl(pts[2], pts[4])
    C = eucl(pts[0], pts[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def draw_animated_bar(frame, x, y, w, h, pct, label="RISK"):
    # background
    cv2.rectangle(frame, (x,y), (x+w, y+h), (40,40,40), -1)
    # border
    cv2.rectangle(frame, (x,y), (x+w, y+h), (200,200,200), 1)
    # filled width
    fill_w = int((pct/100.0) * w)
    # color gradient green->yellow->red
    if pct < 50:
        color = (0, 220, 0)
    elif pct < 75:
        color = (0, 200, 200)
    else:
        color = (0, 0, 255)
    cv2.rectangle(frame, (x+2, y+2), (x+fill_w-2, y+h-2), color, -1)
    cv2.putText(frame, f"{label}: {pct:.0f}%", (x+6, y+h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

def draw_pulsing_ring(frame, cx, cy, base_radius, intensity):
    # intensity 0..1 controls alpha and radius oscillation
    t = time.time()
    r = base_radius + int(6.0 * intensity * (0.5 + 0.5*math.sin(2.0 * t)))
    overlay = frame.copy()
    alpha = min(0.6, 0.2 + 0.8 * intensity)
    cv2.circle(overlay, (cx, cy), r, (0,0,255), thickness=6)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

def main(args):
    # fallback image (your uploaded path). The assistant provided this path when you uploaded earlier:
    fallback_image = args.image_fallback or "/mnt/data/e466e5e0-c7e8-4a44-8060-06cfa2eeeec6.png"

    # initialize video capture
    cap = None
    if args.camera is not None:
        try:
            cap = cv2.VideoCapture(int(args.camera))
            # try to grab a frame to confirm camera works
            ok, _ = cap.read()
            if not ok:
                cap.release()
                cap = None
        except Exception:
            cap = None

    if cap is None:
        print("Camera not available or cannot be opened. Falling back to static image:", fallback_image)
        img = None
        # prefer cv2.imread, but some cv2 builds in minimal environments may not expose imread
        try:
            img = cv2.imread(fallback_image)
        except AttributeError:
            img = None
        if img is None:
            # try PIL as a fallback loader
            try:
                from PIL import Image
                pil_img = Image.open(fallback_image).convert("RGB")
                arr = np.array(pil_img)
                # convert RGB->BGR for OpenCV consistency
                img = arr[:, :, ::-1].copy()
            except Exception:
                img = None
        if img is None:
            print("Fallback image not found or could not be loaded. Exiting.")
            return
        # convert image to a small video stream loop
        loop_frame = img.copy()
        show_static = True
    else:
        show_static = False

    # Mediapipe setup
    # optional YOLO model (ONNX) setup
    yolo_model = None
    if getattr(args, 'yolo_onnx', None):
        try:
            yolo_model = yolo_load_model(args.yolo_onnx, input_size=args.yolo_input_size)
            print("Loaded YOLO model:", args.yolo_onnx)
        except Exception as e:
            print("Failed to load YOLO model:", e)

    if USE_MEDIAPIPE and yolo_model is None:
        mp_face_mesh = mp.solutions.face_mesh
        fm = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                   refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        print("Using MediaPipe FaceMesh for EAR detection.")
    else:
        print("MediaPipe not available or YOLO provided — using Haar cascade face+eye fallback (less accurate).")
        # load cascades (used when no mediapipe and YOLO missing)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    # sliding windows for EAR & blink logic
    ear_history = collections.deque(maxlen=30)   # last ~1s @30fps
    blink_count = 0
    blink_state = False
    last_blink_time = 0.0

    # fatigue metric: map smoothed EAR to percent: lower EAR -> higher fatigue
    def ear_to_fatigue(ear):
        # typical EAR awake ~0.25-0.30, closed ~0.10-0.15
        if ear <= 0:
            return 0.0
        # invert and clamp
        # map EAR range 0.08 (closed) - 0.32 (open) to 100..0 (fatigue)
        low = 0.08
        high = 0.32
        val = (high - ear) / (high - low)
        val = max(0.0, min(1.0, val))
        return val * 100.0

    win_name = "EV Safety — Fatigue Demo (press q to quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 640)

    fps_last = time.time()
    frames = 0

    while True:
        if show_static:
            frame = loop_frame.copy()
            time.sleep(1.0/10.0)  # slow loop for static image demo
        else:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed; stopping.")
                break

        h, w = frame.shape[:2]
        ear = 0.0
        detected = False

        if USE_MEDIAPIPE:
            # feed to mediapipe (expects RGB)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = fm.process(rgb)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                # convert to pixel coords
                pts_left = []
                pts_right = []
                for idx in LM_LEFT:
                    p = lm[idx]
                    pts_left.append((int(p.x * w), int(p.y * h)))
                for idx in LM_RIGHT:
                    p = lm[idx]
                    pts_right.append((int(p.x * w), int(p.y * h)))
                ear_l = eye_aspect_ratio(pts_left)
                ear_r = eye_aspect_ratio(pts_right)
                ear = (ear_l + ear_r) / 2.0
                detected = True
                # draw eye contours
                for p in pts_left + pts_right:
                    cv2.circle(frame, p, 1, (0,255,0), -1)
        else:
            # If a YOLO model was provided, prefer its detections
            if yolo_model is not None:
                dets = yolo_infer(yolo_model, frame, conf_thresh=args.yolo_conf, nms_thresh=args.yolo_nms)
                # classify detections into faces and eyes using provided class ids
                face_dets = [d for d in dets if d['class_id'] == args.yolo_face_class]
                eye_dets = [d for d in dets if args.yolo_eye_class is not None and args.yolo_eye_class >= 0 and d['class_id'] == args.yolo_eye_class]
                if len(eye_dets) >= 2:
                    # take two highest-confidence eyes
                    eye_dets = sorted(eye_dets, key=lambda x: x['conf'], reverse=True)[:2]
                    pts = []
                    for ed in eye_dets:
                        x_e, y_e, w_e, h_e = ed['box']
                        cx = x_e + w_e//2
                        cy = y_e + h_e//2
                        pts.append((cx, cy))
                    # approximate EAR from box aspect ratio
                    ear_vals = []
                    for ed in eye_dets:
                        _, _, ew, eh = ed['box']
                        # approximate EAR from height/width ratio (scaled)
                        ear_vals.append(max(0.05, min(0.35, 0.25 * (eh / (ew + 1e-6)) * 2.0)))
                    ear = float(sum(ear_vals)/len(ear_vals))
                    detected = True
                    # draw eye boxes
                    for ed in eye_dets:
                        x_e, y_e, w_e, h_e = ed['box']
                        cv2.rectangle(frame, (x_e, y_e), (x_e+w_e, y_e+h_e), (255,200,0), 2)
                elif len(face_dets) > 0:
                    # estimate eye positions from face bbox
                    fd = face_dets[0]
                    x_f, y_f, w_f, h_f = fd['box']
                    # left eye region
                    lx = int(x_f + 0.14 * w_f)
                    ly = int(y_f + 0.26 * h_f)
                    lw = int(0.2 * w_f)
                    lh = int(0.12 * h_f)
                    rx = int(x_f + 0.66 * w_f)
                    ry = ly
                    rw = lw
                    rh = lh
                    pts = [(lx+lw//2, ly+lh//2), (rx+rw//2, ry+rh//2)]
                    ear_vals = [max(0.05, min(0.35, 0.25 * (lh / (lw + 1e-6)) * 2.0)), max(0.05, min(0.35, 0.25 * (rh / (rw + 1e-6)) * 2.0))]
                    ear = float(sum(ear_vals)/len(ear_vals))
                    detected = True
                    cv2.rectangle(frame, (x_f, y_f), (x_f+w_f, y_f+h_f), (200,180,0), 2)
                    cv2.rectangle(frame, (lx, ly), (lx+lw, ly+lh), (255,200,0), 1)
                    cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (255,200,0), 1)
                else:
                    # fallback to Haar cascades if available
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    if len(faces) > 0:
                        # use first face
                        (x,y,fw,fh) = faces[0]
                        roi_gray = gray[y:y+fh, x:x+fw]
                        eyes = eye_cascade.detectMultiScale(roi_gray)
                        if len(eyes) >= 2:
                            # pick two largest eyes
                            eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
                            pts = []
                            for (ex,ey,ew,eh) in eyes:
                                cx = x + ex + ew//2
                                cy = y + ey + eh//2
                                pts.append((cx, cy))
                            ear = 0.22
                            detected = True
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    # use first face
                    (x,y,fw,fh) = faces[0]
                    roi_gray = gray[y:y+fh, x:x+fw]
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    if len(eyes) >= 2:
                        # pick two largest eyes
                        eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
                        pts = []
                        for (ex,ey,ew,eh) in eyes:
                            cx = x + ex + ew//2
                            cy = y + ey + eh//2
                            pts.append((cx, cy))
                        # approximate ear using eye heights/widths (very rough)
                        ear = 0.22
                        detected = True

        # update EAR history
        ear_history.append(ear)
        smoothed_ear = float(sum(ear_history)/len(ear_history)) if len(ear_history) else 0.0

        # blink detection: quick drop under threshold
        BLINK_EAR_THRESH = 0.16
        BLINK_MIN_INTERVAL = 0.2
        now = time.time()
        if smoothed_ear < BLINK_EAR_THRESH and not blink_state:
            # started a blink
            if now - last_blink_time > BLINK_MIN_INTERVAL:
                blink_state = True
                last_blink_time = now
        if smoothed_ear >= BLINK_EAR_THRESH and blink_state:
            blink_state = False
            blink_count += 1

        # fatigue percent from smoothed EAR
        fatigue_pct = ear_to_fatigue(smoothed_ear)

        # draw overlays
        # top-left stats panel
        cv2.rectangle(frame, (10,10), (300,110), (20,20,20), -1)
        cv2.putText(frame, f"EAR: {smoothed_ear:.3f}", (18,36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220,220,220), 2)
        cv2.putText(frame, f"Blink count: {blink_count}", (18,66), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
        fps = 1.0 / (time.time() - fps_last) if (time.time() - fps_last) > 0 else 0.0
        cv2.putText(frame, f"FPS: {fps:.1f}", (18,96), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)

        # animated risk bar bottom
        draw_animated_bar(frame, 20, h-70, w-40, 50, fatigue_pct, label="FATIGUE")

        # pulsing ring near top-right when fatigue high
        intensity = max(0.0, (fatigue_pct - 40.0) / 60.0)  # starts pulsing beyond 40%
        if intensity > 0.01:
            draw_pulsing_ring(frame, w-140, 80, 40, intensity)

        # small circular gauge for fatigue (top-right)
        cx, cy = w-80, 40
        cv2.circle(frame, (cx, cy), 40, (50,50,50), -1)
        # draw gauge arc
        angle = int((fatigue_pct / 100.0) * 270)  # 0..270 degrees
        start = 135
        end = start + angle
        for a in range(start, start+270, 6):
            theta = math.radians(a)
            rx = int(cx + 34 * math.cos(theta))
            ry = int(cy + 34 * math.sin(theta))
            color = (0,255,0) if a <= end else (30,30,30)
            cv2.circle(frame, (rx,ry), 2, color, -1)
        cv2.putText(frame, f"{fatigue_pct:.0f}%", (cx-25, cy+6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 1)

        # title and detection label
        cv2.putText(frame, "EV Safety — Fatigue Monitor", (18, h-110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        status = "TRACKING" if detected else "NO FACE"
        color_status = (0,255,0) if detected else (0,140,255)
        cv2.putText(frame, status, (18, h-90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_status, 2)

        # show frame
        cv2.imshow(win_name, frame)
        frames += 1
        fps_last = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    if cap:
        cap.release()
    cv2.destroyAllWindows()
    if USE_MEDIAPIPE:
        try:
            fm.close()
        except Exception:
            pass

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--camera", default="0", help="camera index (0) or None to force fallback")
    p.add_argument("--image-fallback", default=None, help="path to fallback image if camera unavailable")
    # YOLO ONNX options
    p.add_argument("--yolo-onnx", default=None, help="path to YOLO ONNX model for face/eye detection")
    p.add_argument("--yolo-input-size", type=int, default=320, help="YOLO model input size (square)")
    p.add_argument("--yolo-conf", type=float, default=0.35, help="confidence threshold for YOLO detections")
    p.add_argument("--yolo-nms", type=float, default=0.45, help="NMS threshold for YOLO detections")
    p.add_argument("--yolo-face-class", type=int, default=0, help="class id for face in YOLO model (default 0)")
    p.add_argument("--yolo-eye-class", type=int, default=-1, help="class id for eye in YOLO model (use -1 if model doesn't provide eye class)")
    args = p.parse_args()
    main(args)
