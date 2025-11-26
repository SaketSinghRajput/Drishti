"""
Example demonstrating MediaPipe hand landmark detection and grip confidence.

Run with:
  python brain/examples/hand_pose_example.py --model-path brain/models/hand_landmark_lite.tflite

Press 'q' to quit.
"""
import argparse
import time
import json
import cv2

from brain.utils.mediapipe_hand_landmarker import MediaPipeHandLandmarker, compute_grip_confidence_from_keypoints


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', default='brain/models/hand_landmark_lite.tflite')
    p.add_argument('--camera-id', type=int, default=0)
    args = p.parse_args()

    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    hand = MediaPipeHandLandmarker(args.model_path)
    if not hand.is_available():
        print('Hand landmarker not available. Check model path and runtime.')
        return

    last_print = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Try simple full-frame detection (heuristic)
        kps = hand.detect_hand_landmarks(frame, (0.0, 0.0, 1.0, 1.0))
        conf = compute_grip_confidence_from_keypoints(kps)

        # Draw
        for (x, y) in kps:
            h, w = frame.shape[:2]
            cv2.circle(frame, (int(x * w), int(y * h)), 3, (0, 255, 0), -1)
        cv2.putText(frame, f'Grip: {conf:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow('HandGrip', frame)
        if time.time() - last_print > 1.0:
            print(json.dumps({'timestamp': time.time(), 'grip_confidence': conf, 'hand_detected': len(kps) > 0, 'keypoints_count': len(kps)}))
            last_print = time.time()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
