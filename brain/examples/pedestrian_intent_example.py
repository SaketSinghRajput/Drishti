"""
Example demonstrating pedestrian intent scoring using pose landmarks.

Usage:
  python brain/examples/pedestrian_intent_example.py --video sample.mp4

This example expects the project's camera handler to be available; here we
show a simplified local pipeline that reads frames, invokes a provided pose
landmark function, and computes an intent score using the `camera_handler_pose_enrichment` helpers.
"""
import argparse
import cv2
import time
import json

from brain.data_collection.sensors.camera_handler_pose_enrichment import compute_pedestrian_intent, compute_cyclist_lean_angle


def fake_pose_detector(frame):
    # Placeholder: this should call the real MediaPipe pose detector
    # Here we return an empty list (no poses) so the example remains runnable
    return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', default=0)
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    prev_kps = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # should be list of people with pose keypoints, here we simulate one person
        people = []
        # For a real run, replace fake_pose_detector with MediaPipe pose detector wrapper
        # and iterate detected people

        # Example: annotate if any
        for person in people:
            curr_kps = person.get('keypoints')
            score = compute_pedestrian_intent(prev_kps, curr_kps, camera_velocity_window=5)
            lean = compute_cyclist_lean_angle(curr_kps)
            # draw boxes or text
        # we print a placeholder
        print(json.dumps({'timestamp': time.time(), 'pedestrians_walking_towards': 0, 'max_intent_score': 0.0, 'avg_intent': 0.0}))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


if __name__ == '__main__':
    main()
