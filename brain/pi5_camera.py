import time
from picamera2 import Picamera2, Picamera2Config
import cv2
import numpy as np


class Pi5Camera:
    def __init__(self):
        # Configure the camera for 640x480 @ 30fps
        self.picam2 = Picamera2()
        config = self.picam2.create_configuration(
            main={"size": (640, 480), "format": "BGR888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        print("✅ PiCamera2 Started Successfully")

    def read(self):
        # Capture a single frame as an array (OpenCV format)
        try:
            # wait for a frame (timeout 1 sec)
            frame = self.picam2.capture_array()
            return True, frame
        except Exception as e:
            print(f"Camera Error: {e}")
            return False, None

    def release(self):
        self.picam2.stop()
