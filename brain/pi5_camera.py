import time
from picamera2 import Picamera2
import cv2
import numpy as np

class Pi5Camera:
    def __init__(self):
        print("📷 Initializing PiCamera2...")
        self.picam2 = Picamera2()
        
        # CORRECTED METHOD: Use create_video_configuration
        config = self.picam2.create_video_configuration(
            main={"size": (640, 480), "format": "BGR888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        print("✅ PiCamera2 Started Successfully")

    def read(self):
        try:
            # Capture a single frame as an array (OpenCV format)
            frame = self.picam2.capture_array()
            return True, frame
        except Exception as e:
            print(f"Camera Error: {e}")
            return False, None

    def release(self):
        self.picam2.stop()
        self.picam2.close()
