import numpy as np
import cv2

from data_collection.sensors.driver_drowsiness.camera_interface import CameraInterface


class FakeCap:
    def __init__(self):
        self._open = True
    def isOpened(self):
        return True
    def read(self):
        return True, np.zeros((480,640,3), dtype=np.uint8)
    def set(self, a, b):
        pass
    def release(self):
        pass


def test_camera_enters_low_power_on_no_face(monkeypatch):
    # patch VideoCapture to our FakeCap so no real hardware is needed
    monkeypatch.setattr('cv2.VideoCapture', lambda *args, **kwargs: FakeCap())
    cam = CameraInterface(device=0, width=640, height=480, fps=30)
    cam.enable_low_power_mode = True
    cam.low_power_fps = 2
    # initially not low power
    status = cam.get_status()
    assert status['low_power_active'] is False
    # simulate detector reporting no face for threshold times
    for _ in range(cam.no_face_frames_threshold):
        cam.mark_no_face_detected()
    status = cam.get_status()
    assert status['low_power_active'] is True
    assert status['current_fps'] == cam.low_power_fps
    # simulate face present
    cam.clear_no_face_counter()
    status = cam.get_status()
    assert status['low_power_active'] is False

