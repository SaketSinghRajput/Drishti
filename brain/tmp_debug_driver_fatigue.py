import sys
import types

# inject minimal fake cv2 to avoid import errors during local debug
sys.modules['cv2'] = types.ModuleType('cv2')
from data_collection.sensors.driver_fatigue_handler import DriverFatigueHandler
from tests.test_data_collection.test_driver_fatigue_state_classification import _FakeCamera, _FakeDetector
import time
cfg={}
h=DriverFatigueHandler(cfg, use_cv=False)
# set thresholds
h._threshold_fatigue_low=0.3
h._threshold_drowsy=0.6
h._threshold_micro=0.8
h._hysteresis_dwell_s=0.0
h._hysteresis_margin=0.0
# inject
h._camera=_FakeCamera()
h._detector=_FakeDetector(ear=0.01, mar=0.1, confidence=0.9)
h._profiler=None
s=h.read()
print('sample:', s)
print('events:', list(h._events))
print('current_state:', h._current_state)
