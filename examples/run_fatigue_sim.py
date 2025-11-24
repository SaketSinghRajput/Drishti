#!/usr/bin/env python3
"""
Run a simulated fatigue demo without hardware.
- Prints EAR, blink count, and fatigue percent to the CLI every `--interval` seconds.
- Optionally shows a simple OpenCV UI if an X server is available (`--show-ui`).
- Intended for WSL simulation when camera/hardware aren't attached.
"""
import time
import argparse
import math
import random
import sys
from datetime import datetime

try:
    import cv2
    import numpy as np
    CV2_OK = True
except Exception:
    CV2_OK = False


def ear_to_fatigue(ear):
    if ear <= 0:
        return 0.0
    low = 0.08
    high = 0.32
    val = (high - ear) / (high - low)
    val = max(0.0, min(1.0, val))
    return val * 100.0


class SimulatedFatigue:
    def __init__(self, interval=1.0):
        self.interval = float(interval)
        self.ear = 0.30  # start awake
        self.blink_count = 0
        self.last_blink = 0.0
        self.t = 0.0
        self.start_time = time.time()

    def step(self):
        """Advance simulation by self.interval seconds and return a sample dict"""
        self.t = time.time() - self.start_time
        # slowly drift EAR down to simulate fatigue over minutes
        fatigue_drift = 0.0008 * self.interval  # adjust rate
        self.ear = max(0.08, self.ear - fatigue_drift + random.uniform(-0.005, 0.003))
        # occasional blink event: quick drop then recover inside same interval
        # simulate blink probability that increases with fatigue
        fatigue_pct = ear_to_fatigue(self.ear)
        blink_prob = 0.02 + (fatigue_pct / 200.0)  # base + scaled
        blinked = False
        if random.random() < blink_prob:
            self.blink_count += 1
            blinked = True
        sample = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'ear': round(self.ear, 3),
            'blink_count': self.blink_count,
            'fatigue_pct': round(fatigue_pct, 1),
            'blinked': blinked,
            'sim_time_s': round(self.t, 1),
        }
        return sample


def draw_ui(sample, w=960, h=540):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # background
    img[:] = (30, 30, 40)
    # title
    cv2.putText(img, 'Simulated Fatigue Demo', (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230,230,230), 2)
    # metrics panel
    cv2.rectangle(img, (20, 60), (420, 160), (20,20,20), -1)
    cv2.putText(img, f"EAR: {sample['ear']:.3f}", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220,220,220), 2)
    cv2.putText(img, f"Blinks: {sample['blink_count']}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 1)
    cv2.putText(img, f"Time: {sample['sim_time_s']}s", (220, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    # fatigue bar
    bar_x, bar_y, bar_w, bar_h = 20, h-100, w-40, 60
    cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (60,60,60), -1)
    pct = sample['fatigue_pct']
    fill_w = int((pct/100.0) * bar_w)
    color = (0,220,0) if pct < 50 else ((0,200,200) if pct < 75 else (0,0,255))
    cv2.rectangle(img, (bar_x+2, bar_y+2), (bar_x+fill_w-2, bar_y+bar_h-2), color, -1)
    cv2.putText(img, f"FATIGUE: {pct:.0f}%", (bar_x+6, bar_y+bar_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # pulse indicator when fatigue high
    intensity = max(0.0, (pct - 40.0) / 60.0)
    if intensity > 0.01:
        cx, cy = w-120, 80
        r = 20 + int(15 * intensity)
        overlay = img.copy()
        cv2.circle(overlay, (cx, cy), r, (0,0,255), thickness=6)
        cv2.addWeighted(overlay, min(0.6, 0.2+0.8*intensity), img, 1.0-min(0.6,0.2+0.8*intensity), 0, img)
        cv2.putText(img, 'ALERT', (cx-30, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    return img


def can_show_ui():
    if not CV2_OK:
        return False
    # quick check whether imshow will work: try to create a namedWindow and destroy it
    try:
        cv2.namedWindow('simtest', cv2.WINDOW_NORMAL)
        cv2.destroyWindow('simtest')
        return True
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--interval', type=float, default=1.0, help='seconds between updates')
    p.add_argument('--show-ui', action='store_true', help='display a live OpenCV UI if available')
    p.add_argument('--duration', type=float, default=0.0, help='run duration seconds (0 = forever)')
    args = p.parse_args()

    sim = SimulatedFatigue(interval=args.interval)
    show_ui = args.show_ui and can_show_ui()
    if args.show_ui and not show_ui:
        print('UI requested but not available (no cv2 GUI). Falling back to CLI-only mode.')

    print('Starting simulated fatigue run; interval=%.2fs show_ui=%s' % (args.interval, show_ui))
    start = time.time()
    try:
        while True:
            sample = sim.step()
            # CLI output
            print(f"[{sample['timestamp']}] EAR={sample['ear']:.3f} BLINKS={sample['blink_count']} FATIGUE={sample['fatigue_pct']:.1f}% blinked={sample['blinked']}")
            sys.stdout.flush()
            # UI update
            if show_ui:
                frame = draw_ui(sample)
                try:
                    cv2.imshow('Simulated Fatigue', frame)
                    key = cv2.waitKey(int(args.interval*1000)) & 0xFF
                    if key == ord('q'):
                        break
                except Exception as e:
                    print('UI error:', e)
                    show_ui = False
            else:
                time.sleep(args.interval)

            if args.duration > 0 and (time.time() - start) >= args.duration:
                print('Finished: duration reached')
                break
    except KeyboardInterrupt:
        print('\nInterrupted by user')
    finally:
        if CV2_OK:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == '__main__':
    main()
