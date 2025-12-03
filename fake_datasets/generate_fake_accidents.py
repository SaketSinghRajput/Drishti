"""
Generate synthetic telemetry data for accident prediction experiments.

Usage:
    python generate_fake_accidents.py --n 10000 --out accidents.csv --seed 42

Outputs a CSV with columns described in README.md
"""
import argparse
import datetime
import random
import csv

WEATHERS = ['clear', 'rain', 'fog', 'snow']
LIGHTS = ['day', 'night', 'dawn', 'dusk']
ROAD_TYPES = ['highway', 'urban', 'rural']

def make_row(ts, rng):
    speed = max(0.0, rng.gauss(60, 18))  # km/h
    steering = rng.gauss(0, 6)
    brake = min(1.0, max(0.0, rng.random() * rng.random()))
    accel_x = rng.gauss(0, 1.2)
    accel_y = rng.gauss(0, 0.6)
    accel_z = 9.81 + rng.gauss(0, 0.2)
    lane_offset = rng.gauss(0, 0.35)
    weather = rng.choice(WEATHERS)
    light = rng.choice(LIGHTS)
    drowsy = min(1.0, max(0.0, rng.gauss(0.15, 0.18)))
    distraction = min(1.0, max(0.0, rng.random()))
    nearby = rng.poisson(1) if hasattr(rng, 'poisson') else max(0, int(rng.gauss(1.2, 1)))
    road = rng.choice(ROAD_TYPES)

    # accident probability heuristics
    prob = 0.01
    if speed > 100: prob += 0.06
    if weather in ('rain', 'fog', 'snow'): prob += 0.03
    if abs(steering) > 20: prob += 0.05
    if brake > 0.7: prob += 0.04
    if drowsy > 0.6: prob += 0.06
    if distraction > 0.7: prob += 0.04
    if abs(lane_offset) > 1.0: prob += 0.03
    prob = min(0.95, prob)
    accident = 1 if rng.random() < prob else 0

    return {
        'timestamp': ts.isoformat(),
        'vehicle_speed_kph': round(speed, 2),
        'steering_angle_deg': round(steering, 2),
        'brake_pressure': round(brake, 3),
        'accel_x': round(accel_x, 3),
        'accel_y': round(accel_y, 3),
        'accel_z': round(accel_z, 3),
        'lane_offset_m': round(lane_offset, 3),
        'weather': weather,
        'light_condition': light,
        'driver_drowsiness': round(drowsy, 3),
        'distraction_level': round(distraction, 3),
        'nearby_vehicles': int(nearby),
        'road_type': road,
        'accident': int(accident)
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=1000, help='number of rows to generate')
    p.add_argument('--out', type=str, default='sample_accident_dataset.csv', help='output CSV file')
    p.add_argument('--seed', type=int, default=42, help='random seed')
    args = p.parse_args()

    rng = random.Random(args.seed)

    start = datetime.datetime.utcnow()

    fieldnames = ['timestamp','vehicle_speed_kph','steering_angle_deg','brake_pressure','accel_x','accel_y','accel_z','lane_offset_m','weather','light_condition','driver_drowsiness','distraction_level','nearby_vehicles','road_type','accident']

    with open(args.out, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(args.n):
            ts = start + datetime.timedelta(seconds=i*1)
            row = make_row(ts, rng)
            writer.writerow(row)

    print(f'Wrote {args.n} rows to {args.out}')

if __name__ == '__main__':
    main()
