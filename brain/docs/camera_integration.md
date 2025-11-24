# Camera Integration Guide

This document describes the camera subsystem: handler, adapter, model loader, and utilities.

## Overview
The camera subsystem performs object detection on frames and emits metadata-only
records (no raw frames stored by default). It supports simulation mode for development
and several model backends for deployment (TFLite, ONNX, YOLOv5).

## Components
- `CameraHandler`: captures frames (or simulates them) and runs detection. Returns metadata-only dicts.
- `CameraAdapter`: aggregates frame-level detections into window-level features.
- `ModelLoader`: helpers to instantiate detection backends with lazy imports.
- `CameraUtils`: helpers for distance estimation, lane offset, NMS, and filtering.

## Simulation vs Production

Simulation is enabled by default in `config/default_config.yaml` (enable_simulation: true).
To run in production, provide a `model_path`, set `enable_simulation: false`, and install
camera runtime dependencies (OpenCV, TFLite/ONNX runtime or ultralytics).

Note on lane detection: `enable_lane_detection` is currently a placeholder. The present
implementation does NOT perform visual lane line detection; instead `CameraHandler`
estimates lane center offset from detected object positions. If visual lane lines are
required, set `enable_lane_detection: true` and implement a lane-finding model or a
lightweight OpenCV Canny+Hough pipeline (this is not provided by default).

## Privacy

By default the handler only stores metadata (object class, bbox, estimated distance,
relative speed, flags). Frame retention is configurable and limited by `max_retained_frames_per_trip`.

Retention enforcement: `CameraHandler` enforces `max_retained_frames_per_trip` before
passing raw frames to the `on_frame_flagged` hook — once the per-trip cap is reached,
the handler will still call the hook with metadata only but will not pass raw frames.

## Configuration
See `sensors.camera` in `config/default_config.yaml` for all options (fps, model_type, focal length, etc.).

## Example
Run `examples/camera_example.py` to collect 10 seconds of simulated data and print camera features.
