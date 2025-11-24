# EV Safety Guardian - Dashboard

This document describes the main dashboard for real-time monitoring, incident review, annotation, and analytics.

See the `dashboard/` package for implementation details and `scripts/run_dashboard.py` for a runnable example.

## Quick start

```bash
python scripts/run_dashboard.py
# Open http://localhost:5004
```

## Features

- Live risk scores
- Incident review with video playback and synchronized sensor timelines
- Annotation tools (TP/FP/TN/FN)
- Analytics: false alarm rate, latency, feature importance
