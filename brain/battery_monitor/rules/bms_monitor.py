"""High-level BMS monitor orchestrating rules, feature extraction, ML, and telemetry.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from config.config_loader import ConfigLoader
from battery_monitor.rules import RuleEngine
from battery_monitor.ml import BatteryMLInference, BMSFeatureExtractor
from utils.telemetry import TelemetryCollector

# actuation + storage
from battery_monitor.alerts.alert_manager import AlertManager
from battery_monitor.alerts.ble_stub import BLEBroadcastStub
from battery_monitor.storage.event_logger import EventLogger

logger = logging.getLogger(__name__)


class BMSMonitor:
    def __init__(self, config: ConfigLoader, telemetry_collector: TelemetryCollector):
        self.config = config
        self.telemetry = telemetry_collector
        self.rule_engine = RuleEngine(config)
        self.extractor = BMSFeatureExtractor(config)
        self.ml = BatteryMLInference(config)
        # initialize attributes so stop() can always reference them
        self.ble_stub = None
        self.event_logger = None
        self.alert_manager = None
        # instantiate alerting/storage subsystems (best-effort)
        try:
            self.ble_stub = BLEBroadcastStub(config)
        except Exception:
            logger.exception("Failed to initialize BLEBroadcastStub")
            self.ble_stub = None
        try:
            self.event_logger = EventLogger(config)
        except Exception:
            logger.exception("Failed to initialize EventLogger")
            self.event_logger = None
        try:
            self.alert_manager = AlertManager(config, telemetry=self.telemetry, ble_stub=self.ble_stub, event_logger=self.event_logger)
        except Exception:
            logger.exception("Failed to initialize AlertManager")
            self.alert_manager = None

    def process_telemetry(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        alerts: List[Dict[str, Any]] = []
        if not samples:
            return alerts
        latest = samples[-1]
        # run rule engine
        rules = self.rule_engine.evaluate(latest)
        for r in rules:
            alert = {
                "source": "rule",
                "severity": r.get("severity"),
                "reason": r.get("reason"),
                "rule": r.get("rule_name"),
                "value": r.get("value"),
                "anomaly_score": None,
            }
            alerts.append(alert)
            try:
                # Note: ExplanationLogger is not used for BMS alerts. Alerts are
                # recorded via TelemetryCollector and persisted to EventLogger.
                # This keeps explanation logs focused on model prediction explanations.
                self.telemetry.log_battery_alert(alert, telemetry_snapshot=samples)
            except Exception:
                logger.exception("Failed to log battery alert")
            # trigger actuation and storage
            try:
                if self.alert_manager is not None:
                    self.alert_manager.trigger_alert(alert, telemetry_snapshot=samples)
            except Exception:
                logger.exception("AlertManager.trigger_alert failed for rule alert")

        # ML inference
        try:
            if self.config.get("battery_monitor.ml.enable", True):
                feats = self.extractor.extract_features(samples)
                res = self.ml.predict_anomaly(feats)
                rul = self.ml.predict_rul(feats)
                ml_alert = {
                    "source": "ml",
                    "severity": "MEDIUM" if res.get("is_anomaly") else "LOW",
                    "reason": "ML anomaly" if res.get("is_anomaly") else "",
                    "anomaly_score": res.get("anomaly_score"),
                    "rul_hours": rul,
                    "value": res.get("anomaly_score"),
                }
                if res.get("is_anomaly"):
                    alerts.append(ml_alert)
                    try:
                        # See note above regarding ExplanationLogger integration.
                        self.telemetry.log_battery_alert(ml_alert, telemetry_snapshot=samples)
                    except Exception:
                        logger.exception("Failed to log ML alert")
                    try:
                        if self.alert_manager is not None:
                            # include features and scores when available
                            self.alert_manager.trigger_alert(ml_alert, telemetry_snapshot=samples, features=feats, scores=res)
                    except Exception:
                        logger.exception("AlertManager.trigger_alert failed for ML alert")
        except Exception:
            logger.exception("ML inference failed")

        return alerts

    def start(self):
        # reserved for future threaded monitors
        pass

    def stop(self):
        # Close/cleanup subsystems if they were instantiated.
        try:
            if getattr(self, "alert_manager", None) is not None:
                try:
                    self.alert_manager.close()
                except Exception:
                    logger.exception("alert_manager.close() failed")
        except Exception:
            logger.exception("Error while closing alert_manager")
        try:
            if getattr(self, "ble_stub", None) is not None:
                try:
                    self.ble_stub.close()
                except Exception:
                    logger.exception("ble_stub.close() failed")
        except Exception:
            logger.exception("Error while closing ble_stub")
        try:
            if getattr(self, "event_logger", None) is not None:
                try:
                    self.event_logger.close()
                except Exception:
                    logger.exception("event_logger.close() failed")
        except Exception:
            logger.exception("Error while closing event_logger")
