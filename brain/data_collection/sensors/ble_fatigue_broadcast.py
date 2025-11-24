"""BLEFatigueBroadcast: write compact fatigue broadcast packets to outbox JSON.

This mirrors the simple 'ble_stub' pattern used elsewhere in the repo and is
intended for V2V-style outbox files. It is intentionally file-based and
thread-safe so it works in CI/dev without actual BLE hardware.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict

from utils.ble_utils import hash_node_id

logger = logging.getLogger(__name__)


class BLEFatigueBroadcast:
    def __init__(self, config: Any, mesh_handler: Any = None):
        # support dict-style config: {"sensors": {"driver_fatigue": {...}}}
        if hasattr(config, "get"):
            sect = config.get("sensors", {}) or {}
            sect = sect.get("driver_fatigue", {}) or {}
        else:
            sect = config
        bc = sect.get("ble_broadcast", {}) if isinstance(sect, dict) else {}
        self.enable = bool(bc.get("enable", False))
        self.outbox_path = str(bc.get("outbox_path", "data/ble_fatigue_outbox.json"))
        self.broadcast_interval_s = float(bc.get("broadcast_interval_s", 5))
        self.broadcast_states = set(bc.get("broadcast_states", ["DROWSY", "MICROSLEEP"]))
        # privacy salt
        self._salt = str(config.get("privacy", {}).get("node_id_salt", "")) if hasattr(config, "get") else ""

        self._lock = threading.Lock()
        # track last broadcast timestamp per state (epoch seconds)
        self._last_broadcast_ts: dict[str, float] = {}
        # tx power for future real BLE backend (dBm-ish integer)
        try:
            self.tx_power = int(bc.get("tx_power", 0))
        except Exception:
            self.tx_power = 0
        try:
            os.makedirs(os.path.dirname(self.outbox_path) or ".", exist_ok=True)
        except Exception:
            # ignore directory errors; file writes will fail later and be logged
            pass
        # backend selection: 'auto'|'bleak'|'pybluez'|'stub'
        self.backend = str(bc.get("ble_backend", "auto")).lower()
        # service uuid and manufacturer id for advertising
        self.service_uuid = str(bc.get("service_uuid", "0000FATT-0000-1000-8000-00805F9B34FB"))
        try:
            self.manufacturer_id = int(bc.get("manufacturer_id", 65535))
        except Exception:
            self.manufacturer_id = 65535

        # validate tx_power bounds (-30..20 typical)
        try:
            if self.tx_power < -30:
                self.tx_power = -30
            if self.tx_power > 20:
                self.tx_power = 20
        except Exception:
            self.tx_power = 0

        # optional mesh handler delegate (simulated mesh or platform-specific)
        self._mesh_handler = mesh_handler

        # determine advertiser implementation if available (pluggable)
        self._adv_impl = None
        if self.backend in ("auto", "bleak"):
            try:
                import bleak  # type: ignore

                # prefer a generic advertise entry point if provided by bleak
                if hasattr(bleak, "start_advertising"):
                    self._adv_impl = getattr(bleak, "start_advertising")
                else:
                    # leave _adv_impl as None; tests may patch it
                    self._adv_impl = None
            except Exception:
                self._adv_impl = None
        if self.backend == "pybluez" and self._adv_impl is None:
            try:
                import bluetooth  # type: ignore

                # pybluez advertising APIs are platform-specific; provide a small shim
                def _pybluez_adv(payload: bytes, **kw):
                    # pybluez does not provide BLE advertising in a portable way here;
                    # leave as no-op placeholder for environments with pybluez support.
                    return None

                self._adv_impl = _pybluez_adv
            except Exception:
                self._adv_impl = None

    def should_broadcast(self, state: str) -> bool:
        return state in self.broadcast_states

    def maybe_broadcast(self, state: str, score: float, reason: str, vehicle_id: str) -> None:
        """Call this frequently; will only emit when interval elapsed for this state.

        Keeps file-outbox behavior for testing, but avoids emitting on every sample.
        """
        if not self.enable:
            return
        if not self.should_broadcast(state):
            return

        now = datetime.now(timezone.utc).timestamp()
        last = self._last_broadcast_ts.get(state, 0.0)
        if now - last < float(self.broadcast_interval_s):
            return
        # perform broadcast and update timestamp
        self.broadcast(state, score, reason, vehicle_id)
        self._last_broadcast_ts[state] = now

    def broadcast(self, state: str, score: float, reason: str, vehicle_id: str) -> None:
        if not self.enable:
            return
        if not self.should_broadcast(state):
            return

        try:
            packet: Dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "vehicle_id": hash_node_id(vehicle_id or "", salt=self._salt),
                "status": "FatiguedDriver",
                "state": state,
                "score": float(score),
                "reason": reason,
            }
            line = json.dumps(packet, default=str)
            with self._lock:
                # write to JSON outbox for debugging and CI
                with open(self.outbox_path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            # try real BLE advertising if backend available (best-effort)
            try:
                self._real_ble_broadcast(packet)
            except Exception:
                # do not fail broadcasting path if BLE backend not available
                logger.debug("Real BLE broadcast not available or failed; outbox written")
            # update last broadcast timestamp for this state
            try:
                self._last_broadcast_ts[state] = datetime.now(timezone.utc).timestamp()
            except Exception:
                pass
            logger.info("Wrote BLE fatigue packet to outbox: %s", packet)
        except Exception:
            logger.exception("Failed to write BLE fatigue outbox packet")

    def close(self) -> None:
        # file-based: nothing to close but keep API stable
        return None

    def _real_ble_broadcast(self, packet: Dict[str, Any]) -> None:
        """Best-effort real BLE advertising hook.

        This attempts to use `bleak` if installed. If not available, it's a no-op.
        Implementations of BLE advertising differ by platform; keep this
        lightweight and non-fatal.
        """
        # compact JSON payload
        try:
            payload = json.dumps(packet, separators=(",", ":")).encode("utf-8")
        except Exception:
            payload = str(packet).encode("utf-8")

        # delegate to mesh_handler if present (simulated or platform-specific)
        try:
            if getattr(self, "_mesh_handler", None) is not None and hasattr(self._mesh_handler, "broadcast_fatigue"):
                def _call_mesh():
                    try:
                        # mesh handler expected to accept bytes payload
                        self._mesh_handler.broadcast_fatigue(payload)
                    except Exception:
                        logger.exception("mesh_handler.broadcast_fatigue failed")

                threading.Thread(target=_call_mesh, daemon=True).start()
                return
        except Exception:
            logger.exception("Error while delegating to mesh_handler")

        # call pluggable advertiser implementation if present
        if getattr(self, "_adv_impl", None) is not None:
            try:
                def _call_impl():
                    try:
                        # many advertise APIs accept bytes or str; pass common args
                        self._adv_impl(payload, tx_power=self.tx_power, service_uuid=self.service_uuid, manufacturer_id=self.manufacturer_id)
                    except Exception:
                        logger.exception("BLE advertiser implementation failed")

                t = threading.Thread(target=_call_impl, daemon=True)
                t.start()
                return
            except Exception:
                logger.exception("Failed to invoke BLE advertiser implementation")

        # attempt a generic bleak advertising flow if available or fall back to
        # a Linux HCI subprocess-based advertising sequence. Both paths are
        # best-effort and non-fatal; file outbox remains the reliable fallback.
        try:
            import platform
            import subprocess

            # Linux HCI path (requires BlueZ and privileges). Build an LE
            # advertising packet (flags + manufacturer data) and push it to
            # the controller using HCI commands. This is intentionally
            # best-effort: failures will be logged and ignored.
            if platform.system().lower() == "linux":
                try:
                    # set TX power (best-effort)
                    try:
                        subprocess.run(["hciconfig", "hci0", "txpower", str(self.tx_power)], check=False, capture_output=True)
                    except Exception:
                        logger.debug("hciconfig txpower command not available or failed")

                    # Build advertising data: Flags + Manufacturer data with
                    # company id and minified payload.
                    payload_bytes = payload
                    max_adv = 31
                    # Reserve room: flags (3 bytes), mfr header (1 type + 2 company)
                    max_payload = max_adv - (3 + 1 + 2)
                    if len(payload_bytes) > max_payload:
                        payload_bytes = payload_bytes[:max_payload]

                    comp_id = int(self.manufacturer_id) & 0xFFFF
                    comp_lo = comp_id & 0xFF
                    comp_hi = (comp_id >> 8) & 0xFF

                    ad = []
                    # Flags AD structure: length=2, type=0x01, flags=0x06
                    ad += [2, 0x01, 0x06]
                    # Manufacturer Specific Data: len, type 0xFF, company id (LE), payload
                    mfr_len = 1 + 2 + len(payload_bytes)
                    ad += [mfr_len, 0xFF, comp_lo, comp_hi]
                    ad += list(payload_bytes)

                    # pad to 31 bytes
                    if len(ad) < 31:
                        ad += [0] * (31 - len(ad))

                    # hcitool cmd 0x08 0x0008 <len=31> <bytes...>
                    cmd = ["hcitool", "-i", "hci0", "cmd", "0x08", "0x0008", "%02x" % 31]
                    cmd += ["%02x" % (b) for b in ad]
                    try:
                        subprocess.run(cmd, check=False, capture_output=True)
                        # enable advertising (best-effort)
                        subprocess.run(["hciconfig", "hci0", "leadv", "3"], check=False, capture_output=True)
                        # advertise briefly then disable
                        import time as _t

                        _t.sleep(min(2.5, float(self.broadcast_interval_s) / 2.0))
                        subprocess.run(["hciconfig", "hci0", "noleadv"], check=False, capture_output=True)
                        return
                    except Exception:
                        logger.exception("HCI advertising commands failed")
                except Exception:
                    logger.exception("Linux HCI advertising path failed")

            # As a secondary path, try Bleak's advertising API if present
            import bleak  # type: ignore
            if hasattr(bleak, "start_advertising"):
                import asyncio

                async def _do_adv():
                    try:
                        await getattr(bleak, "start_advertising")(payload, service_uuids=[self.service_uuid])
                        await asyncio.sleep(min(2.5, float(self.broadcast_interval_s) / 2.0))
                        if hasattr(bleak, "stop_advertising"):
                            await getattr(bleak, "stop_advertising")()
                    except Exception:
                        logger.exception("bleak advertise sequence failed")

                def _run_async():
                    try:
                        asyncio.run(_do_adv())
                    except Exception:
                        logger.exception("asyncio run for bleak advertise failed")

                threading.Thread(target=_run_async, daemon=True).start()
                return
        except Exception:
            logger.exception("Exception in generic advertiser fallback")

        # no advertiser available; leave file outbox as fallback
        logger.debug("No BLE advertiser available; skipped real BLE advertising")
