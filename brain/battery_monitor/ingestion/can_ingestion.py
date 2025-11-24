"""CAN ingestion for BMS telemetry using python-can.

Provides a simple threaded listener that buffers CAN messages for reading
by higher-level parsers or handlers. This is a best-effort scaffold and
handles missing hardware gracefully by setting `_available` to False.
"""

import threading
import queue
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class CANIngestion:
    """Simple CAN ingestion wrapper.

    Methods:
        start(): attempt to open CAN bus and start listener thread
        read_message(timeout=1.0): return (can_id, data_bytes) or None
        stop(): stop listener and close bus
        get_status(): return dict with connection and counts
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.interface = self.config.get('battery_monitor.ingestion.can_interface', 'can0')
        self.bitrate = int(self.config.get('battery_monitor.ingestion.can_bitrate', 500000))
        self._queue = queue.Queue()
        self._thread = None
        self._stop_event = threading.Event()
        self._available = False
        self._msg_count = 0
        self._error_count = 0
        self._bus = None

    def start(self):
        try:
            import can

            # attempt to open bus (SocketCAN style). Some backends accept a
            # bitrate parameter; try to pass it but fall back if not supported.
            try:
                self._bus = can.interface.Bus(channel=self.interface, bustype='socketcan', bitrate=self.bitrate)
            except TypeError:
                # Bus implementation may not accept bitrate; retry without it
                self._bus = can.interface.Bus(channel=self.interface, bustype='socketcan')
            self._available = True if self._bus is not None else False

            def _reader():
                while not self._stop_event.is_set():
                    try:
                        msg = self._bus.recv(timeout=1.0)
                        if msg is None:
                            continue
                        self._msg_count += 1
                        # put (arbitration_id, data)
                        self._queue.put((msg.arbitration_id, bytes(msg.data)))
                    except Exception as e:
                        self._error_count += 1
                        logger.exception('CAN read error: %s', e)

            self._thread = threading.Thread(target=_reader, daemon=True)
            self._thread.start()
            # refresh availability based on thread state
            if self._thread and self._thread.is_alive():
                self._available = True
        except Exception as e:
            # hardware may be unavailable; mark as not available
            self._available = False
            self._error_count += 1
            logger.warning('Failed to initialize CAN bus (%s): %s', self.interface, e)

    def read_message(self, timeout: float = 1.0) -> Optional[Tuple[int, bytes]]:
        try:
            item = self._queue.get(timeout=timeout)
            return item
        except Exception:
            return None

    def stop(self):
        self._stop_event.set()
        try:
            if self._thread:
                self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self._bus:
                self._bus.shutdown()
        except Exception:
            pass
        self._available = False

    def get_status(self) -> dict:
        return {
            'available': self._available,
            'message_count': self._msg_count,
            'error_count': self._error_count,
        }
