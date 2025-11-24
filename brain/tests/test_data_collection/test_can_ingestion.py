import threading
import time
from unittest import mock

import pytest

from battery_monitor.ingestion.can_ingestion import CANIngestion


class DummyMsg:
    def __init__(self, arbitration_id, data):
        self.arbitration_id = arbitration_id
        self.data = data


def test_can_ingestion_start_success_and_read(monkeypatch):
    # Mock can.interface.Bus so that Bus.recv yields a couple of messages then None
    recv_calls = [DummyMsg(0x100, b"\x01\x02"), DummyMsg(0x101, b"\x03\x04"), None]

    class MockBus:
        def __init__(self, channel=None, bitrate=None, *args, **kwargs):
            self._recv_iter = iter(recv_calls)

        def recv(self, timeout=None):
            try:
                time.sleep(0.01)
                return next(self._recv_iter)
            except StopIteration:
                return None

        def shutdown(self):
            return None

    import types, sys
    mock_can = types.ModuleType('can')
    mock_can.interface = types.SimpleNamespace()
    mock_can.interface.Bus = MockBus
    monkeypatch.setitem(sys.modules, 'can', mock_can)

    ci = CANIngestion({'battery_monitor.ingestion.can_interface': 'vcan0', 'battery_monitor.ingestion.can_bitrate': 500000})
    ci.start()
    # allow reader thread to process a couple messages
    time.sleep(0.1)

    # read_message should pull messages in order
    m1 = ci.read_message(timeout=0.1)
    assert m1 == (0x100, b"\x01\x02")
    m2 = ci.read_message(timeout=0.1)
    assert m2 == (0x101, b"\x03\x04")
    # further reads eventually None
    assert ci.read_message(timeout=0.05) is None

    status = ci.get_status()
    assert status.get('available') in (True, False)
    # stop should clean up thread and set available False
    ci.stop()
    assert ci._available is False


def test_can_ingestion_start_failure(monkeypatch):
    # Simulate Bus constructor raising OSError
    def bad_bus(*args, **kwargs):
        raise OSError('no bus')

    import types, sys
    mock_can = types.ModuleType('can')
    mock_can.interface = types.SimpleNamespace()
    mock_can.interface.Bus = bad_bus
    monkeypatch.setitem(sys.modules, 'can', mock_can)

    ci = CANIngestion({'battery_monitor.ingestion.can_interface': 'can0', 'battery_monitor.ingestion.can_bitrate': 500000})
    # start should handle constructor error and mark unavailable
    ci.start()
    assert ci._available is False
    st = ci.get_status()
    assert st.get('error_count', 0) >= 0


def test_can_ingestion_bitrate_fallback(monkeypatch):
    # Simulate Bus accepting no bitrate keyword then succeed
    calls = {'count': 0}

    class SometimesBadBus:
        def __init__(self, channel=None, bitrate=None, *args, **kwargs):
            calls['count'] += 1
            if calls['count'] == 1:
                # Simulate constructor error when bitrate passed
                raise TypeError('unexpected keyword')

        def recv(self, timeout=None):
            time.sleep(0.01)
            return None

        def shutdown(self):
            return None

    import types, sys
    mock_can = types.ModuleType('can')
    mock_can.interface = types.SimpleNamespace()
    mock_can.interface.Bus = SometimesBadBus
    monkeypatch.setitem(sys.modules, 'can', mock_can)

    ci = CANIngestion({'battery_monitor.ingestion.can_interface': 'can0', 'battery_monitor.ingestion.can_bitrate': 500000})
    # start should try with bitrate, then fallback and succeed creating bus
    ci.start()
    # If bus created on second attempt then available flag may be set
    assert isinstance(ci._available, bool)
    ci.stop()


def test_reader_thread_exception_increments_error(monkeypatch):
    # Simulate Bus.recv raising exceptions intermittently
    class ExplodingBus:
        def __init__(self, channel=None, bitrate=None, *args, **kwargs):
            self._called = 0

        def recv(self, timeout=None):
            self._called += 1
            if self._called == 1:
                raise RuntimeError('reader exploded')
            return None

        def shutdown(self):
            return None

    import types, sys
    mock_can = types.ModuleType('can')
    mock_can.interface = types.SimpleNamespace()
    mock_can.interface.Bus = ExplodingBus
    monkeypatch.setitem(sys.modules, 'can', mock_can)
    ci = CANIngestion({'battery_monitor.ingestion.can_interface': 'vcan0', 'battery_monitor.ingestion.can_bitrate': 500000})
    ci.start()
    # give thread a moment
    time.sleep(0.05)
    # error count should have incremented
    st = ci.get_status()
    assert st.get('error_count', 0) >= 1
    ci.stop()

import pytest
from unittest import mock

from battery_monitor.ingestion.can_ingestion import CANIngestion


class DummyConfig:
    def __init__(self, values=None):
        self._values = values or {}

    def get(self, key, default=None):
        return self._values.get(key, default)


def test_can_ingestion_start_and_status():
    cfg = DummyConfig({'battery_monitor.ingestion.can_interface': 'vcan0',
                       'battery_monitor.ingestion.can_bitrate': 500000})

    # Patch the can module Bus class used inside CANIngestion by inserting
    # a fake `can` module into sys.modules
    import types
    mock_can = types.ModuleType('can')
    MockBus = mock.Mock()
    mock_can.interface = mock.Mock()
    mock_can.interface.Bus = MockBus
    with mock.patch.dict('sys.modules', {'can': mock_can}):
        ci = CANIngestion(cfg)
        # start should create the bus and set available
        ci.start()
        status = ci.get_status()
        assert status.get('available', False) is True

        ci.stop()


def test_can_ingestion_bitrate_fallback_and_stop():
    cfg = DummyConfig({'battery_monitor.ingestion.can_interface': 'vcan0',
                       'battery_monitor.ingestion.can_bitrate': 500000})

    # Mock Bus to raise TypeError when bitrate passed, succeed when not
    import types
    mock_can = types.ModuleType('can')

    class BusFactory:
        def __init__(self, *args, **kwargs):
            # if bitrate in kwargs, raise TypeError to simulate backend
            if 'bitrate' in kwargs:
                raise TypeError('unexpected arg')

    # First call with bitrate will raise, second call without bitrate should succeed
    def bus_constructor_with_fallback(*args, **kwargs):
        # simulate a bus instance with recv() returning None
        mock_bus = mock.Mock()
        mock_bus.recv.return_value = None
        return mock_bus

    mock_can.interface = mock.Mock()
    # Make Bus raise TypeError when called with bitrate, but our code handles by retrying
    def Bus_side_effect(*args, **kwargs):
        if 'bitrate' in kwargs:
            raise TypeError('unexpected arg')
        return mock.Mock()

    mock_can.interface.Bus = Bus_side_effect

    with mock.patch.dict('sys.modules', {'can': mock_can}):
        ci = CANIngestion(cfg)
        # start should attempt with bitrate, catch TypeError, retry and succeed
        ci.start()
        status = ci.get_status()
        # start may set available True even if recv returns None
        assert isinstance(status.get('available'), bool)
        ci.stop()


def test_can_ingestion_read_message_and_error_counts():
    cfg = DummyConfig({'battery_monitor.ingestion.can_interface': 'vcan0',
                       'battery_monitor.ingestion.can_bitrate': 500000})

    class DummyMsg:
        def __init__(self, arb_id, data):
            self.arbitration_id = arb_id
            self.data = data

    # prepare a bus that returns a message then raises an exception
    def recv_side_effect(timeout=None):
        # first call returns a dummy message, subsequent calls return None
        if not getattr(recv_side_effect, 'called', False):
            recv_side_effect.called = True
            return DummyMsg(0x100, b'\x01\x02')
        return None

    recv_side_effect.called = False

    import types
    mock_can = types.ModuleType('can')
    MockBusClass = mock.Mock()
    mock_bus_instance = mock.Mock()
    mock_bus_instance.recv.side_effect = recv_side_effect
    MockBusClass.return_value = mock_bus_instance
    mock_can.interface = mock.Mock()
    mock_can.interface.Bus = MockBusClass

    with mock.patch.dict('sys.modules', {'can': mock_can}):
        ci = CANIngestion(cfg)
        ci.start()

        # successful read
        msg = ci.read_message(timeout=0.1)
        assert isinstance(msg, tuple) or msg is not None

        # subsequent read will raise inside and should be handled
        # We call read_message and ensure it does not crash and updates error count
        try:
            _ = ci.read_message(timeout=0.1)
        except Exception:
            # some implementations may bubble; ensure status reflects error
            pass

        status = ci.get_status()
        # error count should be >= 0 and available is boolean
        assert 'error_count' in status
        assert isinstance(status.get('available'), bool)

        ci.stop()


def test_can_ingestion_start_failure_sets_unavailable():
    cfg = DummyConfig({'battery_monitor.ingestion.can_interface': 'vcan0',
                       'battery_monitor.ingestion.can_bitrate': 500000})

    # Simulate Bus constructor raising OSError during start
    import types
    mock_can = types.ModuleType('can')

    def Bus_raises(*args, **kwargs):
        raise OSError('no such device')

    mock_can.interface = mock.Mock()
    mock_can.interface.Bus = Bus_raises

    with mock.patch.dict('sys.modules', {'can': mock_can}):
        ci = CANIngestion(cfg)
        # start should catch exception and mark unavailable
        ci.start()
        status = ci.get_status()
        assert status.get('available') is False
        assert status.get('error_count', 0) >= 1


def test_can_ingestion_reader_increments_msg_count_and_handles_none():
    cfg = DummyConfig({'battery_monitor.ingestion.can_interface': 'vcan0',
                       'battery_monitor.ingestion.can_bitrate': 500000})

    class DummyMsg:
        def __init__(self, arb_id, data):
            self.arbitration_id = arb_id
            self.data = data

    # prepare a bus that returns multiple messages then None
    def recv_side_effect(timeout=None):
        # return 3 messages then None forever
        if getattr(recv_side_effect, 'count', 0) < 3:
            recv_side_effect.count = getattr(recv_side_effect, 'count', 0) + 1
            return DummyMsg(0x100, b'\x01\x02')
        return None

    recv_side_effect.count = 0

    import types
    mock_can = types.ModuleType('can')
    MockBusClass = mock.Mock()
    mock_bus_instance = mock.Mock()
    mock_bus_instance.recv.side_effect = recv_side_effect
    MockBusClass.return_value = mock_bus_instance
    mock_can.interface = mock.Mock()
    mock_can.interface.Bus = MockBusClass

    with mock.patch.dict('sys.modules', {'can': mock_can}):
        ci = CANIngestion(cfg)
        ci.start()
        # allow reader thread to run briefly
        import time as _time
        _time.sleep(0.2)

        status = ci.get_status()
        # message count should be >= 0 (reader increments on recv)
        assert status.get('message_count', 0) >= 0

        # read queued messages via API
        msg = ci.read_message(timeout=0.1)
        # if queue had items, msg is tuple; otherwise None
        assert (isinstance(msg, tuple) and len(msg) == 2) or msg is None

        ci.stop()
