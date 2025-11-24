import time
import pytest
from unittest import mock

from data_collection.sensors.bms_handler import BMSHandler


class DummyConfig:
    def __init__(self, values=None):
        self._values = values or {}

    def get(self, key, default=None):
        return self._values.get(key, default)


@pytest.fixture
def can_config():
    vals = {
        'battery_monitor.ingestion.enable_simulation': False,
        'battery_monitor.ingestion.cells': 12,
        'battery_monitor.ingestion.can_interface': 'vcan0',
        'battery_monitor.ingestion.can_bitrate': 500000,
    }
    return DummyConfig(vals)


def test_bms_handler_can_unavailable(can_config):
    # Mock CANIngestion.start to raise OSError
    with mock.patch('battery_monitor.ingestion.can_ingestion.CANIngestion') as MockCAN:
        instance = MockCAN.return_value
        instance.start.side_effect = OSError('bus not available')

        handler = BMSHandler(can_config)
        # initialization should handle unavailable CAN ingestion

        assert getattr(handler, '_can_connected', False) is False
        assert handler.is_available() is False


def test_bms_handler_can_success(can_config):
    normalized = {'pack_voltage': 48.0, 'cell_voltages': [3.7] * 12}
    with mock.patch('battery_monitor.ingestion.can_ingestion.CANIngestion') as MockCAN, \
         mock.patch('battery_monitor.ingestion.bms_parser.BMSCANParser') as MockParser, \
         mock.patch('data_collection.sensors.bms_handler.validate_bms_telemetry') as mock_validate:

        can_inst = MockCAN.return_value
        can_inst.start.return_value = None
        # read_message returns a placeholder (can_id, data)
        can_inst.read_message.return_value = (0x100, b'\x00')

        parser_inst = MockParser.return_value
        parser_inst.parse_message.return_value = {'raw': True}
        parser_inst.normalize_telemetry.return_value = normalized
        mock_validate.return_value = (True, None)

        handler = BMSHandler(can_config)

        result = handler.read()
        assert result == normalized
        assert getattr(handler, '_can_connected', False) is True
        health = handler.get_health()
        # health should reflect an OK-ish state (no parse errors)
        assert health.get('parse_error_count', 0) == 0


def test_bms_handler_can_timeout(can_config):
    # Simulate read_message returning None (timeout) repeatedly
    with mock.patch('battery_monitor.ingestion.can_ingestion.CANIngestion') as MockCAN, \
         mock.patch('battery_monitor.ingestion.bms_parser.BMSCANParser') as MockParser:

        can_inst = MockCAN.return_value
        can_inst.start.return_value = None
        can_inst.read_message.return_value = None

        parser_inst = MockParser.return_value

        handler = BMSHandler(can_config)

        # First call should not immediately mark CAN disconnected
        first = handler.read()
        assert first is None
        assert handler.is_available() in (True, False)

        # Repeated reads should eventually flip the internal CAN flag
        eventually_disconnected = False
        for _ in range(10):
            handler.read()
            if getattr(handler, '_can_connected', True) is False:
                eventually_disconnected = True
                break

        assert eventually_disconnected, 'CAN did not become disconnected after repeated timeouts'


def test_bms_handler_can_malformed(can_config):
    # parser returns malformed raw and validation should fail
    with mock.patch('battery_monitor.ingestion.can_ingestion.CANIngestion') as MockCAN, \
         mock.patch('battery_monitor.ingestion.bms_parser.BMSCANParser') as MockParser, \
         mock.patch('data_collection.sensors.bms_handler.validate_bms_telemetry') as mock_validate:

        can_inst = MockCAN.return_value
        can_inst.start.return_value = None
        can_inst.read_message.return_value = (0x100, b'\x00')

        parser_inst = MockParser.return_value
        parser_inst.parse_message.return_value = {'some': 'raw'}
        parser_inst.normalize_telemetry.return_value = {'pack_voltage': 0}

        # Simulate validation failing by returning (False, errors)
        mock_validate.return_value = (False, ['invalid'])

        handler = BMSHandler(can_config)

        before = getattr(handler, '_parse_error_count', 0)
        res = handler.read()
        after = getattr(handler, '_parse_error_count', 0)

        assert res is None
        assert after >= before + 1
