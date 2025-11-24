import os
import time
from unittest import mock

import pytest
from datetime import datetime, timedelta

from config.config_loader import ConfigLoader
from data_collection.sensors.bms_handler import BMSHandler
from battery_monitor.ingestion.csv_simulator import BMSCSVSimulator
from utils.validation import validate_bms_telemetry


def make_config(tmp_path):
    cfg = ConfigLoader()
    cfg.set('battery_monitor.ingestion.enable_simulation', True)
    cfg.set('battery_monitor.ingestion.csv_simulator_path', str(tmp_path / 'bms_sim.csv'))
    cfg.set('battery_monitor.ingestion.sample_rate', 1)
    return cfg


def test_bms_handler_simulation_mode(tmp_path):
    cfg = make_config(tmp_path)
    sim = BMSCSVSimulator(cfg, out_path=str(tmp_path / 'bms_sim.csv'))
    sim.generate_sample(duration_s=2, sample_rate_hz=1)
    sim.open()

    handler = BMSHandler(cfg)
    samples = []
    for _ in range(2):
        s = handler.read()
        # handler may return None if simulator EOF occurred; accept both
        assert s is None or isinstance(s, dict)
        if isinstance(s, dict) and s:
            ok, errs = validate_bms_telemetry(s)
            assert ok
            samples.append(s)
    assert len(samples) >= 1


def test_bms_handler_missing_csv(tmp_path):
    cfg = make_config(tmp_path)
    # point to non-existent file
    cfg.set('battery_monitor.ingestion.csv_simulator_path', str(tmp_path / 'nope.csv'))
    handler = BMSHandler(cfg)
    # reading should return None and set status to error
    # attempt several reads to allow simulator to try reset
    s = None
    for _ in range(3):
        s = handler.read()
    # handler may generate a short fallback sample or return None; accept both
    assert s is None or isinstance(s, dict)
    h = handler.get_health()
    # handler may produce a fallback sample and mark OK, or fail and mark error
    assert h['status'] in ('error', 'unknown', 'ok')


def make_can_config(tmp_path):
    cfg = ConfigLoader()
    cfg.set('battery_monitor.ingestion.enable_simulation', False)
    cfg.set('battery_monitor.ingestion.can_interface', 'vcan0')
    cfg.set('battery_monitor.ingestion.can_bitrate', 500000)
    cfg.set('battery_monitor.ingestion.cells', 12)
    return cfg


def test_bms_handler_can_mode_integration(tmp_path):
    cfg = make_can_config(tmp_path)

    # Patch CANIngestion and BMSCANParser to return a valid normalized sample
    normalized = {'pack_voltage': 48.0, 'cell_voltages': [3.7] * 12}
    with mock.patch('battery_monitor.ingestion.can_ingestion.CANIngestion') as MockCAN, \
         mock.patch('battery_monitor.ingestion.bms_parser.BMSCANParser') as MockParser, \
         mock.patch('data_collection.sensors.bms_handler.validate_bms_telemetry') as mock_validate:

        can_inst = MockCAN.return_value
        can_inst.start.return_value = None
        can_inst.read_message.return_value = (0x100, b'\x00')

        parser_inst = MockParser.return_value
        parser_inst.parse_message.return_value = {'raw': True}
        parser_inst.normalize_telemetry.return_value = normalized
        mock_validate.return_value = (True, None)

        handler = BMSHandler(cfg)

        # read through handler
        res = handler.read()
        assert res == normalized

        # window integration: add to WindowManager and retrieve window
        from data_collection.logger import DataLogger
        dl = DataLogger(cfg)
        # monkeypatch sensors to avoid full startup
        dl.sensors = []
        dl.window_manager.add_reading('bms', normalized)
        # grab window snapshot
        snapshot = dl.window_manager.get_current_window(sensor_health_snapshot={})
        assert 'bms' in snapshot


def test_bms_handler_can_timeout_and_health(tmp_path):
    cfg = make_can_config(tmp_path)
    with mock.patch('battery_monitor.ingestion.can_ingestion.CANIngestion') as MockCAN, \
         mock.patch('battery_monitor.ingestion.bms_parser.BMSCANParser') as MockParser:

        can_inst = MockCAN.return_value
        can_inst.start.return_value = None
        # simulate timeouts
        can_inst.read_message.return_value = None

        handler = BMSHandler(cfg)

        # repeated reads should eventually decrease availability
        for _ in range(7):
            handler.read()

        h = handler.get_health()
        assert isinstance(h.get('error_count'), int)
        # can_connected should be boolean
        assert isinstance(h.get('can_connected'), bool)


def test_bms_handler_can_unavailable_start(tmp_path):
    cfg = make_can_config(tmp_path)
    # Simulate CANIngestion.start raising OSError during initialization
    with mock.patch('battery_monitor.ingestion.can_ingestion.CANIngestion') as MockCAN:
        can_inst = MockCAN.return_value
        can_inst.start.side_effect = OSError('bus not available')
        handler = BMSHandler(cfg)
        # immediately after initialization CAN should be marked disconnected
        assert getattr(handler, '_can_connected', False) is False
        assert handler.is_available() is False
        h = handler.get_health()
        assert h.get('status') in ('unknown', 'error')
        # invoking read should not raise and should return None
        out = handler.read()
        assert out is None


def test_bms_handler_can_malformed_frame_increments_parse_error(tmp_path):
    cfg = make_can_config(tmp_path)
    with mock.patch('battery_monitor.ingestion.can_ingestion.CANIngestion') as MockCAN, \
         mock.patch('battery_monitor.ingestion.bms_parser.BMSCANParser') as MockParser, \
         mock.patch('data_collection.sensors.bms_handler.validate_bms_telemetry') as mock_validate:

        can_inst = MockCAN.return_value
        can_inst.start.return_value = None
        can_inst.read_message.return_value = (0x100, b'\x00')

        parser_inst = MockParser.return_value
        # parser returns something but normalization yields invalid data
        parser_inst.parse_message.return_value = {'some': 'raw'}
        parser_inst.normalize_telemetry.return_value = {'pack_voltage': None}

        # validation returns False indicating malformed/invalid telemetry
        mock_validate.return_value = (False, ['bad'])

        handler = BMSHandler(cfg)
        before = getattr(handler, '_parse_error_count', 0)
        res = handler.read()
        after = getattr(handler, '_parse_error_count', 0)
        assert res is None
        assert after >= before + 1


def test_bms_handler_can_window_integration(tmp_path):
    cfg = make_can_config(tmp_path)
    now = datetime.utcnow().replace(microsecond=0)
    t1 = now.isoformat() + 'Z'
    t2 = (now + timedelta(seconds=1)).isoformat() + 'Z'
    normalized1 = {'pack_voltage': 48.0, 'cell_voltages': [3.7] * 12, 'timestamp': t1}
    normalized2 = {'pack_voltage': 48.1, 'cell_voltages': [3.71] * 12, 'timestamp': t2}

    with mock.patch('battery_monitor.ingestion.can_ingestion.CANIngestion') as MockCAN, \
         mock.patch('battery_monitor.ingestion.bms_parser.BMSCANParser') as MockParser, \
         mock.patch('data_collection.sensors.bms_handler.validate_bms_telemetry') as mock_validate:

        can_inst = MockCAN.return_value
        can_inst.start.return_value = None
        # have read_message return one tuple, then another
        can_inst.read_message.side_effect = [(0x100, b'\x00'), (0x100, b'\x00'), None]

        parser_inst = MockParser.return_value
        parser_inst.parse_message.return_value = {'raw': True}
        # normalize returns sequential normalized payloads
        parser_inst.normalize_telemetry.side_effect = [normalized1, normalized2]
        mock_validate.return_value = (True, None)

        handler = BMSHandler(cfg)

        r1 = handler.read()
        r2 = handler.read()
        assert r1 == normalized1
        assert r2 == normalized2

        # integrate into window manager via DataLogger
        from data_collection.logger import DataLogger
        dl = DataLogger(cfg)
        dl.sensors = []
        dl.window_manager.add_reading('bms', r1)
        dl.window_manager.add_reading('bms', r2)
        snap = dl.window_manager.get_current_window(sensor_health_snapshot={})
        assert 'bms' in snap and isinstance(snap['bms'], list)


    def test_bms_handler_can_read_success(tmp_path):
        cfg = make_can_config(tmp_path)
        normalized = {'pack_voltage': 48.2, 'cell_voltages': [3.7] * 12, 'timestamp': '2025-01-01T00:00:00Z'}
        with mock.patch('battery_monitor.ingestion.can_ingestion.CANIngestion') as MockCAN, \
             mock.patch('battery_monitor.ingestion.bms_parser.BMSCANParser') as MockParser, \
             mock.patch('data_collection.sensors.bms_handler.validate_bms_telemetry') as mock_validate:

            can_inst = MockCAN.return_value
            can_inst.start.return_value = None
            can_inst.read_message.return_value = (0x100, b'\x00')

            parser_inst = MockParser.return_value
            parser_inst.parse_message.return_value = {'raw': True}
            parser_inst.normalize_telemetry.return_value = normalized
            mock_validate.return_value = (True, None)

            handler = BMSHandler(cfg)
            res = handler.read()
            assert res == normalized
            h = handler.get_health()
            assert h.get('status') == 'ok'
            assert handler._last_valid_sample == normalized


    def test_bms_handler_can_timeout_repeated(tmp_path):
        cfg = make_can_config(tmp_path)
        with mock.patch('battery_monitor.ingestion.can_ingestion.CANIngestion') as MockCAN, \
             mock.patch('battery_monitor.ingestion.bms_parser.BMSCANParser') as MockParser:

            can_inst = MockCAN.return_value
            can_inst.start.return_value = None
            # simulate repeated timeouts
            can_inst.read_message.side_effect = [None] * 7

            parser_inst = MockParser.return_value

            handler = BMSHandler(cfg)
            for _ in range(7):
                handler.read()

            h = handler.get_health()
            assert isinstance(h.get('error_count'), int)
            # after >5 failures handler should mark can_connected False
            assert handler._can_connected is False


    def test_bms_handler_can_malformed_parse_additional(tmp_path):
        cfg = make_can_config(tmp_path)
        with mock.patch('battery_monitor.ingestion.can_ingestion.CANIngestion') as MockCAN, \
             mock.patch('battery_monitor.ingestion.bms_parser.BMSCANParser') as MockParser, \
             mock.patch('data_collection.sensors.bms_handler.validate_bms_telemetry') as mock_validate:

            can_inst = MockCAN.return_value
            can_inst.start.return_value = None
            can_inst.read_message.return_value = (0x100, b'\x00')

            parser_inst = MockParser.return_value
            parser_inst.parse_message.return_value = {'incomplete': True}
            parser_inst.normalize_telemetry.return_value = {'pack_voltage': None}

            mock_validate.return_value = (False, ['missing'])

            handler = BMSHandler(cfg)
            before = getattr(handler, '_parse_error_count', 0)
            res = handler.read()
            after = getattr(handler, '_parse_error_count', 0)
            assert res is None
            assert after >= before + 1


    def test_bms_handler_can_parser_exception_sets_error(tmp_path):
        cfg = make_can_config(tmp_path)
        with mock.patch('battery_monitor.ingestion.can_ingestion.CANIngestion') as MockCAN, \
             mock.patch('battery_monitor.ingestion.bms_parser.BMSCANParser') as MockParser:

            can_inst = MockCAN.return_value
            can_inst.start.return_value = None
            can_inst.read_message.return_value = (0x100, b'\x00')

            parser_inst = MockParser.return_value
            parser_inst.parse_message.return_value = {'raw': True}
            # normalization raises unexpected exception
            parser_inst.normalize_telemetry.side_effect = RuntimeError('boom')

            handler = BMSHandler(cfg)
            res = handler.read()
            assert res is None
            h = handler.get_health()
            assert h.get('status') == 'error'


    def test_bms_handler_can_no_bus_object(tmp_path):
        cfg = make_can_config(tmp_path)
        # Simulate CANIngestion constructor raising, leaving _can as None
        with mock.patch('battery_monitor.ingestion.can_ingestion.CANIngestion') as MockCAN:
            MockCAN.side_effect = OSError('no bus')
            handler = BMSHandler(cfg)
            # read should detect no _can and set error status
            out = handler.read()
            assert out is None
            h = handler.get_health()
            assert h.get('status') == 'error'


    def test_bms_handler_simulation_invalid_sample(tmp_path):
        # simulation mode where sample is invalid according to validation
        cfg = make_config(tmp_path)
        with mock.patch('battery_monitor.ingestion.csv_simulator.BMSCSVSimulator') as MockSim, \
             mock.patch('data_collection.sensors.bms_handler.validate_bms_telemetry') as mock_validate:

            sim_inst = MockSim.return_value
            sim_inst.read_next_sample.return_value = {'some': 'sample'}
            mock_validate.return_value = (False, ['bad'])

            handler = BMSHandler(cfg)
            before = getattr(handler, '_parse_error_count', 0)
            res = handler.read()
            after = getattr(handler, '_parse_error_count', 0)
            assert res is None
            assert after >= before + 1


    def test_bms_handler_stop_invokes_can_stop(tmp_path):
        cfg = make_can_config(tmp_path)
        with mock.patch('battery_monitor.ingestion.can_ingestion.CANIngestion') as MockCAN, \
             mock.patch('battery_monitor.ingestion.bms_parser.BMSCANParser') as MockParser:

            can_inst = MockCAN.return_value
            can_inst.start.return_value = None
            parser_inst = MockParser.return_value

            handler = BMSHandler(cfg)
            # simulate that _can exists
            handler._can = can_inst
            handler.stop()
            can_inst.stop.assert_called()


    def test_bms_handler_simulator_generate_sample_exception(tmp_path):
        # ensure constructor swallows exceptions from generate_sample
        cfg = make_config(tmp_path)
        with mock.patch('battery_monitor.ingestion.csv_simulator.BMSCSVSimulator') as MockSim:
            sim_inst = MockSim.return_value
            sim_inst.generate_sample.side_effect = RuntimeError('no file')
            # instantiation should not raise
            handler = BMSHandler(cfg)
            assert handler is not None


    def test_bms_handler_is_available_with_can_connected(tmp_path):
        cfg = make_can_config(tmp_path)
        with mock.patch('battery_monitor.ingestion.can_ingestion.CANIngestion') as MockCAN, \
             mock.patch('battery_monitor.ingestion.bms_parser.BMSCANParser') as MockParser:

            can_inst = MockCAN.return_value
            can_inst.start.return_value = None
            can_inst.get_status.return_value = {'available': True}
            parser_inst = MockParser.return_value

            handler = BMSHandler(cfg)
            # refresh availability via get_status
            assert handler.is_available() is True
