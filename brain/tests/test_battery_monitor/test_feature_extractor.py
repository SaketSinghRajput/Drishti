import pytest

from battery_monitor.ml.feature_extractor import BMSFeatureExtractor


def make_sample(volts=None, temps=None, current=0.0, soc=90.0, soh=95.0, faults=None):
    return {"cell_voltages": volts, "cell_temps": temps, "pack_current": current, "soc": soc, "soh": soh, "fault_codes": faults}


def test_extract_features_single_sample():
    fe = BMSFeatureExtractor()
    s = make_sample(volts=[3.7] * 4, temps=[30.0] * 4, current=10.0)
    features = fe.extract_features([s])
    assert features["cell_voltage_mean"] == pytest.approx(3.7)
    assert features["cell_temp_mean"] == pytest.approx(30.0)
    assert features["current_rms"] == pytest.approx(10.0)


def test_extract_features_empty_list():
    fe = BMSFeatureExtractor()
    with pytest.raises(ValueError):
        fe.extract_features([])


def test_missing_cell_voltages_uses_sentinel():
    fe = BMSFeatureExtractor()
    s = make_sample(volts=None, temps=[25.0])
    features = fe.extract_features([s])
    assert features["cell_voltage_mean"] == -999.0
