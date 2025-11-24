import csv

from battery_monitor.ingestion.csv_simulator import BMSCSVSimulator


def test_generate_sample(tmp_path):
    sim = BMSCSVSimulator(config={}, out_path=str(tmp_path / 'sim.csv'))
    path = sim.generate_sample(duration_s=3, sample_rate_hz=1)
    assert path
    with open(path, 'r', encoding='utf-8') as fh:
        rdr = list(csv.reader(fh))
    # header + 3 rows
    assert len(rdr) >= 4


def test_read_next_sample_and_reset(tmp_path):
    sim = BMSCSVSimulator(config={}, out_path=str(tmp_path / 'sim2.csv'))
    sim.generate_sample(duration_s=2, sample_rate_hz=1)
    sim.open()
    s1 = sim.read_next_sample()
    s2 = sim.read_next_sample()
    assert s1 is not None and s2 is not None
    sim.reset()
    s3 = sim.read_next_sample()
    assert s3 is not None
