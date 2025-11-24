// Minimal dashboard JS: polls API endpoints and updates charts/cards
document.addEventListener('DOMContentLoaded', function () {
  // placeholders for charts
  const vcCtx = document.getElementById('vcChart').getContext('2d');
  const cellVoltCtx = document.getElementById('cellVoltChart').getContext('2d');
  const cellTempCtx = document.getElementById('cellTempChart').getContext('2d');
  const socSohCtx = document.getElementById('socSohChart').getContext('2d');

  const vcChart = new Chart(vcCtx, {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Voltage (V)', data: [], borderColor: 'rgba(75,192,192,1)', tension: 0.2 }, { label: 'Current (A)', data: [], borderColor: 'rgba(255,99,132,1)', tension: 0.2, yAxisID: 'B' }] },
    options: { responsive: true, scales: { B: { position: 'right' } } }
  });

  const cellChart = new Chart(cellVoltCtx, { type: 'bar', data: { labels: [], datasets: [{ label: 'Cell Voltage', data: [], backgroundColor: 'rgba(54,162,235,0.6)' }] }, options: { responsive: true } });

  const cellTempChart = new Chart(cellTempCtx, { type: 'bar', data: { labels: [], datasets: [{ label: 'Cell Temp (°C)', data: [], backgroundColor: 'rgba(255,159,64,0.6)' }] }, options: { responsive: true } });

  const socSohChart = new Chart(socSohCtx, { type: 'line', data: { labels: [], datasets: [{ label: 'SoC (%)', data: [], borderColor: 'rgba(75,192,192,1)', tension: 0.2 }, { label: 'SoH (%)', data: [], borderColor: 'rgba(153,102,255,1)', tension: 0.2 }] }, options: { responsive: true } });

  function updateCards(sample) {
    document.getElementById('pack_voltage').textContent = sample.pack_voltage ? sample.pack_voltage.toFixed(2) + ' V' : '-- V';
    document.getElementById('pack_current').textContent = sample.pack_current ? sample.pack_current.toFixed(2) + ' A' : '-- A';
    document.getElementById('soc').textContent = sample.soc ? sample.soc.toFixed(1) + ' %' : '-- %';
    document.getElementById('soh').textContent = sample.soh ? sample.soh.toFixed(1) + ' %' : '-- %';
    const cm = sample.computed_metrics || {};
    document.getElementById('voltage_diff').textContent = cm.voltage_diff ? cm.voltage_diff.toFixed(3) + ' V' : '-- V';
  }

  async function fetchLatest() {
    try {
      const res = await fetch('/api/telemetry/latest');
      const data = await res.json();
      const sample = data.latest || data;
      if (!sample || sample.latest === null) return;
      // update charts
      const t = sample.timestamp || new Date().toISOString();
      vcChart.data.labels.push(t);
      vcChart.data.datasets[0].data.push(sample.pack_voltage || null);
      vcChart.data.datasets[1].data.push(sample.pack_current || null);
      if (vcChart.data.labels.length > 60) { vcChart.data.labels.shift(); vcChart.data.datasets.forEach(ds => ds.data.shift()); }
      vcChart.update();

      // cell voltages
      const cvs = sample.cell_voltages || [];
      cellChart.data.labels = cvs.map((_, i) => 'C' + (i+1));
      cellChart.data.datasets[0].data = cvs;
      cellChart.update();

      // cell temperatures
      const cts = sample.cell_temps || [];
      cellTempChart.data.labels = cts.map((_, i) => 'T' + (i+1));
      cellTempChart.data.datasets[0].data = cts;
      cellTempChart.update();

      // SoC / SoH trends (sliding window 60)
      socSohChart.data.labels.push(t);
      socSohChart.data.datasets[0].data.push(sample.soc || null);
      socSohChart.data.datasets[1].data.push(sample.soh || null);
      if (socSohChart.data.labels.length > 60) { socSohChart.data.labels.shift(); socSohChart.data.datasets.forEach(ds => ds.data.shift()); }
      socSohChart.update();

      updateCards(sample);
      // highlight high temp
      const maxTemp = (sample.computed_metrics && sample.computed_metrics.max_cell_temp) || null;
      const sohCard = document.getElementById('soh');
      if (maxTemp !== null && maxTemp > 60) {
        sohCard.classList.add('text-danger');
      } else {
        sohCard.classList.remove('text-danger');
      }
    } catch (err) {
      console.error('Failed to fetch latest telemetry', err);
      document.getElementById('status').textContent = 'Disconnected';
      document.getElementById('status').className = 'text-danger';
    }
  }

  async function fetchAlerts() {
    try {
      const res = await fetch('/api/alerts?limit=20');
      const obj = await res.json();
      const tbody = document.querySelector('#alertsTable tbody');
      tbody.innerHTML = '';
      for (const a of obj.alerts || []) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${a.ts||a.ts||''}</td><td><span class='badge ${'badge-'+(a.severity||'LOW')}' style='padding:.5em'>${a.severity||''}</span></td><td>${a.source||''}</td><td>${a.reason||''}</td><td>${a.alert&&a.alert.anomaly_score? a.alert.anomaly_score:'' || a.anomaly_score||''}</td>`;
        tbody.appendChild(tr);
      }
    } catch (err) {
      console.error('Failed to fetch alerts', err);
    }
  }

  // start polling
  fetchLatest(); fetchAlerts();
  const p1 = setInterval(fetchLatest, 1000);
  const p2 = setInterval(fetchAlerts, 5000);

  window.addEventListener('beforeunload', () => { clearInterval(p1); clearInterval(p2); });
});
