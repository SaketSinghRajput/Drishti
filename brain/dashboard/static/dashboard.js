// Dashboard JS: real-time polling, incidents, modal, annotations, analytics
const DEFAULT_POLL_MS = 1000;
let pollIntervalMs = DEFAULT_POLL_MS;
let riskChart = null;
let incidentTimelineChart = null;
let analyticsCharts = {};

let fatigueGauge = null;
let fatigueMetricsChart = null;
let fatigueEventSource = null;
let sseErrorCount = 0;
let lastFatigueAlertAt = 0;
let fatiguePollingInterval = null;

let bleNeighborsChart = null;
let blePacketStatsGauge = null;
let bleEventSource = null;
let bleSSEErrorCount = 0;
let blePollingInterval = null;

async function fetchJson(url, opts) {
  try {
    const r = await fetch(url, opts);
    return await r.json();
  } catch (err) {
    console.error("fetch error", err);
    return null;
  }
}

function initRiskChart() {
  const ctx = document.getElementById('riskChart').getContext('2d');
  riskChart = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Risk Score', data: [], borderColor: 'rgba(255,99,132,1)', tension: 0.2, fill:false }] },
    options: { scales: { x: { display: false }, y: { min: 0, max: 1 } }, animation:false }
  });
}

function initFatigueGauge() {
  const ctx = document.getElementById('fatigueGauge').getContext('2d');
  fatigueGauge = new Chart(ctx, {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [0, 100],
        backgroundColor: ['rgba(76,175,80,0.8)', 'rgba(200,200,200,0.2)'],
        borderWidth: 0,
        circumference: 180,
        rotation: 270
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '75%',
      animation: { duration: 300 },
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false }
      }
    }
  });
}

function initFatigueMetricsChart() {
  const ctx = document.getElementById('fatigueMetricsChart').getContext('2d');
  fatigueMetricsChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'PERCLOS', data: [], borderColor: 'rgba(255,159,64,1)', tension: 0.2, fill: false },
        { label: 'Blink Rate', data: [], borderColor: 'rgba(54,162,235,1)', tension: 0.2, fill: false, yAxisID: 'y1' }
      ]
    },
    options: {
      responsive: true,
      scales: {
        x: { display: false },
        y: { min: 0, max: 1, title: { display: true, text: 'PERCLOS' } },
        y1: { position: 'right', min: 0, max: 30, title: { display: true, text: 'Blinks/min' } }
      },
      animation: { duration: 300 }
    }
  });
}

function initBLENeighborsChart() {
  const ctx = document.getElementById('bleNeighborsChart').getContext('2d');
  bleNeighborsChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Neighbors Count',
        data: [],
        borderColor: 'rgba(33,150,243,1)',
        backgroundColor: 'rgba(33,150,243,0.1)',
        tension: 0.2,
        fill: true
      }]
    },
    options: {
      responsive: true,
      scales: {
        x: { display: false },
        y: { min: 0, beginAtZero: true, ticks: { stepSize: 1 } }
      },
      animation: { duration: 300 }
    }
  });
}

function initBLEPacketStatsGauge() {
  const ctx = document.getElementById('blePacketStatsGauge').getContext('2d');
  blePacketStatsGauge = new Chart(ctx, {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [0, 0, 100],
        backgroundColor: [
          'rgba(76,175,80,0.8)',
          'rgba(33,150,243,0.8)',
          'rgba(200,200,200,0.2)'
        ],
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '70%',
      animation: { duration: 300 },
      plugins: {
        legend: { display: false },
        tooltip: { enabled: true }
      }
    }
  });
}

function connectFatigueSSE() {
  if (fatigueEventSource) {
    fatigueEventSource.close();
  }

  fatigueEventSource = new EventSource('/api/fatigue/stream');
  sseErrorCount = 0;

  fatigueEventSource.onmessage = function(event) {
    try {
      const data = JSON.parse(event.data);
      updateFatigueUI(data);
      sseErrorCount = 0;
    } catch (err) {
      console.error('Failed to parse fatigue SSE data', err);
    }
  };

  fatigueEventSource.onerror = function(err) {
    console.error('Fatigue SSE error', err);
    sseErrorCount++;
    if (sseErrorCount >= 3) {
      console.warn('SSE failed 3+ times, falling back to polling');
      if (fatigueEventSource) {
        fatigueEventSource.close();
        fatigueEventSource = null;
      }
      if (!fatiguePollingInterval) {
        fatiguePollingInterval = setInterval(pollFatigue, 1000);
      }
    } else {
      fatigueEventSource.close();
      setTimeout(connectFatigueSSE, 5000);
    }
  };
}

function updateFatigueUI(data) {
  if (!data || typeof data !== 'object') return;

  const score = data.drowsiness_score || 0.0;
  const state = data.fatigue_state || 'UNKNOWN';
  const perclos = data.perclos || 0.0;
  const blinkRate = data.blink_rate || 0.0;
  const headPose = (data.head_pose && typeof data.head_pose === 'object') ? data.head_pose : { pitch: 0, yaw: 0, roll: 0 };

  if (fatigueGauge) {
    const percentage = score * 100;
    fatigueGauge.data.datasets[0].data = [percentage, 100 - percentage];
    const colors = {
      'NORMAL': 'rgba(76,175,80,0.8)',
      'TIRED': 'rgba(255,235,59,0.8)',
      'DROWSY': 'rgba(255,152,0,0.8)',
      'MICROSLEEP': 'rgba(244,67,54,0.8)'
    };
    fatigueGauge.data.datasets[0].backgroundColor[0] = colors[state] || colors['NORMAL'];
    fatigueGauge.update('none');
  }

  const stateBadge = document.getElementById('fatigueState');
  if (stateBadge) {
    stateBadge.textContent = state;
    stateBadge.className = 'badge fs-5 ' + getStateBadgeClass(state);
    stateBadge.style.transition = 'all 0.3s ease-in-out';
  }

  if (fatigueMetricsChart) {
    const ts = new Date().toLocaleTimeString();
    fatigueMetricsChart.data.labels.push(ts);
    fatigueMetricsChart.data.datasets[0].data.push(perclos);
    fatigueMetricsChart.data.datasets[1].data.push(blinkRate);
    if (fatigueMetricsChart.data.labels.length > 60) {
      fatigueMetricsChart.data.labels.shift();
      fatigueMetricsChart.data.datasets[0].data.shift();
      fatigueMetricsChart.data.datasets[1].data.shift();
    }
    fatigueMetricsChart.update('none');
  }

  const pitchEl = document.getElementById('headPitch');
  if (pitchEl) pitchEl.textContent = headPose.pitch.toFixed(1) + ' deg';
  const yawEl = document.getElementById('headYaw');
  if (yawEl) yawEl.textContent = headPose.yaw.toFixed(1) + ' deg';
  const rollEl = document.getElementById('headRoll');
  if (rollEl) rollEl.textContent = headPose.roll.toFixed(1) + ' deg';

  triggerFatigueAlert(state);
}

function getStateBadgeClass(state) {
  const classes = {
    'NORMAL': 'bg-success',
    'TIRED': 'bg-warning',
    'DROWSY': 'bg-orange',
    'MICROSLEEP': 'bg-danger'
  };
  return classes[state] || 'bg-secondary';
}

function triggerFatigueAlert(state) {
  if (state === 'DROWSY' || state === 'MICROSLEEP') {
    const now = Date.now();
    const timeSinceLastAlert = now - lastFatigueAlertAt;
    if (timeSinceLastAlert < 10000) {
      return;
    }
    lastFatigueAlertAt = now;

    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-danger alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3';
    alertDiv.style.zIndex = '9999';
    alertDiv.innerHTML = `
      <strong>[ALERT] ${state} DETECTED!</strong> Driver attention required.
      <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.body.appendChild(alertDiv);

    setTimeout(() => {
      alertDiv.classList.remove('show');
      setTimeout(() => alertDiv.remove(), 300);
    }, 5000);
  }
}

async function pollFatigue() {
  const data = await fetchJson('/api/fatigue');
  if (data) {
    updateFatigueUI(data);
  }
}

function connectBLESSE() {
  if (bleEventSource) {
    bleEventSource.close();
  }

  bleEventSource = new EventSource('/api/ble/stream');
  bleSSEErrorCount = 0;

  bleEventSource.onmessage = function(event) {
    try {
      const data = JSON.parse(event.data);
      updateBLEUI(data);
      bleSSEErrorCount = 0;
    } catch (err) {
      console.error('Failed to parse BLE SSE data', err);
    }
  };

  bleEventSource.onerror = function(err) {
    console.error('BLE SSE error', err);
    bleSSEErrorCount++;
    if (bleSSEErrorCount >= 3) {
      console.warn('BLE SSE failed 3+ times, falling back to polling');
      if (bleEventSource) {
        bleEventSource.close();
        bleEventSource = null;
      }
      if (!blePollingInterval) {
        blePollingInterval = setInterval(pollBLE, 1000);
      }
    } else {
      bleEventSource.close();
      setTimeout(connectBLESSE, 5000);
    }
  };
}

async function pollBLE() {
  const data = await fetchJson('/api/ble');
  if (data) {
    updateBLEUI(data);
  }
}

function updateBLEUI(data) {
  if (!data || typeof data !== 'object') return;

  const neighborsCount = data.neighbors_count || 0;
  const selectedMAC = data.selected_device_mac || null;
  const selectedName = data.selected_device_name || null;
  const packetStats = data.packet_stats || { total_sent: 0, total_received: 0, failed_sends: 0 };
  const lastSent = data.last_packet_sent;
  const lastReceived = data.last_packet_received;

  const statusBadge = document.getElementById('bleConnectionStatus');
  if (statusBadge) {
    if (selectedMAC) {
      statusBadge.textContent = 'CONNECTED';
      statusBadge.className = 'badge bg-success fs-6 mb-2';
    } else if (neighborsCount > 0) {
      statusBadge.textContent = 'SCANNING';
      statusBadge.className = 'badge bg-warning fs-6 mb-2';
    } else {
      statusBadge.textContent = 'DISCONNECTED';
      statusBadge.className = 'badge bg-secondary fs-6 mb-2';
    }
  }

  const neighborsEl = document.getElementById('bleNeighborsCount');
  if (neighborsEl) {
    neighborsEl.textContent = neighborsCount;
    neighborsEl.style.transition = 'color 0.3s ease-in-out';
    neighborsEl.style.color = neighborsCount > 0 ? '#4CAF50' : '#999';
  }

  const deviceEl = document.getElementById('bleSelectedDevice');
  if (deviceEl) {
    deviceEl.textContent = selectedName || 'None';
  }
  const macEl = document.getElementById('bleSelectedMAC');
  if (macEl) {
    macEl.textContent = selectedMAC || '-';
  }

  if (bleNeighborsChart) {
    const ts = new Date().toLocaleTimeString();
    bleNeighborsChart.data.labels.push(ts);
    bleNeighborsChart.data.datasets[0].data.push(neighborsCount);

    if (bleNeighborsChart.data.labels.length > 60) {
      bleNeighborsChart.data.labels.shift();
      bleNeighborsChart.data.datasets[0].data.shift();
    }
    bleNeighborsChart.update('none');
  }

  if (blePacketStatsGauge) {
    const total = (packetStats.total_sent || 0) + (packetStats.total_received || 0);
    const sent = packetStats.total_sent || 0;
    const received = packetStats.total_received || 0;
    const remaining = Math.max(0, 100 - total);
    blePacketStatsGauge.data.datasets[0].data = [sent, received, remaining];
    blePacketStatsGauge.update('none');
  }

  const sentEl = document.getElementById('blePacketsSent');
  if (sentEl) sentEl.textContent = packetStats.total_sent || 0;
  const receivedEl = document.getElementById('blePacketsReceived');
  if (receivedEl) receivedEl.textContent = packetStats.total_received || 0;
  const failedEl = document.getElementById('blePacketsFailed');
  if (failedEl) failedEl.textContent = packetStats.failed_sends || 0;

  updateBLEPacketHistory(lastSent, lastReceived);
}

function updateBLEPacketHistory(lastSent, lastReceived) {
  const tbody = document.getElementById('blePacketHistoryBody');
  if (!tbody) return;

  const maxRows = 10;
  if (lastSent) {
    const row = createPacketRow(lastSent, 'SENT');
    tbody.insertBefore(row, tbody.firstChild);
  }
  if (lastReceived) {
    const row = createPacketRow(lastReceived, 'RECEIVED');
    tbody.insertBefore(row, tbody.firstChild);
  }
  while (tbody.children.length > maxRows) {
    tbody.removeChild(tbody.lastChild);
  }
}

function createPacketRow(packet, type) {
  const row = document.createElement('tr');
  const ts = packet && packet.timestamp ? new Date(packet.timestamp * 1000).toLocaleTimeString() : '-';
  const vid = packet && (packet.vid || packet.vehicle_hash) ? (packet.vid || packet.vehicle_hash).substring(0, 8) : '-';
  const fatigue = packet && (packet.fatigue_pct || packet.fatigue || packet.fatigue_score) ? (packet.fatigue_pct || packet.fatigue || packet.fatigue_score) : '-';
  const risk = packet && (packet.risk_pct || packet.risk) ? (packet.risk_pct || packet.risk) : '-';

  row.innerHTML = `
    <td>${ts}</td>
    <td><span class="badge ${type === 'SENT' ? 'bg-success' : 'bg-info'}">${type}</span></td>
    <td>${vid}</td>
    <td>${fatigue}%</td>
    <td>${risk}%</td>
  `;
  row.style.animation = 'fadeIn 0.3s ease-in-out';
  return row;
}

async function sendTestPacket() {
  const btn = document.getElementById('bleSendTestBtn');
  if (!btn) return;

  btn.disabled = true;
  btn.textContent = 'Sending...';

  try {
    const response = await fetch('/api/ble/send_test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ num_packets: 1 })
    });

    if (response.ok) {
      await response.json();
      btn.className = 'btn btn-success btn-sm mt-2 w-100';
      btn.textContent = 'Sent!';
      setTimeout(() => {
        btn.className = 'btn btn-primary btn-sm mt-2 w-100';
        btn.textContent = 'Send Test Packet';
        btn.disabled = false;
      }, 2000);
    } else {
      throw new Error('Send failed');
    }
  } catch (err) {
    console.error('Failed to send test packet', err);
    btn.className = 'btn btn-danger btn-sm mt-2 w-100';
    btn.textContent = 'Failed';
    setTimeout(() => {
      btn.className = 'btn btn-primary btn-sm mt-2 w-100';
      btn.textContent = 'Send Test Packet';
      btn.disabled = false;
    }, 2000);
  }
}

async function pollLatest() {
  const data = await fetchJson('/api/realtime/latest');
  if (!data) return;
  const ts = new Date().toLocaleTimeString();
  const score = data.risk_score ?? null;
  if (score !== null && riskChart) {
    riskChart.data.labels.push(ts);
    riskChart.data.datasets[0].data.push(score);
    if (riskChart.data.labels.length > 60) {
      riskChart.data.labels.shift();
      riskChart.data.datasets[0].data.shift();
    }
    riskChart.update('none');
    const el = document.getElementById('riskValue');
    if (el) el.innerText = (score*100).toFixed(0) + '%';
  }
  const alerts = await fetchJson('/api/alerts?limit=20');
  if (alerts && Array.isArray(alerts)) {
    const el = document.getElementById('alertsList');
    if (el) el.innerHTML = alerts.map(a=>`<div>${a.timestamp || '-'} - ${a.alert_level} - ${a.risk_score}</div>`).join('');
  }
}

// --- Incidents ---
async function loadIncidents() {
  const tbl = document.querySelector('#incidentsTable tbody');
  if (!tbl) return;
  tbl.innerHTML = '<tr><td colspan="6">Loading...</td></tr>';
  let data = await fetchJson('/api/incidents');
  if (!Array.isArray(data)) data = [];
  tbl.innerHTML = data.map(i=>`<tr data-id="${i.incident_id}"><td>${i.incident_id}</td><td>${i.timestamp || ''}</td><td>${(i.risk_score||'').toFixed? (i.risk_score.toFixed(2)): (i.risk_score||'')}</td><td>${i.alert_level||''}</td><td>${i.has_video? 'Yes':'No'}</td><td>${i.annotation_count||0}</td></tr>`).join('');
  tbl.querySelectorAll('tr[data-id]').forEach(r=>r.addEventListener('click', ()=>openIncidentModal(r.getAttribute('data-id'))));
}

async function openIncidentModal(incidentId) {
  const modalEl = document.getElementById('incidentModal');
  const modal = new bootstrap.Modal(modalEl);
  document.getElementById('incidentModalLabel').innerText = `Incident ${incidentId}`;
  const video = document.getElementById('incidentVideo');
  const timelineCanvas = document.getElementById('incidentTimeline');
  video.pause(); video.src = '';
  if (incidentTimelineChart) { incidentTimelineChart.destroy(); incidentTimelineChart = null; }

  const framesResp = await fetchJson(`/api/incidents/${incidentId}/frames`);
  const frames = framesResp && framesResp.frames ? framesResp.frames : [];
  if (frames.length && frames[0].url) {
    video.src = frames[0].url;
  }

  const timeline = await fetchJson(`/api/incidents/${incidentId}/timeline`);
  const times = timeline && timeline.timestamps ? timeline.timestamps : [];
  const datasets = [];
  if (timeline && timeline.series) {
    Object.keys(timeline.series).forEach((k, idx)=>{
      datasets.push({label:k,data:timeline.series[k],borderColor: `hsl(${(idx*60)%360} 70% 50%)`,fill:false});
    });
  }
  incidentTimelineChart = new Chart(timelineCanvas.getContext('2d'), {type:'line', data:{labels:times, datasets:datasets}, options:{animation:false}});

  await refreshAnnotationsList(incidentId);

  const form = document.getElementById('annotationForm');
  form.onsubmit = async (e)=>{
    e.preventDefault();
    const label = form.querySelector('input[name="label"]:checked').value;
    const severity = document.getElementById('severitySelect').value;
    const notes = document.getElementById('annotationNotes').value;
    const reviewer = document.getElementById('reviewerId').value || 'anonymous';
    const body = {incident_id: incidentId, label, notes, reviewer_id: reviewer, severity_override: severity};
    const res = await fetch(`/api/annotations`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    if (res.ok) {
      await refreshAnnotationsList(incidentId);
      form.reset();
    } else {
      alert('Failed to submit annotation');
    }
  };

  modal.show();
}

async function refreshAnnotationsList(incidentId) {
  const list = document.getElementById('annotationList');
  list.innerHTML = 'Loading annotations...';
  let anns = await fetchJson(`/api/annotations?incident_id=${encodeURIComponent(incidentId)}`);
  if (!Array.isArray(anns)) anns = [];
  list.innerHTML = anns.map(a=>`<div class="border p-2 mb-1"><strong>${a.label}</strong> by ${a.reviewer_id} @ ${new Date(a.created_at*1000).toLocaleString()}<div>${a.notes||''}</div></div>`).join('');
}

// --- Analytics ---
async function initAnalytics() {
  const fr = await fetchJson('/api/analytics/false_alarm_rate');
  if (fr) {
    const el = document.getElementById('card-false-rate'); if (el) el.innerText = fr.rate==null? 'N/A' : (fr.rate*100).toFixed(1)+'%';
    const ctx = document.getElementById('chart-false-rate').getContext('2d');
    const labels = fr.time_series ? fr.time_series.map(x=>x.date) : [];
    const data = fr.time_series ? fr.time_series.map(x=>x.rate) : [];
    analyticsCharts.falseRate = new Chart(ctx, {type:'line', data:{labels, datasets:[{label:'False Alarm Rate',data}]}});
  }
  const lat = await fetchJson('/api/analytics/detection_latency');
  if (lat) {
    const el = document.getElementById('card-latency-p95'); if (el) el.innerText = lat.p95 || 'N/A';
    const ctx = document.getElementById('chart-latency').getContext('2d');
    analyticsCharts.latency = new Chart(ctx, {type:'bar', data:{labels:['p95','p99','mean'], datasets:[{label:'ms',data:[lat.p95||0, lat.p99||0, lat.mean||0]}]}});
  }
  const fi = await fetchJson('/api/analytics/feature_importance');
  if (fi) {
    const ctx = document.getElementById('chart-feature-importance').getContext('2d');
    analyticsCharts.feature = new Chart(ctx, {type:'bar', data:{labels:fi.top_features.map(f=>f.feature), datasets:[{label:'mean_abs_shap', data:fi.top_features.map(f=>f.mean_abs_shap)}]}});
  }
}

// Tab handling
document.addEventListener('DOMContentLoaded', ()=>{
  initRiskChart();
  initFatigueGauge();
  initFatigueMetricsChart();
  initBLENeighborsChart();
  initBLEPacketStatsGauge();

  pollLatest();
  setInterval(pollLatest, pollIntervalMs);

  try {
    connectFatigueSSE();
  } catch (err) {
    console.warn('SSE not available, using polling', err);
    setInterval(pollFatigue, 1000);
  }

  try {
    connectBLESSE();
  } catch (err) {
    console.warn('BLE SSE not available, using polling', err);
    setInterval(pollBLE, 1000);
  }

  document.getElementById('incidents-tab').addEventListener('click', ()=>{ loadIncidents(); });
  document.getElementById('analytics-tab').addEventListener('click', ()=>{ initAnalytics(); });

  const sendBtn = document.getElementById('bleSendTestBtn');
  if (sendBtn) {
    sendBtn.addEventListener('click', sendTestPacket);
  }
});

window.addEventListener('beforeunload', () => {
  if (fatigueEventSource) fatigueEventSource.close();
  if (bleEventSource) bleEventSource.close();
});
