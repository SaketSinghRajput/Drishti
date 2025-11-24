// Dashboard JS: real-time polling, incidents, modal, annotations, analytics
const DEFAULT_POLL_MS = 1000;
let pollIntervalMs = DEFAULT_POLL_MS;
let riskChart = null;
let incidentTimelineChart = null;
let analyticsCharts = {};

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
  // update alerts
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
  const data = await fetchJson('/api/incidents');
  if (!Array.isArray(data)) data = [];
  tbl.innerHTML = data.map(i=>`<tr data-id="${i.incident_id}"><td>${i.incident_id}</td><td>${i.timestamp || ''}</td><td>${(i.risk_score||'').toFixed? (i.risk_score.toFixed(2)): (i.risk_score||'')}</td><td>${i.alert_level||''}</td><td>${i.has_video? 'Yes':'No'}</td><td>${i.annotation_count||0}</td></tr>`).join('');
  // attach click handlers
  tbl.querySelectorAll('tr[data-id]').forEach(r=>r.addEventListener('click', ()=>openIncidentModal(r.getAttribute('data-id'))));
}

async function openIncidentModal(incidentId) {
  const modalEl = document.getElementById('incidentModal');
  const modal = new bootstrap.Modal(modalEl);
  document.getElementById('incidentModalLabel').innerText = `Incident ${incidentId}`;
  const video = document.getElementById('incidentVideo');
  const timelineCanvas = document.getElementById('incidentTimeline');
  // clear previous
  video.pause(); video.src = '';
  if (incidentTimelineChart) { incidentTimelineChart.destroy(); incidentTimelineChart = null; }

  // fetch frames
  const framesResp = await fetchJson(`/api/incidents/${incidentId}/frames`);
  const frames = framesResp && framesResp.frames ? framesResp.frames : [];
  if (frames.length && frames[0].url) {
    // build a small blob playlist by using first frame as poster (for simplicity)
    video.src = frames[0].url;
  }

  // fetch timeline
  const timeline = await fetchJson(`/api/incidents/${incidentId}/timeline`);
  const times = timeline && timeline.timestamps ? timeline.timestamps : [];
  const datasets = [];
  if (timeline && timeline.series) {
    Object.keys(timeline.series).forEach((k, idx)=>{
      datasets.push({label:k,data:timeline.series[k],borderColor: `hsl(${(idx*60)%360} 70% 50%)`,fill:false});
    });
  }
  incidentTimelineChart = new Chart(timelineCanvas.getContext('2d'), {type:'line', data:{labels:times, datasets:datasets}, options:{animation:false}});

  // load annotations
  await refreshAnnotationsList(incidentId);

  // annotation submit
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
  const anns = await fetchJson(`/api/annotations?incident_id=${encodeURIComponent(incidentId)}`);
  if (!Array.isArray(anns)) anns = [];
  list.innerHTML = anns.map(a=>`<div class="border p-2 mb-1"><strong>${a.label}</strong> by ${a.reviewer_id} @ ${new Date(a.created_at*1000).toLocaleString()}<div>${a.notes||''}</div></div>`).join('');
}

// --- Analytics ---
async function initAnalytics() {
  // false alarm rate chart
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
  pollLatest();
  setInterval(pollLatest, pollIntervalMs);

  // tab events
  document.getElementById('incidents-tab').addEventListener('click', ()=>{ loadIncidents(); });
  document.getElementById('analytics-tab').addEventListener('click', ()=>{ initAnalytics(); });
});

