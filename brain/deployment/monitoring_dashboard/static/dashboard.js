async function fetchJson(url) {
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error('http ' + r.status);
    return await r.json();
  } catch (e) {
    console.warn('fetch failed', url, e);
    return null;
  }
}

async function refresh() {
  const sys = await fetchJson('/api/system_stats');
  if (sys) {
    document.getElementById('cpu').innerText = (sys.cpu_percent || 0) + '%';
    document.getElementById('memory').innerText = Math.round((sys.memory.used||0)/1024/1024) + ' MB';
    document.getElementById('disk').innerText = Math.round((sys.disk.used||0)/1024/1024) + ' MB';
    document.getElementById('uptime').innerText = Math.round((sys.uptime_s||0)/60) + ' min';
  }
  const inf = await fetchJson('/api/inference_stats');
  if (inf) {
    const el = document.getElementById('inference');
    if (el) el.innerText = `inferences: ${inf.total_inferences} mean: ${inf.latency_ms.mean || '-'} ms`;
  }
  const sh = await fetchJson('/api/sensor_health');
  if (sh) {
    const el = document.getElementById('sensor_health');
    if (el) el.innerText = JSON.stringify(sh);
  }
}

setInterval(refresh, 2000);
refresh();
