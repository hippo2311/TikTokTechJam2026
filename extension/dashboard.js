const API = "http://34.124.152.42:8000";
const $ = id => document.getElementById(id);
async function refresh() {
  try {
    const data = await (await fetch(`${API}/stats`)).json();
    $("accuracy").textContent = `${data.accuracy}%`;
    $("signalValue").textContent = `${data.accuracy}%`;
    $("total").textContent = data.total;
    $("correct").textContent = data.correct;
    $("wrong").textContent = data.wrong;
    $("updated").textContent = `Updated ${new Date().toLocaleTimeString()}`;
    const m = data.confusionMatrix || {};
    $("tp").textContent = m.tp || 0; $("fp").textContent = m.fp || 0;
    $("fn").textContent = m.fn || 0; $("tn").textContent = m.tn || 0;
    renderLineChart(data.history || []);
    renderCases(data.wrongCases || []);
  } catch (e) { $("updated").textContent = "Backend offline"; }
}
function renderCases(rows) { $("cases").innerHTML = rows.length ? rows.map(r => `<div class="case"><span class="badge">WRONG</span>Predicted: <b>${r.prediction || "unknown"}</b> → Correct: <b>${r.label}</b><br><small>${r.page || "local capture"}</small></div>`).join("") : '<div class="empty">No wrong detections yet.</div>'; }
function renderLineChart(rows) {
  const el = $("lineChart"); if (!rows.length) { el.innerHTML = '<div class="empty">No reviewed data yet.</div>'; return; }
  const w = 900, h = 210, pad = 18, points = rows.map((r, i) => `${pad + i * ((w - pad * 2) / Math.max(1, rows.length - 1))},${h - pad - (r.accuracy / 100) * (h - pad * 2)}`).join(" ");
  el.innerHTML = `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><polyline points="${points}"/>${rows.map((r,i)=>{const x=pad+i*((w-pad*2)/Math.max(1,rows.length-1));const y=h-pad-(r.accuracy/100)*(h-pad*2);return `<circle cx="${x}" cy="${y}" r="5"><title>${r.date}: ${r.accuracy}%</title></circle>`}).join("")}</svg>`;
}
refresh(); setInterval(refresh, 3000);
