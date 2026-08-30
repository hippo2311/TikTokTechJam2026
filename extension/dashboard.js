const API = "http://34.124.152.42:8000";
const $ = id => document.getElementById(id);
async function refresh() {
  try {
    const data = await (await fetch(`${API}/stats`)).json();
    $("accuracy").textContent = `${data.accuracy}%`;
    $("total").textContent = data.total;
    $("correct").textContent = data.correct;
    $("wrong").textContent = data.wrong;
    $("updated").textContent = `Updated ${new Date().toLocaleTimeString()}`;
    const m = data.confusionMatrix || {};
    $("tp").textContent = m.tp || 0; $("fp").textContent = m.fp || 0;
    $("fn").textContent = m.fn || 0; $("tn").textContent = m.tn || 0;
    renderCases(data.wrongCases || []);
  } catch (e) { $("updated").textContent = "Backend offline"; }
}
function renderCases(rows) { $("cases").innerHTML = rows.length ? rows.map(r => `<div class="case"><span class="badge">WRONG</span>Predicted: <b>${r.prediction || "unknown"}</b> → Correct: <b>${r.label}</b><br><small>${r.page || "local capture"}</small></div>`).join("") : '<div class="empty">No wrong detections yet.</div>'; }
refresh(); setInterval(refresh, 3000);
