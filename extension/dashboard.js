const API = "http://34.124.152.42:8000";
const $ = id => document.getElementById(id);
let allWrong = [], page = Math.max(1, Number(sessionStorage.getItem("dashboardPage")) || 1);
function setPage(nextPage) {
  page = Math.max(1, Number(nextPage) || 1);
  sessionStorage.setItem("dashboardPage", String(page));
}
async function refresh(resetPage = false) {
  if (resetPage) setPage(1);
  try {
    const data = await (await fetch(`${API}/stats`)).json();
    $("accuracy").textContent = `${data.accuracy}%`;
    $("signalValue").textContent = `${data.accuracy}%`;
    document.documentElement.style.setProperty("--accuracy", `${data.accuracy}%`);
    const warning = $("warning");
    warning.textContent = data.total < 20 ? "Collecting more reviews" : data.accuracy < 70 ? "Critical accuracy — review needed" : data.accuracy < 85 ? "Accuracy below target" : "Accuracy looks healthy";
    warning.className = `warning ${data.total >= 20 && data.accuracy < 70 ? "critical" : data.total >= 20 && data.accuracy < 85 ? "caution" : "good"}`;
    $("total").textContent = data.total;
    $("correct").textContent = data.correct;
    $("wrong").textContent = data.wrong;
    $("updated").textContent = `Updated ${new Date().toLocaleTimeString()}`;
    const m = data.confusionMatrix || {};
    $("tp").textContent = m.tp || 0; $("fp").textContent = m.fp || 0;
    $("fn").textContent = m.fn || 0; $("tn").textContent = m.tn || 0;
    renderLineChart(data.history || []);
    allWrong = data.recentPredictions || [];
    renderCases();
  } catch (e) { $("updated").textContent = "Backend offline"; }
}
function renderCases() { const filter=$("statusFilter")?.value||"all", filtered=filter==="all"?allWrong:allWrong.filter(r=>r.status===filter), pageCount=Math.ceil(filtered.length/20); setPage(Math.min(page,Math.max(1,pageCount))); const start=(page-1)*20, rows=filtered.slice(start,start+20); $("cases").innerHTML=rows.length?rows.map(r=>`<div class="case"><span class="badge">${r.status?.toUpperCase()||"UNREVIEWED"}</span>Prediction: <b>${r.verdict||"unknown"}</b> · Actual: <b>${r.actual||"—"}</b> · Confidence: <b>${r.confidence ?? "—"}%</b><br><small>${new Date(r.createdAt).toLocaleString()}</small>${r.imageUrl?` · <a href="${API}${r.imageUrl}" target="_blank" rel="noopener" class="view-image">View image ↗</a>`:""} <select class="edit-status" data-id="${r.id}"><option value="">Edit status</option><option value="correct">Correct</option><option value="wrong">Wrong</option><option value="unreviewed">Unreviewed</option></select></div>`).join(""):'<div class="empty">No predictions found.</div>'; $("pager").innerHTML=Array.from({length:pageCount},(_,i)=>`<button class="${i+1===page?"active":""}" data-page="${i+1}">${i+1}</button>`).join(""); }
document.addEventListener("click",e=>{if(e.target.dataset.page){e.preventDefault();setPage(e.target.dataset.page);renderCases();}});
$("updateDb")?.addEventListener("click",()=>refresh(true));
$("statusFilter")?.addEventListener("change",()=>{setPage(1);renderCases();});
document.addEventListener("change",async e=>{if(!e.target.classList.contains("edit-status")||!e.target.value)return;await fetch(`${API}/predictions/${e.target.dataset.id}/review`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({status:e.target.value})});refresh();});
function renderLineChart(rows) {
  const el = $("lineChart"); if (!rows.length) { el.innerHTML = '<div class="empty">No reviewed data yet.</div>'; return; }
  const w = 900, h = 210, pad = 18, max = Math.max(1, ...rows.map(r => Math.max(r.correct, r.wrong))), x = i => pad + i * ((w - pad * 2) / Math.max(1, rows.length - 1)), y = v => h - pad - (v / max) * (h - pad * 2);
  const line = key => rows.map((r,i) => `${x(i)},${y(r[key] || 0)}`).join(" ");
  el.innerHTML = `<div class="chart-legend"><span><i class="cyan"></i>Correct</span><span><i class="pink"></i>Wrong</span></div><svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><polyline class="correct-line" points="${line("correct")}"/><polyline class="wrong-line" points="${line("wrong")}"/>${rows.map((r,i)=>`<circle class="correct-dot" cx="${x(i)}" cy="${y(r.correct||0)}" r="5"><title>${r.date}: ${r.correct||0} correct</title></circle><circle class="wrong-dot" cx="${x(i)}" cy="${y(r.wrong||0)}" r="5"><title>${r.date}: ${r.wrong||0} wrong</title></circle>`).join("")}</svg>`;
}
refresh(); setInterval(()=>refresh(false), 3000);
