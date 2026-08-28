const http = require("node:http");
const token = process.env.HF_TOKEN;
const model = process.env.HF_MODEL || "jacoballessio/ai-image-detect-distilled";
const server = http.createServer(async (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*"); res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") return res.end();
  if (req.method !== "POST" || req.url !== "/detect") return send(res, 404, { error: "Not found" });
  if (!token) return send(res, 500, { error: "HF_TOKEN is not set" });
  try {
    const body = await readJson(req); const image = Buffer.from(body.image.split(",")[1], "base64");
    const response = await fetch(`https://router.huggingface.co/hf-inference/models/${model}`, { method: "POST", headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/octet-stream" }, body: image });
    const data = await response.json(); if (!response.ok) return send(res, response.status, { error: data.error || "Hugging Face request failed" });
    const top = (Array.isArray(data) ? data : []).sort((a,b) => b.score-a.score)[0]; const ai = /ai|fake|generated|synthetic/i.test(top?.label || "");
    send(res, 200, { verdict: ai ? "ai-generated" : "not-ai", confidence: Math.round((top?.score || 0)*100), note: `${top?.label || "Unknown"} classification from Hugging Face` });
  } catch (error) { send(res, 500, { error: error.message }); }
});
function readJson(req) { return new Promise((resolve,reject)=>{ let data=""; req.on("data",c=>data+=c); req.on("end",()=>resolve(JSON.parse(data))); req.on("error",reject); }); }
function send(res,status,value) { res.writeHead(status,{"Content-Type":"application/json"}); res.end(JSON.stringify(value)); }
server.listen(3000,()=>console.log(`Hugging Face detector: http://localhost:3000 using ${model}`));
