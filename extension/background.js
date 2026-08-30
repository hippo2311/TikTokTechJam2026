const CAPTURE_MESSAGE = "AI_IMAGE_CHECK_CAPTURE";
const API = "http://34.124.152.42:8000";

chrome.commands.onCommand.addListener(async (command) => {
  if (command !== "capture-and-check") return;
  await captureAndOpenResult();
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === "AI_IMAGE_CHECK_FEEDBACK") {
    chrome.storage.local.get(["lastCapture", "lastResult"]).then(async ({ lastCapture, lastResult }) => {
      const response = await fetch(`${API}/feedback`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ feedback: message.feedback, label: message.label, prediction: lastResult?.verdict, predictionId: lastResult?.id, image: lastCapture, page: sender.tab?.url, createdAt: new Date().toISOString() }) });
      if (response.ok) await chrome.storage.local.set({ pendingFeedback: false });
      sendResponse({ ok: response.ok });
    }).catch(error => sendResponse({ ok: false, error: error.message }));
    return true;
  }
  if (message?.type !== CAPTURE_MESSAGE) return;
  captureAndOpenResult(sender.tab?.id, message.rect).then(() => sendResponse({ ok: true })).catch((error) => {
    sendResponse({ ok: false, error: error.message });
  });
  return true;
});

async function captureAndOpenResult(requestingTabId, rect) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id || tab.url?.startsWith("about:") || tab.url?.startsWith("chrome:")) {
    throw new Error("This page does not allow screen capture.");
  }

  await autoConfirmPendingFeedback(tab.url);

  let dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, { format: "png" });
  if (rect) dataUrl = await cropDataUrl(dataUrl, rect);
  await chrome.storage.local.set({ lastCapture: dataUrl, capturedAt: Date.now() });

  let result;
  try {
    const detectorResponse = await fetch(`${API}/detect`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image: dataUrl }) });
    if (!detectorResponse.ok) throw new Error(await detectorResponse.text());
    result = await detectorResponse.json();
  } catch (error) {
    result = { verdict: "unavailable", confidence: null, note: "Detector is not running: " + error.message };
  }
  await chrome.storage.local.set({ lastResult: result, pendingFeedback: true });
  await chrome.tabs.sendMessage(requestingTabId || tab.id, { type: "SHOW_RESULT", result }).catch(() => {});
}

async function autoConfirmPendingFeedback(page) {
  const { pendingFeedback, lastCapture, lastResult } = await chrome.storage.local.get(["pendingFeedback", "lastCapture", "lastResult"]);
  if (!pendingFeedback || !lastResult) return;
  const response = await fetch(`${API}/feedback`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ feedback: "correct", label: lastResult.verdict, prediction: lastResult.verdict, image: lastCapture, page, createdAt: new Date().toISOString() }) });
  if (response.ok) await chrome.storage.local.set({ pendingFeedback: false });
}

async function cropDataUrl(dataUrl, rect) {
  const image = await createImageBitmap(await (await fetch(dataUrl)).blob());
  const sx=image.width/rect.viewportWidth, sy=image.height/rect.viewportHeight;
  const canvas=new OffscreenCanvas(Math.max(1,rect.width*sx),Math.max(1,rect.height*sy));
  canvas.getContext("2d").drawImage(image,rect.x*sx,rect.y*sy,rect.width*sx,rect.height*sy,0,0,canvas.width,canvas.height);
  const blob=await canvas.convertToBlob({type:"image/png"});let binary="";for(const byte of new Uint8Array(await blob.arrayBuffer()))binary+=String.fromCharCode(byte);return `data:image/png;base64,${btoa(binary)}`;
}
