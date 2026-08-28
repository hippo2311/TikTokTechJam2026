const button = document.querySelector("#capture");
const status = document.querySelector("#status");
button.addEventListener("click", async () => {
  button.disabled = true;
  button.textContent = "Capturing…";
  try {
    const response = await chrome.runtime.sendMessage({ type: "AI_IMAGE_CHECK_CAPTURE" });
    if (!response?.ok) throw new Error(response?.error || "Unable to capture image");
    status.innerHTML = '<div class="status-icon">✓</div><div><strong>Image captured</strong><small>Ready for AI analysis</small></div>';
  } catch (error) {
    status.innerHTML = `<div class="status-icon">!</div><div><strong>Something went wrong</strong><small>${error.message}</small></div>`;
  } finally { button.disabled = false; button.innerHTML = "<span>⌘</span> Capture & check"; }
});
