(() => {
  if (document.getElementById("ai-image-check-bubble")) return;
  const bubble = document.createElement("button"); bubble.id="ai-image-check-bubble"; bubble.type="button"; bubble.title="Select and check image"; bubble.textContent="✦";
  const panel=document.createElement("div"); panel.id="ai-image-check-panel"; panel.innerHTML='<strong>AI Image Check</strong><span>Click to select an image area</span><div class="ai-feedback"><button data-feedback="correct">Correct</button><button data-feedback="wrong">Wrong</button></div><div class="ai-labels"><button data-label="ai-generated">AI-generated</button><button data-label="not-ai">Human-made</button></div>';
  const shade=document.createElement("div"); shade.id="ai-image-check-shade"; const selection=document.createElement("div"); selection.id="ai-image-check-selection"; document.documentElement.append(bubble,panel,shade,selection);
  let sx,sy,dragging=false;
  bubble.onclick=()=>{shade.classList.add("is-active");panel.querySelector("span").textContent="Drag to select an image area";panel.classList.add("is-visible");};
  shade.onmousedown=e=>{dragging=true;sx=e.clientX;sy=e.clientY;selection.classList.add("is-active");};
  shade.onmousemove=e=>{if(!dragging)return;Object.assign(selection.style,{left:Math.min(sx,e.clientX)+"px",top:Math.min(sy,e.clientY)+"px",width:Math.abs(e.clientX-sx)+"px",height:Math.abs(e.clientY-sy)+"px"});};
  shade.onmouseup=async e=>{if(!dragging)return;dragging=false;shade.classList.remove("is-active");selection.classList.remove("is-active");const rect={x:Math.min(sx,e.clientX),y:Math.min(sy,e.clientY),width:Math.abs(e.clientX-sx),height:Math.abs(e.clientY-sy),viewportWidth:innerWidth,viewportHeight:innerHeight};if(rect.width<12||rect.height<12)return;bubble.classList.add("is-loading");panel.querySelector("span").textContent="Capturing selected area…";try{const r=await chrome.runtime.sendMessage({type:"AI_IMAGE_CHECK_CAPTURE",rect});if(!r?.ok)throw Error(r?.error||"Unable to capture area");}catch(err){panel.querySelector("span").textContent=err.message;bubble.classList.remove("is-loading");}};
  panel.addEventListener("click", async e => { const button=e.target.closest("button"); if(!button)return; clearTimeout(autoConfirmTimer); const feedback=button.dataset.feedback; if(feedback==="correct"){await recordFeedback("correct",lastVerdict); showThanks();} if(feedback==="wrong"){panel.querySelector(".ai-labels").classList.add("is-visible");} if(button.dataset.label){await recordFeedback("wrong",button.dataset.label);showThanks();} });
  let lastVerdict="uncertain";
  let autoConfirmTimer;
  let feedbackRecorded=false;
  const AUTO_CONFIRM_DELAY=5000;
  function showThanks(){panel.querySelector("span").textContent="Thanks — feedback recorded";panel.querySelector(".ai-feedback").style.display="none";panel.querySelector(".ai-labels").classList.remove("is-visible");}
  async function recordFeedback(type,label){if(feedbackRecorded)return;feedbackRecorded=true;try{await chrome.runtime.sendMessage({type:"AI_IMAGE_CHECK_FEEDBACK",feedback:type,label});}catch(_){} }
  chrome.runtime.onMessage.addListener(m=>{if(m?.type!=="SHOW_RESULT")return;bubble.classList.remove("is-loading");const r=m.result||{};lastVerdict=r.verdict||"uncertain";feedbackRecorded=false;clearTimeout(autoConfirmTimer);panel.querySelector("span").textContent=r.verdict==="ai-generated"?`Likely AI-generated (${r.confidence}%)`:r.verdict==="not-ai"?`Likely human-made (${r.confidence}%)`:(r.note||"Unable to determine");panel.querySelector(".ai-feedback").style.display="flex";panel.classList.add("is-visible");autoConfirmTimer=setTimeout(async()=>{await recordFeedback("correct",lastVerdict);showThanks();},AUTO_CONFIRM_DELAY);});
})();
