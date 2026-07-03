/* ════════════════════════════════════════════════════════════════
   canvas.js  –  Drawing canvas + prediction logic for index.html
   ════════════════════════════════════════════════════════════════ */

// ── State ────────────────────────────────────────────────────────────────────
const canvas    = document.getElementById("drawingCanvas");
const ctx       = canvas.getContext("2d");
let   painting  = false;
let   snapshots = [];           // undo stack
let   currentLogId = null;
let   activeTab = "draw";

// ── Canvas initialise ─────────────────────────────────────────────────────────
(function initCanvas() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineCap    = "round";
  ctx.lineJoin   = "round";
  ctx.strokeStyle = "#000000";
  ctx.lineWidth   = 16;

  // Mouse
  canvas.addEventListener("mousedown",  startPainting);
  canvas.addEventListener("mousemove",  draw);
  canvas.addEventListener("mouseup",    stopPainting);
  canvas.addEventListener("mouseleave", stopPainting);

  // Touch
  canvas.addEventListener("touchstart",  e => { e.preventDefault(); startPainting(e.touches[0]); }, { passive:false });
  canvas.addEventListener("touchmove",   e => { e.preventDefault(); draw(e.touches[0]); },           { passive:false });
  canvas.addEventListener("touchend",    () => stopPainting());

  // Brush size
  document.getElementById("brushSize").addEventListener("input", e => {
    ctx.lineWidth = parseInt(e.target.value);
  });
})();

function getPos(evt) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width  / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (evt.clientX - rect.left) * scaleX,
    y: (evt.clientY - rect.top)  * scaleY,
  };
}

function startPainting(evt) {
  snapshots.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
  painting = true;
  const { x, y } = getPos(evt);
  ctx.beginPath();
  ctx.moveTo(x, y);
}

function draw(evt) {
  if (!painting) return;
  const { x, y } = getPos(evt);
  ctx.lineTo(x, y);
  ctx.stroke();
}

function stopPainting() { painting = false; ctx.beginPath(); }

function clearCanvas() {
  snapshots = [];
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  resetResult();
}

function undoCanvas() {
  if (!snapshots.length) return;
  ctx.putImageData(snapshots.pop(), 0, 0);
}


// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(tab) {
  activeTab = tab;
  document.getElementById("panel-draw").classList.toggle("d-none",  tab !== "draw");
  document.getElementById("panel-upload").classList.toggle("d-none", tab !== "upload");
  document.getElementById("tab-draw").classList.toggle("active",    tab === "draw");
  document.getElementById("tab-upload").classList.toggle("active",  tab === "upload");
  resetResult();
}


// ── File upload ────────────────────────────────────────────────────────────────
let uploadedFile = null;

function handleFileSelect(evt) {
  const file = evt.target.files[0];
  if (file) previewFile(file);
}

function handleDrop(evt) {
  evt.preventDefault();
  dragLeave(evt);
  const file = evt.dataTransfer.files[0];
  if (file) previewFile(file);
}

function dragEnter(evt) {
  evt.preventDefault();
  document.getElementById("dropZone").classList.add("dragover");
}
function dragLeave() {
  document.getElementById("dropZone").classList.remove("dragover");
}

function previewFile(file) {
  if (!file.type.startsWith("image/")) {
    alert("Please upload an image file.");
    return;
  }
  uploadedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    const preview = document.getElementById("uploadPreview");
    preview.src = e.target.result;
    preview.classList.remove("d-none");
  };
  reader.readAsDataURL(file);
}


// ── Prediction ────────────────────────────────────────────────────────────────
async function predict() {
  const model       = document.getElementById("modelSelect").value;
  const generateXai = document.getElementById("xaiToggle").checked;

  showLoading(true);
  resetResult();

  try {
    let result;

    if (activeTab === "draw") {
      const imageData = canvas.toDataURL("image/png");
      result = await fetch("/api/predict/canvas", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_data: imageData, model_name: model,
                               generate_xai: generateXai }),
      }).then(r => r.json());

    } else {
      if (!uploadedFile) { alert("Please select an image first."); showLoading(false); return; }
      const formData = new FormData();
      formData.append("file",         uploadedFile);
      formData.append("model_name",   model);
      formData.append("generate_xai", generateXai);
      result = await fetch("/api/predict/upload", {
        method: "POST",
        body: formData,
      }).then(r => r.json());
    }

    if (result.detail) throw new Error(result.detail);

    currentLogId = result.log_id;
    renderResult(result);

  } catch (err) {
    alert("Prediction failed: " + err.message);
    console.error(err);
  } finally {
    showLoading(false);
  }
}


// ── Render result ─────────────────────────────────────────────────────────────
function renderResult(r) {
  document.getElementById("emptyState").classList.add("d-none");
  document.getElementById("resultCard").classList.remove("d-none");

  document.getElementById("resultDigit").textContent = r.predicted_digit;
  document.getElementById("confValue").textContent   = r.confidence.toFixed(2) + "%";
  document.getElementById("procTime").textContent    = r.processing_time_ms.toFixed(1) + " ms";
  document.getElementById("modelUsed").textContent   = r.model_used;

  // Confidence bar
  const bar = document.getElementById("confBar");
  bar.style.width = r.confidence + "%";
  bar.className = "progress-bar progress-bar-striped progress-bar-animated " +
    (r.confidence >= 90 ? "bg-success" : r.confidence >= 70 ? "bg-warning" : "bg-danger");

  // Top-3
  const top3 = document.getElementById("top3Container");
  top3.innerHTML = (r.top3_predictions || []).map((t, i) =>
    `<div class="text-center">
       <div style="font-size:${i===0?'2rem':'1.3rem'};font-weight:700;color:${i===0?'#1565C0':'#555'}">${t.digit}</div>
       <div class="small text-muted">${t.confidence.toFixed(1)}%</div>
     </div>`
  ).join("");

  // All probability bars
  const probBars = document.getElementById("probBars");
  const probs = r.all_probabilities || [];
  const maxProb = Math.max(...probs);
  probBars.innerHTML = probs.map((p, i) =>
    `<div class="prob-row">
       <span class="prob-label">${i}</span>
       <div class="prob-bar-wrap">
         <div class="prob-bar-fill ${p === maxProb ? 'top' : ''}"
              style="width:${Math.max(p, 0.3)}%"></div>
       </div>
       <span class="prob-value">${p.toFixed(1)}%</span>
     </div>`
  ).join("");

  // XAI heatmaps
  const xaiCard = document.getElementById("xaiCard");
  const xaiImgs = document.getElementById("xaiImages");
  const heatmaps = [
    { label: "Grad-CAM",       path: r.gradcam_path  },
    { label: "Saliency Map",   path: r.saliency_path },
  ];
  const available = heatmaps.filter(h => h.path);
  if (available.length) {
    xaiCard.classList.remove("d-none");
    xaiImgs.innerHTML = available.map(h =>
      `<div class="col-md-6 text-center">
         <p class="small fw-semibold mb-1">${h.label}</p>
         <img src="/${h.path}" class="img-fluid rounded shadow-sm" alt="${h.label}"
              style="max-height:200px;object-fit:contain">
       </div>`
    ).join("");
  }
}

function resetResult() {
  document.getElementById("resultCard").classList.add("d-none");
  document.getElementById("xaiCard").classList.add("d-none");
  document.getElementById("emptyState").classList.remove("d-none");
  document.getElementById("feedbackMsg").classList.add("d-none");
  document.getElementById("correctionGroup").classList.add("d-none");
  currentLogId = null;
}


// ── Feedback ──────────────────────────────────────────────────────────────────
function showCorrectionInput() {
  document.getElementById("correctionGroup").classList.toggle("d-none");
}

async function sendFeedback(isCorrect) {
  if (!currentLogId) return;
  const trueLabel = isCorrect
    ? parseInt(document.getElementById("resultDigit").textContent)
    : parseInt(document.getElementById("trueLabel").value);
  if (!isCorrect && (isNaN(trueLabel) || trueLabel < 0 || trueLabel > 9)) {
    alert("Enter a digit 0–9."); return;
  }
  await fetch("/api/predict/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ log_id: currentLogId, true_label: trueLabel,
                           is_correct: isCorrect }),
  });
  document.getElementById("feedbackMsg").classList.remove("d-none");
  document.getElementById("correctionGroup").classList.add("d-none");
}


// ── Loading overlay ────────────────────────────────────────────────────────────
function showLoading(show) {
  document.getElementById("loadingOverlay").classList.toggle("d-none", !show);
  document.getElementById("predictBtn").disabled = show;
}
