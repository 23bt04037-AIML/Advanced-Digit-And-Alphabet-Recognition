/* ════════════════════════════════════════════════════════════════
   dashboard.js  –  Chart.js charts + API calls for dashboard.html
   ════════════════════════════════════════════════════════════════ */

// Chart registry for cleanup
const charts = {};

const PALETTE = [
  "#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
  "#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac"
];

function destroyChart(id) {
  if (charts[id]) { charts[id].destroy(); delete charts[id]; }
}

// ── KPI summary ───────────────────────────────────────────────────────────────
async function loadSummary() {
  const d = await fetch("/api/dashboard/summary").then(r => r.json());
  document.getElementById("kpi-total").textContent  = d.total ?? "–";
  document.getElementById("kpi-conf").textContent   = d.avg_confidence != null
    ? d.avg_confidence.toFixed(1) + "%" : "–";
  document.getElementById("kpi-speed").textContent  = d.avg_time_ms != null
    ? d.avg_time_ms.toFixed(1) + " ms" : "–";
  document.getElementById("kpi-digit").textContent  = d.most_common_digit ?? "–";
  return d;
}

// ── Digit distribution bar chart ──────────────────────────────────────────────
async function loadDigitDist() {
  const d = await fetch("/api/dashboard/digit-distribution").then(r => r.json());
  const dist = d.distribution || {};
  const labels = Object.keys(dist).sort((a,b) => +a - +b);
  const values = labels.map(k => dist[k]);
  destroyChart("digitDist");
  charts.digitDist = new Chart(
    document.getElementById("chartDigitDist"),
    {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "Predictions",
          data: values,
          backgroundColor: PALETTE,
          borderRadius: 6,
          borderSkipped: false,
        }]
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, grid: { color: "#f0f0f0" } },
          x: { grid: { display: false } }
        },
        animation: { duration: 600 }
      }
    }
  );
}

// ── Confidence histogram ──────────────────────────────────────────────────────
async function loadConfHist() {
  const d = await fetch("/api/dashboard/confidence-histogram").then(r => r.json());
  const hist = d.histogram || {};
  const labels = Object.keys(hist);
  const values = Object.values(hist);
  destroyChart("confHist");
  charts.confHist = new Chart(
    document.getElementById("chartConfHist"),
    {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "Count",
          data: values,
          backgroundColor: "rgba(89,161,79,.75)",
          borderColor: "#59a14f",
          borderWidth: 1,
          borderRadius: 5,
        }]
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, grid: { color: "#f0f0f0" } },
          x: { grid: { display: false }, title: { display: true, text: "Confidence %" } }
        }
      }
    }
  );
}

// ── Model usage comparison ────────────────────────────────────────────────────
async function loadModelComparison() {
  const d = await fetch("/api/dashboard/model-comparison").then(r => r.json());
  const models = d.models || [];
  destroyChart("modelComp");
  charts.modelComp = new Chart(
    document.getElementById("chartModelComp"),
    {
      type: "bar",
      data: {
        labels: models.map(m => m.model),
        datasets: [
          {
            label: "Uses",
            data: models.map(m => m.uses),
            backgroundColor: "#4e79a7",
            borderRadius: 5,
            yAxisID: "y",
          },
          {
            label: "Avg Conf %",
            data: models.map(m => m.avg_conf),
            backgroundColor: "#f28e2b",
            borderRadius: 5,
            yAxisID: "y2",
          }
        ]
      },
      options: {
        plugins: { legend: { position: "top" } },
        scales: {
          y:  { beginAtZero: true, position: "left",  grid: { color: "#f0f0f0" } },
          y2: { beginAtZero: true, position: "right", max: 100, grid: { display: false } },
          x:  { grid: { display: false } }
        }
      }
    }
  );
}

// ── Input type pie chart ──────────────────────────────────────────────────────
async function loadInputSplit(summary) {
  destroyChart("inputSplit");
  charts.inputSplit = new Chart(
    document.getElementById("chartInputSplit"),
    {
      type: "doughnut",
      data: {
        labels: ["Canvas", "Upload"],
        datasets: [{
          data: [summary.canvas_count || 0, summary.upload_count || 0],
          backgroundColor: ["#4e79a7", "#f28e2b"],
          borderWidth: 2,
          borderColor: "#fff",
        }]
      },
      options: {
        plugins: {
          legend: { position: "bottom" },
          tooltip: { callbacks: {
            label: ctx => ` ${ctx.label}: ${ctx.parsed}`
          }}
        },
        cutout: "60%",
      }
    }
  );
}

// ── Model evaluation plots (PNG images) ──────────────────────────────────────
function loadModelPlots(model) {
  document.getElementById("plotTraining").src = `/static/plots/${model}_training_curves.png`;
  document.getElementById("plotCM").src       = `/static/plots/${model}_cm.png`;
  document.getElementById("plotROC").src      = `/static/plots/${model}_roc.png`;
  document.getElementById("plotPR").src       = `/static/plots/${model}_pr_curve.png`;
  // Hide broken images gracefully
  ["plotTraining","plotCM","plotROC","plotPR"].forEach(id => {
    const img = document.getElementById(id);
    img.onerror = () => {
      img.style.display = "none";
      img.insertAdjacentHTML("afterend",
        `<div class="text-muted small p-3 text-center">
           <i class="fas fa-info-circle me-1"></i>
           Plot not yet generated. Run <code>python -m training.train</code> first.
         </div>`);
    };
    img.style.display = "block";
  });
}

// ── Runtime logs ──────────────────────────────────────────────────────────────
async function loadLogs(level) {
  const lvl = level ?? document.getElementById("logLevel").value;
  let url = "/api/dashboard/runtime-logs?limit=80";
  if (lvl) url += `&level=${lvl}`;
  const d = await fetch(url).then(r => r.json());
  const console_ = document.getElementById("logConsole");
  if (!d.logs.length) {
    console_.innerHTML = '<span class="text-muted">No log entries found.</span>';
    return;
  }
  console_.innerHTML = d.logs.map(l =>
    `<div class="log-line">
       <span class="log-time">${(l.created_at || "").slice(0,19)}</span>
       <span class="log-${l.level} fw-semibold">[${l.level.padEnd(7)}]</span>
       <span class="ms-2">${l.message}</span>
     </div>`
  ).join("\n");
  console_.scrollTop = 0;
}

// ── Auto-refresh every 60s ────────────────────────────────────────────────────
async function refreshAll() {
  const summary = await loadSummary();
  await Promise.all([
    loadDigitDist(),
    loadConfHist(),
    loadModelComparison(),
    loadInputSplit(summary),
  ]);
  loadModelPlots(document.getElementById("plotModelSelect").value);
  await loadLogs();
}

// Init
refreshAll();
setInterval(refreshAll, 60_000);
