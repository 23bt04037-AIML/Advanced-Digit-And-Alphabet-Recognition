"""
seed_db.py – populate empty DB tables from existing files.

  model_metrics  ← models/*_metrics.json
  runtime_logs   ← logs/run_all.log  (+ any other *.log in logs/)
-
Run once:
    venv\Scripts\python.exe seed_db.py
"""
import json, re, sys, logging
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── DB setup ──────────────────────────────────────────────────────────────────
from backend.database.db import SessionLocal, engine, Base
from backend.database.models import ModelMetrics, RuntimeLog
Base.metadata.create_all(bind=engine)
db = SessionLocal()

# ══════════════════════════════════════════════════════════════════════════════
# 1. model_metrics  ←  models/*_metrics.json
# ══════════════════════════════════════════════════════════════════════════════
MODELS_DIR = ROOT / "models"

# Clear existing rows so re-runs are safe
db.query(ModelMetrics).delete()
db.commit()

best: dict = {}   # model_name → dict of fields

for p in sorted(MODELS_DIR.glob("*_metrics.json")):
    try:
        data = json.loads(p.read_text())

        model_name = data.get("model", p.stem.replace("_full_metrics", "").replace("_metrics", ""))

        def _norm(v):
            if v is None: return None
            return round(float(v) / 100.0, 6) if float(v) > 1.0 else round(float(v), 6)

        acc   = _norm(data.get("accuracy") or data.get("test_accuracy"))
        prec  = _norm(data.get("precision") or data.get("precision_score") or data.get("macro_precision"))
        rec   = _norm(data.get("recall")    or data.get("recall_score")    or data.get("macro_recall"))
        f1    = _norm(data.get("macro_f1")  or data.get("f1_score")        or data.get("f1"))
        auc   = _norm(data.get("auc_roc")   or data.get("auc"))
        cm    = data.get("confusion_matrix")
        epochs = data.get("epochs") or data.get("training_epochs")
        t_sec  = data.get("training_time_seconds") or data.get("train_time_s")
        params = data.get("parameters") or data.get("total_params")

        row = dict(
            model_name=model_name, accuracy=acc, precision_score=prec,
            recall_score=rec, f1_score=f1, auc_roc=auc,
            confusion_matrix=cm, training_epochs=epochs,
            training_time_seconds=t_sec, parameters=params,
        )

        # Keep the entry with the highest accuracy per model
        existing = best.get(model_name)
        if existing is None or (acc or 0) > (existing["accuracy"] or 0):
            best[model_name] = row
            log.info(f"  Best so far: {model_name}  acc={acc}")
    except Exception as e:
        log.warning(f"  Skipped {p.name}: {e}")

for row in best.values():
    db.add(ModelMetrics(**row))

db.commit()
inserted_metrics = len(best)
log.info(f"model_metrics: {inserted_metrics} rows inserted.")



# ══════════════════════════════════════════════════════════════════════════════
# 2. runtime_logs  ←  logs/*.log
# ══════════════════════════════════════════════════════════════════════════════
LOGS_DIR = ROOT / "logs"
LOG_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\s+\|\s+(\w+)\s+\|\s+(.+)$"
)
inserted_logs = 0

if LOGS_DIR.exists():
    # Clear existing rows so we don't duplicate on re-run
    db.query(RuntimeLog).delete()
    db.commit()

    log_files = [LOGS_DIR / "run_all.log"] + [
        p for p in sorted(LOGS_DIR.iterdir())
        if p.is_file() and p.name != "run_all.log" and p.suffix == ".log"
    ]

    for lf in log_files:
        if not lf.exists():
            continue
        module_name = lf.stem
        try:
            lines = lf.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines:
                m = LOG_RE.match(line.strip())
                if not m:
                    continue
                ts_str, level, message = m.group(1), m.group(2), m.group(3).strip()
                try:
                    ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    ts = None

                db.add(RuntimeLog(
                    level    = level,
                    module   = module_name,
                    message  = message,
                    extra    = None,
                    created_at = ts,
                ))
                inserted_logs += 1
            log.info(f"  Parsed {lf.name}: {inserted_logs} entries so far")
        except Exception as e:
            log.warning(f"  Could not parse {lf.name}: {e}")

    db.commit()

log.info(f"runtime_logs: {inserted_logs} rows inserted.")
db.close()
log.info("Done. Refresh DB Browser to see the data.")
