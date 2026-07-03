"""
run_all.py  -  Master pipeline for AdvancedDigitRecognition
===========================================================
Runs every module in dependency order:

  Stage 1  - Training         (all CNN models x all optimizers)
  Stage 2  - Transfer         (MobileNetV2 + ResNet50 with Adam)
  Stage 3  - Optimizer sweep  (medium CNN x 5 optimizers, quick comparison)
  Stage 4  - Cross-validation (all CNN models, 5-fold)
  Stage 5  - Full evaluation  (metrics + plots for every trained model)
  Stage 6  - FGSM attack      (robustness curve for cnn_medium_adam)
  Stage 7  - Adversarial def. (adversarial training on cnn_medium_adam)
  Stage 8  - Reporting        (PDF / CSV / Excel from DB -- skipped if DB absent)

Usage:
    python run_all.py [--skip-transfer] [--skip-adversarial] [--skip-report]
"""

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path

# Suppress TensorFlow OneDNN noise
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Force UTF-8 output on Windows so Unicode chars don't crash
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/run_all.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("run_all")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def section(title: str):
    bar = "=" * 62
    logger.info("\n%s\n  %s\n%s", bar, title, bar)


def run_stage(name: str, fn, *args, **kwargs):
    section(name)
    t0 = time.time()
    try:
        fn(*args, **kwargs)
        logger.info("[OK]  %s completed in %.1fs", name, time.time() - t0)
        return True
    except Exception:
        logger.error("[FAIL] %s FAILED after %.1fs", name, time.time() - t0)
        logger.error(traceback.format_exc())
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Stage implementations
# ─────────────────────────────────────────────────────────────────────────────
def stage_train_cnn():
    """Stage 1: Train only missing model+optimizer combinations.
    AUTO-SKIPS any run where <model>_<optimizer>.keras already exists.
    If all 9 models exist, the entire stage is skipped automatically."""
    from pathlib import Path
    import training.train as _tr
    from training.train import main as train_main

    models_dir = Path("models")
    all_combos = [
        (m, o)
        for m in ["cnn_small", "cnn_medium", "cnn_deep"]
        for o in list(_tr.OPTIMIZER_MAP.keys())
    ]
    missing = [m for m, o in all_combos
               if not (models_dir / f"{m}_{o}.keras").exists()]

    if not missing:
        logger.info("[AUTO-SKIP] All CNN models already exist in models/ — skipping training.")
        return

    models_to_run = list(dict.fromkeys(m for m, _ in [
        (m, o) for m, o in all_combos
        if not (models_dir / f"{m}_{o}.keras").exists()
    ]))
    logger.info("Missing models: %s — training now.", models_to_run)

    class _Args:
        model = "all"
        optimizer = "all"
        epochs = 25
        batch_size = 256

    _orig_builders = dict(_tr.BUILDERS)
    _tr.BUILDERS = {k: v for k, v in _orig_builders.items() if k in models_to_run}
    try:
        train_main(_Args())
    finally:
        _tr.BUILDERS = _orig_builders


def stage_transfer(model_choice="both"):
    """Stage 2: Train MobileNetV2 and/or ResNet50."""
    from training.transfer_learning import run as tl_run
    tl_run(model_choice)


def stage_optimizer_sweep():
    """Stage 3: Systematic optimizer comparison on medium CNN."""
    from training.optimizer_experiments import run as opt_run
    opt_run()


def stage_cross_validation(model="cnn_small", folds=5, epochs=10):
    """Stage 4: Stratified K-fold CV."""
    from evaluation.cross_validation import run_cv
    for m in ["cnn_small", "cnn_medium", "cnn_deep"]:
        logger.info(f"Running {folds}-fold CV for {m} …")
        run_cv(m, n_folds=folds, epochs=epochs)


def stage_full_evaluation():
    """Stage 5: Full evaluation (metrics + all plots) for every trained model."""
    import tensorflow as tf
    from tensorflow import keras
    from training.train import load_mnist
    from evaluation.metrics import full_evaluation, plot_model_comparison

    _, (x_te, y_te, _) = load_mnist()
    models_dir = Path("models")
    summaries = []

    # Evaluate every .keras file (skip *_best.keras and *_adversarial.keras)
    for kfile in sorted(models_dir.glob("*.keras")):
        stem = kfile.stem
        if stem.endswith("_best") or "adversarial" in stem or "distilled" in stem:
            continue
        logger.info(f"  Evaluating {stem} …")
        try:
            model = keras.models.load_model(str(kfile))
            m = full_evaluation(model, stem, x_te, y_te)
            summaries.append(m)
            tf.keras.backend.clear_session()
        except Exception:
            logger.warning(f"  Could not evaluate {stem}: {traceback.format_exc()}")

    if summaries:
        plot_model_comparison(summaries)
        logger.info(f"Evaluated {len(summaries)} model(s); comparison plot saved.")


def stage_fgsm(model_name="cnn_medium_adam", epsilons=None):
    """Stage 6: FGSM robustness evaluation."""
    if epsilons is None:
        epsilons = [0.05, 0.1, 0.2, 0.3]
    from adversarial.fgsm_attack import run as fgsm_run
    fgsm_run(model_name, epsilons)


def stage_adversarial_training(base="cnn_medium_adam"):
    """Stage 7: Adversarial training (fine-tune on FGSM examples)."""
    from adversarial.defenses import adversarial_training
    adversarial_training(base_model_name=base)


def stage_reporting():
    """Stage 8: Generate PDF / CSV / Excel reports (requires DB)."""
    from reporting.report_generator import generate_all_reports
    paths = generate_all_reports()
    for fmt, path in paths.items():
        logger.info(f"  {fmt.upper():6}: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Run the full AdvancedDigitRecognition pipeline.")
    ap.add_argument("--skip-training",    action="store_true",
                    help="Skip Stage 1 CNN training (auto-skipped anyway if all .keras files exist)")
    ap.add_argument("--skip-transfer",    action="store_true", help="Skip transfer learning stage")
    ap.add_argument("--skip-adversarial", action="store_true", help="Skip adversarial stages")
    ap.add_argument("--skip-report",      action="store_true", help="Skip report generation")
    ap.add_argument("--cv-folds",  type=int, default=5,  help="Number of CV folds (default 5)")
    ap.add_argument("--cv-epochs", type=int, default=10, help="Epochs per CV fold (default 10)")
    args = ap.parse_args()

    total_t0 = time.time()
    results = {}

    # -- Stage 1: CNN training grid -------------------------------------------
    # stage_train_cnn() auto-skips models that already have .keras files.
    # --skip-training forces skip regardless.
    if not args.skip_training:
        results["1_train_cnn"] = run_stage(
            "Stage 1 - CNN Training Grid",
            stage_train_cnn,
        )
    else:
        logger.info("[SKIP] Stage 1 - CNN training skipped via --skip-training")

    # -- Stage 2: Transfer learning -------------------------------------------
    if not args.skip_transfer:
        results["2_transfer"] = run_stage(
            "Stage 2 - Transfer Learning (MobileNetV2 + ResNet50)",
            stage_transfer, "both",
        )

    # -- Stage 3: Optimizer comparison sweep ----------------------------------
    results["3_optimizer_sweep"] = run_stage(
        "Stage 3 - Optimizer Comparison Sweep",
        stage_optimizer_sweep,
    )

    # -- Stage 4: Cross-validation --------------------------------------------
    results["4_cross_validation"] = run_stage(
        "Stage 4 - {}-Fold Cross-Validation (all CNN models)".format(args.cv_folds),
        stage_cross_validation,
        folds=args.cv_folds,
        epochs=args.cv_epochs,
    )

    # -- Stage 5: Full evaluation ---------------------------------------------
    results["5_evaluation"] = run_stage(
        "Stage 5 - Full Evaluation (metrics + all plots)",
        stage_full_evaluation,
    )

    # -- Stage 6 & 7: Adversarial ---------------------------------------------
    if not args.skip_adversarial:
        results["6_fgsm"] = run_stage(
            "Stage 6 - FGSM Adversarial Attack",
            stage_fgsm, "cnn_medium_adam",
        )
        results["7_adv_training"] = run_stage(
            "Stage 7 - Adversarial Training Defense",
            stage_adversarial_training, "cnn_medium_adam",
        )

    # -- Stage 8: Reporting ---------------------------------------------------
    if not args.skip_report:
        results["8_reporting"] = run_stage(
            "Stage 8 - Report Generation (PDF / CSV / Excel)",
            stage_reporting,
        )

    # -- Final summary --------------------------------------------------------
    elapsed = time.time() - total_t0
    section("Pipeline Complete  ({:.1f} min total)".format(elapsed / 60))
    for stage, ok in results.items():
        status = "[PASSED]" if ok else "[FAILED]"
        logger.info("  %s  %s", status, stage)

    failed = [s for s, ok in results.items() if not ok]
    if failed:
        logger.warning("%d stage(s) failed: %s", len(failed), failed)
        sys.exit(1)
    else:
        logger.info("All stages passed successfully!")


if __name__ == "__main__":
    main()
