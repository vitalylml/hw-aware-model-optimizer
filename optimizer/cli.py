"""
hw-aware-model-optimizer CLI

Usage examples
--------------
# Full pipeline (analyze → transform → prune+cluster+QAT → export → vela)
python -m optimizer.cli run \
    --model model.h5 \
    --dataset calibration_data.npy \
    --output-dir ./output

# Analysis only (no transforms)
python -m optimizer.cli analyze --model model.h5

# Static transforms only (no TFMOT)
python -m optimizer.cli transform --model model.h5 --output transformed.h5

# Export only (assumes model is already PCQAT-trained)
python -m optimizer.cli export \
    --model pcqat_model.keras \
    --calibration calibration_data.npy \
    --output model.tflite

# Vela compile only
python -m optimizer.cli vela --tflite model.tflite --mac 512
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Configure root logger once here
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("optimizer.cli")


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_analyze(args: argparse.Namespace) -> None:
    from optimizer.analyzer import ModelAnalyzer
    from optimizer.report import print_analysis_report, save_html_report

    analyzer = ModelAnalyzer(args.model)
    report = analyzer.analyze()

    print_analysis_report(report)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report.save(out_dir / "analysis_report.json")
    save_html_report(report, output_path=out_dir / "analysis_report.html")

    n_violations = len(report.violations)
    if n_violations > 0:
        log.warning("%d violation(s) found — review analysis_report.html", n_violations)
    else:
        log.info("No violations found — model is compatible with Ethos-U65 rule set")


def cmd_transform(args: argparse.Namespace) -> None:
    import numpy as np
    import lscquant
    from optimizer.transforms import apply_all_static_transforms

    model = lscquant.load_model(args.model)
    transformed = apply_all_static_transforms(model)

    out_path = Path(args.output or "model_transformed.keras")
    lscquant.save_model(transformed, str(out_path))
    log.info("Static-transformed model saved to %s", out_path)

    if not args.vela_compare:
        return

    from optimizer.exporter import (
        VelaConfig, compile_with_vela, compare_vela_reports,
        export_tflite_int8, make_representative_dataset,
    )
    from optimizer.report import print_vela_comparison, print_vela_report

    if not args.calibration:
        log.error("--vela-compare requires --calibration <samples.npy>")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cal_data = np.load(args.calibration)
    rep_dataset = make_representative_dataset(cal_data)

    # Reload the original model from disk so the comparison uses the
    # untouched pre-transform graph rather than any in-memory artefact.
    original = lscquant.load_model(args.model)

    before_tflite = out_dir / "model_before.tflite"
    after_tflite  = out_dir / "model_after.tflite"

    log.info("Exporting pre-transform model to INT8 TFLite ...")
    export_tflite_int8(original, rep_dataset, output_path=before_tflite)
    log.info("Exporting post-transform model to INT8 TFLite ...")
    export_tflite_int8(transformed, rep_dataset, output_path=after_tflite)

    accel = f"ethos-u65-{args.mac}"
    before_cfg = VelaConfig(
        accelerator_config=accel,
        memory_mode=args.memory_mode,
        output_dir=str(out_dir / "vela_before"),
    )
    after_cfg = VelaConfig(
        accelerator_config=accel,
        memory_mode=args.memory_mode,
        output_dir=str(out_dir / "vela_after"),
    )

    log.info("Compiling pre-transform model with Vela ...")
    before_report = compile_with_vela(before_tflite, config=before_cfg)
    log.info("Compiling post-transform model with Vela ...")
    after_report  = compile_with_vela(after_tflite,  config=after_cfg)

    print_vela_report(before_report)
    print_vela_report(after_report)
    print_vela_comparison(before_report, after_report)

    before_report.save(out_dir / "vela_before.json")
    after_report.save(out_dir / "vela_after.json")
    comparison = compare_vela_reports(before_report, after_report)
    (out_dir / "vela_comparison.json").write_text(
        json.dumps(comparison, indent=2)
    )
    log.info("Vela comparison written to %s", out_dir / "vela_comparison.json")


def cmd_export(args: argparse.Namespace) -> None:
    import numpy as np
    import tensorflow as tf
    import lscquant
    from optimizer.exporter import export_tflite_int8, make_representative_dataset

    model = lscquant.load_model(args.model)
    calibration = np.load(args.calibration)
    rep_dataset = make_representative_dataset(calibration)

    out_path = Path(args.output or "model_optimized.tflite")
    export_tflite_int8(model, rep_dataset, output_path=out_path)


def cmd_vela(args: argparse.Namespace) -> None:
    from optimizer.exporter import VelaConfig, compile_with_vela
    from optimizer.report import print_vela_report

    vela_cfg = VelaConfig(
        accelerator_config=f"ethos-u65-{args.mac}",
        memory_mode=args.memory_mode,
        output_dir=args.output_dir,
    )
    report = compile_with_vela(args.tflite, config=vela_cfg)
    print_vela_report(report)
    report.save(Path(args.output_dir) / "vela_report.json")


def cmd_run(args: argparse.Namespace) -> None:
    """Full pipeline: analyze → transform → TFMOT → export → vela."""
    import numpy as np
    import tensorflow as tf
    import lscquant

    from optimizer.analyzer import ModelAnalyzer
    from optimizer.transforms import apply_all_static_transforms
    from optimizer.tfmot_pipeline import TFMOTPipelineConfig, run_tfmot_pipeline
    from optimizer.exporter import (
        VelaConfig, compile_with_vela,
        export_tflite_int8, make_representative_dataset,
        update_rules_from_vela,
    )
    from optimizer.report import (
        print_analysis_report, print_vela_report, save_html_report
    )
    from optimizer.rules import build_rule_catalogue

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rules = build_rule_catalogue()

    # ── Phase 1: Analysis ────────────────────────────────────────────────
    log.info("━━━ Phase 1: Analysis ━━━")
    analyzer = ModelAnalyzer(args.model)
    analysis = analyzer.analyze()
    print_analysis_report(analysis)
    analysis.save(out_dir / "analysis_report.json")

    # ── Phase 2: Static transforms ───────────────────────────────────────
    log.info("━━━ Phase 2: Static transforms ━━━")
    model = lscquant.load_model(args.model)
    model = apply_all_static_transforms(model)
    transformed_path = out_dir / "model_transformed.keras"
    lscquant.save_model(model, str(transformed_path))

    # ── Phase 3: TFMOT (requires dataset) ────────────────────────────────
    if args.dataset:
        log.info("━━━ Phase 3: TFMOT pipeline ━━━")
        data = np.load(args.dataset)

        # Build a minimal tf.data.Dataset (user should supply labels for real training)
        # Here we create a dummy dataset for structural demonstration
        if args.labels:
            labels = np.load(args.labels)
            dataset = (
                tf.data.Dataset
                .from_tensor_slices((data.astype(np.float32), labels))
                .batch(args.batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
        else:
            log.warning(
                "No --labels provided. Creating dummy dataset (for testing only). "
                "Provide real labels for meaningful accuracy results."
            )
            n_classes = _infer_n_classes(model)
            dummy_labels = np.zeros((len(data), n_classes), dtype=np.float32)
            dummy_labels[:, 0] = 1.0
            dataset = (
                tf.data.Dataset
                .from_tensor_slices((data.astype(np.float32), dummy_labels))
                .batch(args.batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )

        tfmot_cfg = TFMOTPipelineConfig(
            pruning_target_sparsity=args.sparsity,
            pruning_epochs=args.pruning_epochs,
            clustering_epochs=args.clustering_epochs,
            pcqat_epochs=args.pcqat_epochs,
            checkpoint_dir=str(out_dir / "checkpoints"),
            skip_pruning=args.skip_pruning,
            skip_clustering=args.skip_clustering,
            skip_pcqat=args.skip_pcqat,
        )

        model = run_tfmot_pipeline(model, dataset, config=tfmot_cfg)
        tfmot_path = out_dir / "model_tfmot.keras"
        lscquant.save_model(model, str(tfmot_path))
    else:
        log.warning(
            "No --dataset provided — skipping Phase 3 (TFMOT). "
            "Model will be exported as-is (no pruning/quantization)."
        )

    # ── Phase 4: TFLite export ───────────────────────────────────────────
    log.info("━━━ Phase 4: TFLite INT8 export ━━━")
    tflite_path = out_dir / "model_optimized.tflite"

    if args.calibration:
        cal_data = np.load(args.calibration)
    elif args.dataset:
        cal_data = np.load(args.dataset)[:200]  # use first 200 samples
    else:
        log.warning("No calibration data — exporting float model (no INT8 conversion)")
        cal_data = None

    if cal_data is not None:
        rep_dataset = make_representative_dataset(cal_data)
        export_tflite_int8(model, rep_dataset, output_path=tflite_path)
    else:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_path.write_bytes(converter.convert())
        log.info("Exported float TFLite to %s", tflite_path)

    # ── Phase 5: Vela compilation ────────────────────────────────────────
    vela_report = None
    if args.vela:
        log.info("━━━ Phase 5: Vela compilation ━━━")
        vela_cfg = VelaConfig(
            accelerator_config=f"ethos-u65-{args.mac}",
            output_dir=str(out_dir / "vela_output"),
        )
        try:
            vela_report = compile_with_vela(tflite_path, config=vela_cfg)
            print_vela_report(vela_report)
            vela_report.save(out_dir / "vela_report.json")

            # Update rules from Vela insights
            updated_rules = update_rules_from_vela(vela_report, rules)
            rules_path = out_dir / "updated_rules.json"
            rules_path.write_text(json.dumps(
                {k: (v.__dict__ if hasattr(v, "__dict__") else v)
                 for k, v in updated_rules.items()},
                indent=2, default=str,
            ))
            log.info("Updated rules written to %s", rules_path)

        except RuntimeError as e:
            log.warning("Vela unavailable: %s", e)

    # ── HTML report ───────────────────────────────────────────────────────
    save_html_report(
        analysis,
        vela_report=vela_report,
        output_path=out_dir / "optimization_report.html",
    )

    log.info("━━━ Pipeline complete. Outputs in %s ━━━", out_dir)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optimizer",
        description="Ethos-U65 hardware-aware model optimizer",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── analyze ──────────────────────────────────────────────────────────
    p_analyze = sub.add_parser("analyze", help="Analyze model against U65 rules")
    p_analyze.add_argument("--model",      required=True, help="Path to .h5 or .keras model")
    p_analyze.add_argument("--output-dir", default="./output", help="Directory for reports")

    # ── transform ────────────────────────────────────────────────────────
    p_transform = sub.add_parser("transform", help="Apply static transforms (no retraining)")
    p_transform.add_argument("--model",  required=True)
    p_transform.add_argument("--output", default="model_transformed.keras")
    p_transform.add_argument(
        "--vela-compare", action="store_true",
        help="Run Vela on both pre- and post-transform models and report the delta",
    )
    p_transform.add_argument(
        "--calibration", default=None,
        help=".npy of representative samples (required with --vela-compare)",
    )
    p_transform.add_argument("--mac", default="512", choices=["256", "512"])
    p_transform.add_argument(
        "--memory-mode", default="Shared_Sram",
        choices=[
            "Shared_Sram", "Dedicated_Sram", "Sram_Only",
            "Dedicated_Sram_256KB", "Dedicated_Sram_384KB", "Dedicated_Sram_512KB",
        ],
        help="Vela memory mode. Tighter modes (Dedicated_Sram_*KB) force the "
             "scheduler to cascade where Shared_Sram does not.",
    )
    p_transform.add_argument("--output-dir", default="./output")

    # ── export ───────────────────────────────────────────────────────────
    p_export = sub.add_parser("export", help="Convert Keras model to INT8 TFLite")
    p_export.add_argument("--model",       required=True)
    p_export.add_argument("--calibration", required=True, help=".npy file of calibration samples")
    p_export.add_argument("--output",      default="model_optimized.tflite")

    # ── vela ─────────────────────────────────────────────────────────────
    p_vela = sub.add_parser("vela", help="Compile .tflite with Vela for Ethos-U65")
    p_vela.add_argument("--tflite",     required=True)
    p_vela.add_argument("--mac",        default="512", choices=["256", "512"])
    p_vela.add_argument(
        "--memory-mode", default="Shared_Sram",
        choices=[
            "Shared_Sram", "Dedicated_Sram", "Sram_Only",
            "Dedicated_Sram_256KB", "Dedicated_Sram_384KB", "Dedicated_Sram_512KB",
        ],
    )
    p_vela.add_argument("--output-dir", default="./vela_output")

    # ── run (full pipeline) ───────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Run the full optimization pipeline")
    p_run.add_argument("--model",       required=True, help="Input .h5 or .keras model")
    p_run.add_argument("--dataset",     default=None,  help=".npy of training/calibration inputs")
    p_run.add_argument("--labels",      default=None,  help=".npy of one-hot training labels")
    p_run.add_argument("--calibration", default=None,  help=".npy of representative inputs (PTQ)")
    p_run.add_argument("--output-dir",  default="./output")
    p_run.add_argument("--batch-size",  type=int, default=32)
    p_run.add_argument("--sparsity",    type=float, default=0.70,
                       help="Target pruning sparsity (default 0.70)")
    p_run.add_argument("--pruning-epochs",   type=int, default=5)
    p_run.add_argument("--clustering-epochs",type=int, default=3)
    p_run.add_argument("--pcqat-epochs",     type=int, default=3)
    p_run.add_argument("--mac",         default="512", choices=["256", "512"])
    p_run.add_argument("--vela",        action="store_true",
                       help="Run Vela compilation after TFLite export")
    p_run.add_argument("--skip-pruning",    action="store_true")
    p_run.add_argument("--skip-clustering", action="store_true")
    p_run.add_argument("--skip-pcqat",      action="store_true")

    return parser


def _infer_n_classes(model) -> int:
    """Infer number of output classes from model output shape."""
    try:
        out_shape = model.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        return out_shape[-1]
    except Exception:
        return 10


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    handlers = {
        "analyze":   cmd_analyze,
        "transform": cmd_transform,
        "export":    cmd_export,
        "vela":      cmd_vela,
        "run":       cmd_run,
    }

    handler = handlers.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
