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

# Update skills for a HW target (evaluate all rules, rewrite last_eval + enabled)
python -m optimizer.cli update-skills \
    --descriptor skills/update_ethos_u65_256_384k.json
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


def cmd_update_skills(args: argparse.Namespace) -> None:
    """Evaluate every rule in a skills file and update last_eval + enabled flags."""
    import importlib.util
    import tempfile

    # ── Load descriptor ──────────────────────────────────────────────────
    descriptor_path = Path(args.descriptor)
    if not descriptor_path.exists():
        log.error("Descriptor not found: %s", descriptor_path)
        sys.exit(1)
    descriptor = json.loads(descriptor_path.read_text())
    hw_id = descriptor.get("hw_id")
    perf_script_rel = descriptor.get("perf_script")
    if not hw_id or not perf_script_rel:
        log.error("Descriptor must contain 'hw_id' and 'perf_script'")
        sys.exit(1)
    perf_script_path = Path(perf_script_rel)
    if not perf_script_path.exists():
        log.error("perf_script not found: %s", perf_script_path)
        sys.exit(1)

    # ── Locate skills file by hw_id ──────────────────────────────────────
    skills_dir = Path(args.skills_dir)
    skills_file = skills_data = None
    for f in sorted(skills_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("hw_id") == hw_id:
            skills_file, skills_data = f, data
            break
    if skills_file is None:
        log.error("No skills file found for hw_id=%r in %s", hw_id, skills_dir)
        sys.exit(1)
    log.info("Skills file: %s", skills_file)

    # ── Load measure() from perf_script ─────────────────────────────────
    spec = importlib.util.spec_from_file_location("_perf_module", perf_script_path)
    perf_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(perf_module)
    measure_fn = getattr(perf_module, "measure", None)
    if measure_fn is None:
        log.error("perf_script %s must expose a measure(tflite_path) function",
                  perf_script_path)
        sys.exit(1)

    # ── Evaluate each rule ───────────────────────────────────────────────
    fixtures_dir = Path(args.fixtures_dir)
    threshold = args.threshold
    results = []

    with tempfile.TemporaryDirectory(prefix="skills_update_") as tmpdir:
        tmp = Path(tmpdir)
        for rule_entry in skills_data["rules"]:
            rule_id = rule_entry["rule_id"]
            log.info("Evaluating %s …", rule_id)
            fixture_dir = fixtures_dir / rule_id
            before_h5 = fixture_dir / "before.h5"
            after_h5 = fixture_dir / "after.h5"

            if not before_h5.exists() or not after_h5.exists():
                log.warning("No fixtures for %s — skipping", rule_id)
                results.append({"rule_id": rule_id, "status": "skipped",
                                 "reason": "no fixtures"})
                continue

            rule_tmp = tmp / rule_id
            rule_tmp.mkdir(parents=True, exist_ok=True)

            try:
                before_tflite = _export_fixture_to_tflite(
                    before_h5, rule_tmp / "before.tflite")
                after_tflite = _export_fixture_to_tflite(
                    after_h5, rule_tmp / "after.tflite")
            except Exception as exc:
                log.warning("TFLite export failed for %s: %s", rule_id, exc)
                results.append({"rule_id": rule_id, "status": "error",
                                 "reason": f"export: {exc}"})
                continue

            try:
                before_m = measure_fn(before_tflite, rule_tmp / "vela_before")
                after_m = measure_fn(after_tflite, rule_tmp / "vela_after")
            except Exception as exc:
                log.warning("Vela measurement failed for %s: %s", rule_id, exc)
                results.append({"rule_id": rule_id, "status": "error",
                                 "reason": f"vela: {exc}"})
                continue

            before_us = before_m["batch_inference_time_us"]
            after_us = after_m["batch_inference_time_us"]

            if before_us == 0.0:
                log.warning("Vela returned 0 us for %s before-fixture -- skipping",
                             rule_id)
                results.append({"rule_id": rule_id, "status": "skipped",
                                 "reason": "Vela 0 us"})
                continue

            delta_us = after_us - before_us
            delta_pct = (delta_us / before_us) * 100.0

            rule_entry["last_eval"] = {
                "before_us": round(before_us, 2),
                "after_us":  round(after_us, 2),
                "delta_us":  round(delta_us, 2),
                "delta_pct": round(delta_pct, 2),
            }

            if delta_pct <= -threshold:
                rule_entry["enabled"] = True
                status = "enabled"
            else:
                rule_entry["enabled"] = False
                status = "disabled"

            results.append({
                "rule_id": rule_id, "status": status,
                "before_us": before_us, "after_us": after_us,
                "delta_us": delta_us, "delta_pct": delta_pct,
            })
            log.debug("%s  %+.1f µs (%+.1f%%) → %s", rule_id, delta_us, delta_pct, status)

    # ── Persist updated skills ───────────────────────────────────────────
    skills_file.write_text(json.dumps(skills_data, indent=2))
    log.info("Skills file updated: %s", skills_file)

    # ── Print summary ────────────────────────────────────────────────────
    _print_skills_update_summary(hw_id, results, threshold)


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

    # ── update-skills ─────────────────────────────────────────────────────
    p_upd = sub.add_parser(
        "update-skills",
        help="Evaluate rules against HW via a perf script and update the skills file",
    )
    p_upd.add_argument(
        "--descriptor", required=True,
        help='Path to JSON descriptor: {"hw_id": "...", "perf_script": "..."}',
    )
    p_upd.add_argument(
        "--skills-dir", default="./skills",
        help="Directory containing skills JSON files (default: ./skills)",
    )
    p_upd.add_argument(
        "--fixtures-dir", default="./tests/fixtures",
        help="Root directory of rule fixtures (default: ./tests/fixtures)",
    )
    p_upd.add_argument(
        "--threshold", type=float, default=2.0,
        help="Min improvement %% to enable/disable a rule (default: 2.0)",
    )

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


def _export_fixture_to_tflite(h5_path: Path, output_path: Path) -> Path:
    """Convert a Keras fixture .h5 to INT8 TFLite using random calibration data."""
    import numpy as np
    import tensorflow as tf

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(str(h5_path), compile=False)

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    cal_shape = tuple(d if d is not None else 1 for d in input_shape)

    def representative_dataset():
        rng = np.random.default_rng(seed=0)
        for _ in range(50):
            yield [rng.uniform(0.0, 1.0, cal_shape).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    output_path.write_bytes(converter.convert())
    return output_path


def _print_skills_update_summary(
    hw_id: str, results: list, threshold: float
) -> None:
    W = 80
    print()
    print(f"Skills Update -- {hw_id}")
    print("=" * W)
    print(
        f"{'Rule':<14} {'Before (us)':>12} {'After (us)':>12} "
        f"{'Delta (us)':>12} {'Delta %':>9}   Status"
    )
    print("-" * W)

    n_enabled = n_disabled = n_unchanged = n_skip = 0
    for r in results:
        rid = r["rule_id"]
        st = r["status"]
        if st in ("skipped", "error"):
            n_skip += 1
            print(
                f"{rid:<14} {'--':>12} {'--':>12} {'--':>12} {'--':>9}"
                f"   {st} ({r.get('reason', '')})"
            )
            continue
        b, a, d, p = r["before_us"], r["after_us"], r["delta_us"], r["delta_pct"]
        sign = "+" if d >= 0 else ""
        if st == "enabled":
            n_enabled += 1
        elif st == "disabled":
            n_disabled += 1
        else:
            n_unchanged += 1
        print(
            f"{rid:<14} {b:>12.1f} {a:>12.1f} "
            f"{sign + f'{d:.1f}':>12} {sign + f'{p:.1f}%':>9}   {st}"
        )

    measured = n_enabled + n_disabled + n_unchanged
    print("=" * W)
    print(
        f"Measured: {measured}   enabled: {n_enabled}   disabled: {n_disabled}   "
        f"unchanged: {n_unchanged}   skipped/error: {n_skip}   "
        f"threshold: +/-{threshold:.1f}%"
    )
    print()


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
        "analyze":       cmd_analyze,
        "transform":     cmd_transform,
        "export":        cmd_export,
        "vela":          cmd_vela,
        "run":           cmd_run,
        "update-skills": cmd_update_skills,
    }

    handler = handlers.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
