"""
hw-aware-model-optimizer
========================
Hardware-aware model optimization pipeline for Arm Ethos-U65 NPU deployment.

Quick start
-----------
    from optimizer import ModelAnalyzer, optimize

    # Analyze only
    report = ModelAnalyzer("model.h5").analyze()
    print(report.summary())

    # Full pipeline
    optimize(
        model_path="model.h5",
        dataset_path="calibration.npy",
        output_dir="./output",
        run_vela=True,
    )
"""
from optimizer.analyzer import ModelAnalyzer, AnalysisReport
from optimizer.rules import build_rule_catalogue, APPLICATION_ORDER, SUPPORTED_OPS
from optimizer.transforms import (
    apply_all_static_transforms,
    apply_cascading_optimizations,
    fold_maxpool_into_strided_conv,
    replace_bilinear_resize_with_conv2dtranspose,
    fuse_split_concat_blocks,
    reorder_residual_add_operands,
)
from optimizer.tfmot_pipeline import TFMOTPipelineConfig, run_tfmot_pipeline
from optimizer.exporter import (
    export_tflite_int8,
    make_representative_dataset,
    compile_with_vela,
    compare_vela_reports,
    update_rules_from_vela,
    VelaConfig,
    VelaCascade,
)
from optimizer.report import (
    print_analysis_report,
    print_vela_report,
    print_vela_comparison,
    save_html_report,
)

__version__ = "0.1.0"
__all__ = [
    "ModelAnalyzer",
    "AnalysisReport",
    "build_rule_catalogue",
    "APPLICATION_ORDER",
    "SUPPORTED_OPS",
    "apply_all_static_transforms",
    "apply_cascading_optimizations",
    "fold_maxpool_into_strided_conv",
    "replace_bilinear_resize_with_conv2dtranspose",
    "fuse_split_concat_blocks",
    "reorder_residual_add_operands",
    "TFMOTPipelineConfig",
    "run_tfmot_pipeline",
    "export_tflite_int8",
    "make_representative_dataset",
    "compile_with_vela",
    "compare_vela_reports",
    "update_rules_from_vela",
    "VelaConfig",
    "VelaCascade",
    "print_analysis_report",
    "print_vela_report",
    "print_vela_comparison",
    "save_html_report",
    "optimize",
]


def optimize(
    model_path: str,
    dataset_path: str | None = None,
    labels_path: str | None = None,
    calibration_path: str | None = None,
    output_dir: str = "./output",
    tfmot_config: TFMOTPipelineConfig | None = None,
    vela_config: VelaConfig | None = None,
    run_vela: bool = False,
) -> dict:
    """
    Convenience entry point — runs the full pipeline and returns a results dict.

    Parameters
    ----------
    model_path       : path to .h5 or .keras model
    dataset_path     : path to .npy training/calibration inputs (float32)
    labels_path      : path to .npy one-hot labels (required for TFMOT)
    calibration_path : path to .npy representative samples for PTQ (uses dataset if None)
    output_dir       : directory for all outputs
    tfmot_config     : TFMOTPipelineConfig; defaults used if None
    vela_config      : VelaConfig; defaults used if None
    run_vela         : whether to run Vela compilation

    Returns
    -------
    dict with keys: analysis, tflite_path, vela_report (if run_vela)
    """
    import numpy as np
    import tensorflow as tf
    import lscquant
    from pathlib import Path

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Phase 1
    analysis = ModelAnalyzer(model_path).analyze()
    print_analysis_report(analysis)
    analysis.save(out_dir / "analysis_report.json")
    results["analysis"] = analysis

    # Phase 2
    model = lscquant.load_model(model_path)
    model = apply_all_static_transforms(model)

    # Phase 3
    if dataset_path:
        data = np.load(dataset_path)
        if labels_path:
            labels = np.load(labels_path)
        else:
            n_classes = model.output_shape[-1]
            labels = np.zeros((len(data), n_classes), dtype=np.float32)
            labels[:, 0] = 1.0

        dataset = (
            tf.data.Dataset
            .from_tensor_slices((data.astype(np.float32), labels))
            .batch(32)
            .prefetch(tf.data.AUTOTUNE)
        )
        model = run_tfmot_pipeline(
            model, dataset,
            config=tfmot_config,
            save_intermediates=True,
        )

    # Phase 4
    tflite_path = out_dir / "model_optimized.tflite"
    cal_src = calibration_path or dataset_path
    if cal_src:
        cal_data = np.load(cal_src)[:200]
        export_tflite_int8(model, make_representative_dataset(cal_data), tflite_path)
    else:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_path.write_bytes(converter.convert())

    results["tflite_path"] = str(tflite_path)

    # Phase 5
    if run_vela:
        vela_report = compile_with_vela(tflite_path, config=vela_config)
        print_vela_report(vela_report)
        vela_report.save(out_dir / "vela_report.json")
        results["vela_report"] = vela_report

    # HTML report
    save_html_report(
        analysis,
        vela_report=results.get("vela_report"),
        output_path=out_dir / "optimization_report.html",
    )

    return results
