# hw-aware-model-optimizer

Hardware-aware model optimization pipeline for **Arm Ethos-U65 NPU** deployment.

Applies the 16-rule optimization ruleset (RULE-L01 … RULE-M05) to any Keras `.h5`
model and produces a fully optimized, Vela-compiled `.tflite` ready for deployment.

---

## Project structure

```
hw-aware-model-optimizer/
├── optimizer/
│   ├── __init__.py          # Top-level API  + optimize() convenience function
│   ├── rules.py             # All 16 rules as structured dataclasses
│   ├── analyzer.py          # Phase 1 — rule evaluation against H5 model
│   ├── transforms.py        # Phase 2 — static transforms (no retraining)
│   ├── tfmot_pipeline.py    # Phase 3 — Prune → Cluster → PCQAT
│   ├── exporter.py          # Phase 4 — TFLite INT8 export + Phase 5 — Vela
│   ├── report.py            # Console + HTML report generation
│   └── cli.py               # CLI entry point
├── tests/
│   └── test_optimizer.py
├── pyproject.toml
└── README.md
```

---

## Installation

```bash
cd C:\Repos\LML\hw-aware-model-optimizer

# Core (TF + TFMOT only)
pip install -e .

# With Vela compiler support
pip install -e ".[vela]"

# Full (includes dev/test tools)
pip install -e ".[all]"
```

---

## Pipeline overview

```
H5 model
   │
   ▼ Phase 1 — Analysis
   │  ModelAnalyzer("model.h5").analyze()
   │  → AnalysisReport (violations, warnings, per-layer stats)
   │
   ▼ Phase 2 — Static transforms  (no retraining)
   │  apply_all_static_transforms(model)
   │  Rules applied: M05, L06, C01, C04
   │
   ▼ Phase 3 — TFMOT pipeline  (requires calibration data + labels)
   │  run_tfmot_pipeline(model, dataset, config)
   │  Steps: Prune (L03/M03) → Cluster (L04) → PCQAT (M01/L02)
   │
   ▼ Phase 4 — TFLite INT8 export
   │  export_tflite_int8(model, representative_dataset_fn)
   │
   ▼ Phase 5 — Vela compilation
      compile_with_vela("model.tflite", VelaConfig(...))
      → VelaReport + updated rule parameters
```

---

## CLI usage

```bash
# Analyze only — produces analysis_report.html + .json
python -m optimizer.cli analyze --model model.h5 --output-dir ./output

# Static transforms only — no retraining
python -m optimizer.cli transform --model model.h5 --output transformed.keras

# Export PCQAT model to INT8 TFLite
python -m optimizer.cli export \
    --model pcqat_model.keras \
    --calibration calibration.npy \
    --output model.tflite

# Vela compile
python -m optimizer.cli vela --tflite model.tflite --mac 512

# Full pipeline
python -m optimizer.cli run \
    --model model.h5 \
    --dataset train_inputs.npy \
    --labels train_labels.npy \
    --output-dir ./output \
    --sparsity 0.70 \
    --pruning-epochs 5 \
    --clustering-epochs 3 \
    --pcqat-epochs 3 \
    --mac 512 \
    --vela

# Skip individual TFMOT steps (e.g. skip pruning if model already sparse)
python -m optimizer.cli run \
    --model model.h5 \
    --dataset calibration.npy \
    --skip-pruning \
    --skip-clustering \
    --output-dir ./output
```

Or use the installed script:

```bash
hw-optimizer run --model model.h5 --dataset data.npy --output-dir ./output --vela
```

---

## Python API

```python
# One-liner full pipeline
from optimizer import optimize

results = optimize(
    model_path="model.h5",
    dataset_path="train_inputs.npy",
    labels_path="train_labels.npy",
    output_dir="./output",
    run_vela=True,
)
print(results["tflite_path"])

# Analyze only
from optimizer import ModelAnalyzer
report = ModelAnalyzer("model.h5").analyze()
print(report.summary())
for v in report.violations:
    print(f"[{v.rule_id}] {v.layer_name}: {v.detail}")

# Static transforms only
import tensorflow as tf
import lscquant
from optimizer import apply_all_static_transforms
model = lscquant.load_model("model.h5")
model = apply_all_static_transforms(model)
model.save("model_transformed.keras")

# Custom TFMOT config
from optimizer import TFMOTPipelineConfig, run_tfmot_pipeline
cfg = TFMOTPipelineConfig(
    pruning_target_sparsity=0.80,
    n_clusters=16,
    pcqat_epochs=5,
    skip_clustering=False,
)
optimized = run_tfmot_pipeline(model, train_dataset, config=cfg)

# Vela with rule update
from optimizer import compile_with_vela, update_rules_from_vela, build_rule_catalogue, VelaConfig
rules = build_rule_catalogue()
vela_report = compile_with_vela("model.tflite", VelaConfig(accelerator_config="ethos-u65-512"))
updated_rules = update_rules_from_vela(vela_report, rules)
```

---

## Rule application order

Rules are applied in this sequence to avoid rework:

```
1.  RULE-M02  Architecture compatibility (gate check)
2.  RULE-M05  Batch size = 1
3.  RULE-M01  Global INT8 policy
4.  RULE-L05  Operator support
5.  RULE-L06  Activation substitution        ← Phase 2 (static)
6.  RULE-C05  LSTM/RNN fusion
7.  RULE-C01  BatchNorm folding              ← Phase 2 (static)
8.  RULE-C02  Conv-BN-ReLU fusion readiness
9.  RULE-C03  Residual Add placement
10. RULE-L01  Channel alignment (% 16)
11. RULE-M04  Input layer channel seeding
12. RULE-L07  Depthwise factorization
13. RULE-C04  Linear stack merge             ← Phase 2 (static)
14. RULE-M03  Global sparsity               ← Phase 3 TFMOT (prune)
15. RULE-L03  Per-layer pruning             ← Phase 3 TFMOT (prune)
16. RULE-L04  Weight clustering             ← Phase 3 TFMOT (cluster)
```

---

## Running tests

```bash
pytest tests/ -v

# With coverage
pytest tests/ --cov=optimizer --cov-report=term-missing
```

---

## Calibration data format

Calibration / representative dataset must be a `.npy` file of shape `(N, H, W, C)`
or `(N, D)` in `float32`. 100–200 samples is sufficient for PTQ.

```python
import numpy as np
# Example: 200 random RGB images at 224x224
calibration = np.random.rand(200, 224, 224, 3).astype(np.float32)
np.save("calibration.npy", calibration)
```

---

## Output files

After a full `run`, the `--output-dir` contains:

| File | Description |
|---|---|
| `analysis_report.json` | Phase 1 findings (violations, warnings, layer stats) |
| `analysis_report.html` | Human-readable HTML analysis report |
| `model_transformed.keras` | Phase 2 output (static transforms applied) |
| `checkpoints/model_after_pruning.keras` | Phase 3 pruning checkpoint |
| `checkpoints/model_after_clustering.keras` | Phase 3 clustering checkpoint |
| `checkpoints/model_after_pcqat.keras` | Phase 3 PCQAT checkpoint |
| `model_optimized.tflite` | Phase 4 INT8 TFLite export |
| `vela_output/*_vela.tflite` | Phase 5 Vela-compiled NPU binary |
| `vela_output/*_Summary.csv` | Phase 5 per-layer performance stats |
| `vela_report.json` | Phase 5 parsed report |
| `updated_rules.json` | Rule parameters updated from Vela insights |
| `optimization_report.html` | Combined analysis + Vela HTML report |
