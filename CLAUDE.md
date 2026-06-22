# hw-aware-model-optimizer — Project context

## Purpose
Hardware-aware model optimization pipeline for Arm Ethos-U65 NPU.
Converts .h5 Keras models to Vela-compiled INT8 TFLite for deployment.

## Optimization rules
16 rules defined in optimizer/rules.py (RULE-L01 … RULE-M05).
Application order is canonical — see APPLICATION_ORDER in rules.py.

## Pipeline phases
1. Analysis     — optimizer/analyzer.py   (ModelAnalyzer)
2. Static xform — optimizer/transforms.py (no retraining)
3. TFMOT        — optimizer/tfmot_pipeline.py (Prune → Cluster → PCQAT)
4. TFLite export— optimizer/exporter.py
5. Vela compile — optimizer/exporter.py (compile_with_vela)

## Key decisions from design session
- RULE-C01 (BN folding): DO NOT apply manually in transforms.py.
  The TFLite converter folds BN into Conv2DBias automatically.
- Channel alignment target: multiples of 16 (NHCWB16 internal format)
- Default sparsity target: 0.70 global, 0.30 for first Conv2D layer
- Default cluster count: 32 (k=16 for maximum Vela compression)
- INT8 throughout; INT16 only for audio/HDR layers

## Hardware target
Ethos-U65, 512 MAC config, Shared_Sram memory mode.
Vela config: ethos-u65-512, Performance optimise, Ethos_U65_High_End.
