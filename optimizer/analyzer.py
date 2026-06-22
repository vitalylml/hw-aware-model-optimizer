"""
Phase 1 — Analysis pass.

Loads an H5 (or .keras) model, walks every layer and combination,
evaluates each rule's condition, and returns a structured AnalysisReport.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Finding:
    rule_id: str
    layer_name: str
    layer_type: str
    severity: str          # "violation" | "warning" | "info"
    detail: str
    suggested_action: str = ""


@dataclass
class LayerInfo:
    name: str
    layer_type: str
    input_shape: Optional[tuple]
    output_shape: Optional[tuple]
    dtype: str
    config: dict
    param_count: int = 0
    sparsity: float = 0.0
    unique_weights: int = 0


@dataclass
class AnalysisReport:
    model_path: str
    total_params: int = 0
    trainable_params: int = 0
    findings: List[Finding] = field(default_factory=list)
    layers: List[LayerInfo] = field(default_factory=list)
    global_sparsity: float = 0.0
    has_dram_risk: bool = False        # model too large for SRAM alone
    is_transformer: bool = False
    job_boundaries: List[str] = field(default_factory=list)  # expected fallback points

    @property
    def violations(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "violation"]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    def summary(self) -> str:
        lines = [
            f"Model         : {self.model_path}",
            f"Total params  : {self.total_params:,}",
            f"Global sparsity: {self.global_sparsity:.1%}",
            f"Violations    : {len(self.violations)}",
            f"Warnings      : {len(self.warnings)}",
            f"Transformer   : {self.is_transformer}",
            f"DRAM risk     : {self.has_dram_risk}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "total_params": self.total_params,
            "global_sparsity": self.global_sparsity,
            "is_transformer": self.is_transformer,
            "has_dram_risk": self.has_dram_risk,
            "violations": [v.__dict__ for v in self.violations],
            "warnings": [w.__dict__ for w in self.warnings],
            "layers": [l.__dict__ for l in self.layers],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        log.info("Analysis report saved to %s", path)


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------

class ModelAnalyzer:
    """
    Evaluates all Ethos-U65 optimization rules against a Keras model.

    Usage
    -----
    analyzer = ModelAnalyzer(model_path="model.h5")
    report   = analyzer.analyze()
    print(report.summary())
    """

    # SRAM threshold: if parameter memory exceeds this, DRAM spilling is likely
    SRAM_THRESHOLD_KB = 512

    def __init__(self, model_path: str | Path):
        self.model_path = str(model_path)
        self._model = None

    # ------------------------------------------------------------------
    def analyze(self) -> AnalysisReport:
        import tensorflow as tf
        import lscquant
        from optimizer.rules import (
            SUPPORTED_OPS, SUPPORTED_ACTIVATIONS,
            FALLBACK_ACTIVATIONS, ATTENTION_LAYER_TYPES,
            WEIGHT_LAYER_TYPES, FOLDABLE_CONV_TYPES,
            CHANNEL_MULTIPLE,
        )

        log.info("Loading model from %s", self.model_path)
        self._model = lscquant.load_model(self.model_path)
        model = self._model

        report = AnalysisReport(model_path=self.model_path)
        report.total_params = model.count_params()
        report.trainable_params = sum(
            np.prod(v.shape) for v in model.trainable_variables
        )

        # ── Per-layer pass ───────────────────────────────────────────────
        all_weights: list[np.ndarray] = []
        first_conv_seen = False

        for layer in model.layers:
            ltype = type(layer).__name__
            cfg = layer.get_config()

            # Collect weight arrays for global sparsity
            w_arrays = [w.numpy() for w in layer.weights if len(w.shape) >= 1]
            all_weights.extend(w_arrays)

            # Compute per-layer sparsity
            layer_sparsity = 0.0
            unique_w = 0
            param_count = sum(np.prod(w.shape) for w in layer.weights)
            if w_arrays:
                flat = np.concatenate([w.flatten() for w in w_arrays])
                layer_sparsity = float(np.mean(flat == 0))
                unique_w = int(np.unique(flat).size)

            # Shapes
            try:
                in_shape = tuple(layer.input_shape)
            except Exception:
                in_shape = None
            try:
                out_shape = tuple(layer.output_shape)
            except Exception:
                out_shape = None

            info = LayerInfo(
                name=layer.name,
                layer_type=ltype,
                input_shape=in_shape,
                output_shape=out_shape,
                dtype=str(layer.dtype),
                config={k: str(v) for k, v in cfg.items()
                        if k in ("filters", "units", "kernel_size",
                                 "activation", "use_bias", "strides")},
                param_count=int(param_count),
                sparsity=layer_sparsity,
                unique_weights=unique_w,
            )
            report.layers.append(info)

            # ── RULE-M05: batch size ─────────────────────────────────────
            if in_shape and len(in_shape) >= 1:
                shape = in_shape[0] if isinstance(in_shape[0], tuple) else in_shape
                bs = shape[0] if shape else None
                if bs not in (None, 1):
                    report.findings.append(Finding(
                        rule_id="RULE-M05", layer_name=layer.name, layer_type=ltype,
                        severity="violation",
                        detail=f"batch_size={bs} — Ethos-U65 requires N=1",
                        suggested_action="Fix batch dimension to 1 at export.",
                    ))

            # ── RULE-M02: attention / transformer ───────────────────────
            if ltype in ATTENTION_LAYER_TYPES:
                report.is_transformer = True
                report.findings.append(Finding(
                    rule_id="RULE-M02", layer_name=layer.name, layer_type=ltype,
                    severity="violation",
                    detail=f"{ltype} is not supported on Ethos-U65 — will become a job boundary",
                    suggested_action="Replace with depthwise conv approximation or migrate to U85.",
                ))
                report.job_boundaries.append(layer.name)

            # ── RULE-L05: unsupported operator ───────────────────────────
            if ltype not in SUPPORTED_OPS and ltype not in ATTENTION_LAYER_TYPES:
                report.findings.append(Finding(
                    rule_id="RULE-L05", layer_name=layer.name, layer_type=ltype,
                    severity="violation",
                    detail=f"{ltype} is not in Ethos-U65 supported op set",
                    suggested_action="Replace with supported equivalent or accept CMSIS-NN fallback.",
                ))
                report.job_boundaries.append(layer.name)

            # ── RULE-L01: channel alignment ──────────────────────────────
            if ltype in WEIGHT_LAYER_TYPES:
                filters = cfg.get("filters") or cfg.get("units")
                if filters and int(filters) % CHANNEL_MULTIPLE != 0:
                    report.findings.append(Finding(
                        rule_id="RULE-L01", layer_name=layer.name, layer_type=ltype,
                        severity="violation",
                        detail=(
                            f"output_channels={filters} is not a multiple of {CHANNEL_MULTIPLE} "
                            f"— NHCWB16 padding overhead"
                        ),
                        suggested_action=(
                            f"Round up to {int(np.ceil(int(filters) / CHANNEL_MULTIPLE) * CHANNEL_MULTIPLE)}."
                        ),
                    ))

            # ── RULE-L06: unsupported activation ─────────────────────────
            activation = cfg.get("activation")
            if isinstance(activation, dict):
                activation = activation.get("class_name", "").lower()
            elif isinstance(activation, str):
                activation = activation.lower()
            if activation and activation not in SUPPORTED_ACTIVATIONS:
                sub = FALLBACK_ACTIVATIONS.get(activation, "relu6")
                report.findings.append(Finding(
                    rule_id="RULE-L06", layer_name=layer.name, layer_type=ltype,
                    severity="violation",
                    detail=f"activation='{activation}' is not natively supported on U65",
                    suggested_action=f"Substitute with '{sub}'.",
                ))

            # ── RULE-L02: float32 dtype ──────────────────────────────────
            if layer.dtype == "float32" and ltype not in ("InputLayer",):
                report.findings.append(Finding(
                    rule_id="RULE-L02", layer_name=layer.name, layer_type=ltype,
                    severity="warning",
                    detail="Layer dtype=float32 — INT8 quantization needed for U65 deployment",
                    suggested_action="Apply QAT or PTQ (Phase 3 pipeline).",
                ))

            # ── RULE-L03: per-layer sparsity ─────────────────────────────
            if ltype in WEIGHT_LAYER_TYPES and param_count > 0:
                threshold = 0.3 if not first_conv_seen else 0.5
                if layer_sparsity < threshold:
                    report.findings.append(Finding(
                        rule_id="RULE-L03", layer_name=layer.name, layer_type=ltype,
                        severity="warning",
                        detail=f"sparsity={layer_sparsity:.1%} below target ({threshold:.0%})",
                        suggested_action="Apply magnitude pruning in TFMOT pipeline.",
                    ))
                if ltype == "Conv2D" and not first_conv_seen:
                    first_conv_seen = True

            # ── RULE-L04: weight clustering ──────────────────────────────
            if ltype in WEIGHT_LAYER_TYPES and unique_w > 32:
                report.findings.append(Finding(
                    rule_id="RULE-L04", layer_name=layer.name, layer_type=ltype,
                    severity="warning",
                    detail=f"unique weight values={unique_w} — weight compression will be suboptimal",
                    suggested_action="Apply k-means clustering (k=16 or 32) in TFMOT pipeline.",
                ))

        # ── Combination pass ─────────────────────────────────────────────
        self._check_combinations(model, report)

        # ── Model-level checks ───────────────────────────────────────────
        if all_weights:
            flat_all = np.concatenate([w.flatten() for w in all_weights])
            report.global_sparsity = float(np.mean(flat_all == 0))
            if report.global_sparsity < 0.5:
                report.findings.append(Finding(
                    rule_id="RULE-M03", layer_name="[model]", layer_type="Model",
                    severity="warning",
                    detail=f"Global sparsity={report.global_sparsity:.1%} — below 50% target",
                    suggested_action="Apply global pruning in TFMOT pipeline.",
                ))

        # DRAM risk: parameter memory > threshold
        param_bytes = report.total_params * 1  # INT8 = 1 byte
        if param_bytes / 1024 > self.SRAM_THRESHOLD_KB:
            report.has_dram_risk = True
            report.findings.append(Finding(
                rule_id="RULE-M03", layer_name="[model]", layer_type="Model",
                severity="warning",
                detail=(
                    f"Estimated weight memory ~{param_bytes//1024} KB exceeds "
                    f"on-chip SRAM threshold ({self.SRAM_THRESHOLD_KB} KB) — "
                    "DRAM spilling expected"
                ),
                suggested_action="Increase sparsity and clustering to reduce encoded weight size.",
            ))

        log.info("Analysis complete: %d violations, %d warnings",
                 len(report.violations), len(report.warnings))
        return report

    # ------------------------------------------------------------------
    def _check_combinations(self, model, report: AnalysisReport) -> None:
        """Check RULE-C01 through RULE-C05 on consecutive layer pairs."""
        from optimizer.rules import FOLDABLE_CONV_TYPES

        layers = model.layers

        for i, layer in enumerate(layers):
            ltype = type(layer).__name__
            prev = self._inbound(model, layer)

            # RULE-C01: unfused BN after Conv
            if ltype == "BatchNormalization" and prev:
                if type(prev).__name__ in FOLDABLE_CONV_TYPES:
                    report.findings.append(Finding(
                        rule_id="RULE-C01",
                        layer_name=layer.name,
                        layer_type=ltype,
                        severity="warning",
                        detail=f"BN follows {prev.name} ({type(prev).__name__}) — foldable offline",
                        suggested_action="Apply fold_batch_norm() transform before TFMOT pipeline.",
                    ))

            # RULE-C02: Conv → BN → activation not ready for Vela fusion
            if ltype == "BatchNormalization" and prev:
                if type(prev).__name__ in FOLDABLE_CONV_TYPES:
                    # Check what comes after this BN
                    next_layer = self._outbound(model, layer)
                    if next_layer:
                        n_type = type(next_layer).__name__
                        n_act = next_layer.get_config().get("activation", "")
                        if n_type not in ("ReLU", "Activation") and str(n_act) not in ("relu", "relu6"):
                            report.findings.append(Finding(
                                rule_id="RULE-C02",
                                layer_name=layer.name,
                                layer_type=ltype,
                                severity="info",
                                detail=f"Conv-BN sequence at {prev.name} not followed by ReLU/ReLU6 — Vela fusion blocked",
                                suggested_action="Ensure activation is relu/relu6 for Vela Conv-BN-ReLU fusion.",
                            ))

            # RULE-C04: two consecutive Dense layers without activation
            if ltype == "Dense" and prev and type(prev).__name__ == "Dense":
                prev_cfg = prev.get_config()
                prev_act = prev_cfg.get("activation", "linear")
                if str(prev_act) in ("linear", None):
                    report.findings.append(Finding(
                        rule_id="RULE-C04",
                        layer_name=layer.name,
                        layer_type=ltype,
                        severity="warning",
                        detail=f"Dense → Dense with no activation between {prev.name} and {layer.name} — mergeable",
                        suggested_action="Merge into single Dense layer using merge_linear_stack().",
                    ))

    # ------------------------------------------------------------------
    @staticmethod
    def _inbound(model, layer):
        """Return first inbound layer, or None."""
        try:
            nodes = layer._inbound_nodes
            if not nodes:
                return None
            inbound = nodes[0].inbound_layers
            if isinstance(inbound, list):
                return inbound[0] if inbound else None
            return inbound
        except Exception:
            return None

    @staticmethod
    def _outbound(model, layer):
        """Return first outbound layer, or None."""
        try:
            nodes = layer._outbound_nodes
            if not nodes:
                return None
            return nodes[0].outbound_layer
        except Exception:
            return None
