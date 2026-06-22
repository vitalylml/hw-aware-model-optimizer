"""
Phase 4 — TFLite INT8 export
Phase 5 — Vela NPU compilation

Phase 4 converts the PCQAT model to a fully INT8 TFLite flatbuffer.
Phase 5 runs the Vela compiler against the .tflite and parses the summary CSV
to update rule parameters for the next optimization iteration.
"""
from __future__ import annotations

import csv
import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generator, Iterator, List, Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 4 — TFLite INT8 export
# ---------------------------------------------------------------------------

def export_tflite_int8(
    model,
    representative_dataset_fn: Callable[[], Iterator],
    output_path: str | Path = "model_optimized.tflite",
    force_full_int8: bool = True,
) -> Path:
    """
    Convert a (PCQAT-wrapped) Keras model to a fully INT8 TFLite flatbuffer.

    Parameters
    ----------
    model                   : PCQAT model from Phase 3 (or any Keras model)
    representative_dataset_fn : callable yielding batches of representative inputs;
                               shape [1, H, W, C] as float32; ~100-200 samples
    output_path             : destination .tflite path
    force_full_int8         : if True, set both input/output dtypes to INT8

    Returns
    -------
    Path to the exported .tflite file
    """
    import tensorflow as tf

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Phase 4: converting to INT8 TFLite → %s", output_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_fn
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    if force_full_int8:
        converter.inference_input_type  = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)

    size_kb = len(tflite_model) / 1024
    log.info("Phase 4: exported  %s  (%.1f KB)", output_path, size_kb)
    return output_path


def make_representative_dataset(
    samples: np.ndarray,
    batch_size: int = 1,
) -> Callable[[], Generator]:
    """
    Build a representative dataset callable from a numpy array of samples.

    Parameters
    ----------
    samples    : numpy array of shape (N, H, W, C) or (N, D), float32
    batch_size : must be 1 for Ethos-U65 (RULE-M05)

    Returns
    -------
    Callable suitable for converter.representative_dataset
    """
    def representative_dataset_fn() -> Generator:
        for i in range(len(samples)):
            batch = samples[i : i + batch_size].astype(np.float32)
            yield [batch]

    return representative_dataset_fn


# ---------------------------------------------------------------------------
# Phase 5 — Vela compilation
# ---------------------------------------------------------------------------

@dataclass
class VelaConfig:
    """Configuration for the Vela NPU compiler."""
    accelerator_config: str = "ethos-u65-512"     # ethos-u65-256 or ethos-u65-512
    optimise: str = "Performance"                  # Performance | Size
    memory_mode: str = "Shared_Sram"               # Shared_Sram | Dedicated_Sram | SRAM_Only
    system_config: str = "Ethos_U65_High_End"
    output_dir: str = "./vela_output"
    extra_args: list[str] = field(default_factory=list)


@dataclass
class VelaLayerStat:
    name: str
    on_npu: bool
    operator: str
    cycles: int = 0
    util_pct: float = 0.0
    memory_bound: bool = False


@dataclass
class VelaCascade:
    """One cascade group reported by the Vela scheduler."""
    index: int
    start: str
    end: str
    mem_usage_bytes: int = 0
    op_count: int = 0


@dataclass
class VelaReport:
    tflite_path: str
    vela_tflite_path: str
    summary_csv: str
    npu_layers: int = 0
    cpu_layers: int = 0
    total_cycles: int = 0
    mean_util_pct: float = 0.0
    memory_bound_layers: List[str] = field(default_factory=list)
    layer_stats: List[VelaLayerStat] = field(default_factory=list)
    raw_summary: dict = field(default_factory=dict)
    # Vela scheduler cascade information (populated from --verbose-schedule stdout)
    cascade_groups: int = 0
    cascaded_ops: int = 0
    uncascaded_ops: int = 0
    cascades: List[VelaCascade] = field(default_factory=list)

    @property
    def job_count(self) -> int:
        """Number of NPU jobs (1 = no fragmentation)."""
        return max(1, self.cpu_layers)

    def summary(self) -> str:
        lines = [
            f"Vela output      : {self.vela_tflite_path}",
            f"NPU layers       : {self.npu_layers}",
            f"CPU fallback     : {self.cpu_layers}",
            f"Total cycles     : {self.total_cycles:,}",
            f"Mean MAC util    : {self.mean_util_pct:.1f}%",
            f"Memory-bound     : {len(self.memory_bound_layers)} layers",
            f"Fragmentation    : {self.job_count} NPU job(s)",
            f"Cascade groups   : {self.cascade_groups} "
            f"(cascaded ops: {self.cascaded_ops} / "
            f"uncascaded ops: {self.uncascaded_ops})",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "tflite_path": self.tflite_path,
            "vela_tflite_path": self.vela_tflite_path,
            "npu_layers": self.npu_layers,
            "cpu_layers": self.cpu_layers,
            "total_cycles": self.total_cycles,
            "mean_util_pct": self.mean_util_pct,
            "memory_bound_layers": self.memory_bound_layers,
            "job_count": self.job_count,
            "cascade_groups": self.cascade_groups,
            "cascaded_ops": self.cascaded_ops,
            "uncascaded_ops": self.uncascaded_ops,
            "cascades": [
                {
                    "index": c.index,
                    "start": c.start,
                    "end": c.end,
                    "mem_usage_bytes": c.mem_usage_bytes,
                    "op_count": c.op_count,
                }
                for c in self.cascades
            ],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def compile_with_vela(
    tflite_path: str | Path,
    config: Optional[VelaConfig] = None,
) -> VelaReport:
    """
    Run the Vela compiler and return a VelaReport with per-layer stats.

    Requires `vela` to be installed:  pip install ethos-u-vela

    Parameters
    ----------
    tflite_path : path to the INT8 .tflite file from Phase 4
    config      : VelaConfig; defaults used if None

    Returns
    -------
    VelaReport with compiled .tflite path and performance stats
    """
    cfg = config or VelaConfig()
    tflite_path = Path(tflite_path)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "vela", str(tflite_path),
        "--accelerator-config", cfg.accelerator_config,
        "--optimise", cfg.optimise,
        "--memory-mode", cfg.memory_mode,
        "--system-config", cfg.system_config,
        "--output-dir", str(output_dir),
        # --verbose-performance is required for Vela to emit the
        # <stem>_per-layer.csv used to populate per-operator stats.
        "--verbose-performance",
        # --verbose-schedule prints per-operator scheduler info (including
        # "Assigned Cascade = N") and a "Cascades:" section listing the
        # cascade groups the scheduler created.
        "--verbose-schedule",
    ] + cfg.extra_args

    log.info("Phase 5: running Vela: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        )
        log.debug("Vela stdout:\n%s", result.stdout)
    except subprocess.CalledProcessError as e:
        log.error("Vela compilation failed:\n%s\n%s", e.stdout, e.stderr)
        raise
    except FileNotFoundError:
        raise RuntimeError(
            "Vela not found. Install with:  pip install ethos-u-vela"
        )

    vela_tflite = output_dir / (tflite_path.stem + "_vela.tflite")

    # Persist Vela's stdout/stderr alongside the artefacts so cascade and
    # schedule information is available for direct inspection even after
    # parsing. This is the source of truth for cascade-section debugging.
    stdout_path = output_dir / f"{tflite_path.stem}_vela_stdout.txt"
    try:
        stdout_path.write_text(result.stdout or "", encoding="utf-8")
    except Exception as e:
        log.warning("Could not write Vela stdout to %s: %s", stdout_path, e)

    # Vela names the summary CSV "<stem>_summary_<system_config>.csv".
    summary_candidates = sorted(output_dir.glob(f"{tflite_path.stem}_summary*.csv"))
    summary_csv = summary_candidates[0] if summary_candidates else None
    per_layer_csv = output_dir / f"{tflite_path.stem}_per-layer.csv"

    report = VelaReport(
        tflite_path=str(tflite_path),
        vela_tflite_path=str(vela_tflite),
        summary_csv=str(summary_csv) if summary_csv else "",
    )

    _parse_vela_stdout(result.stdout, report)
    _parse_vela_cascades(result.stdout, report)

    if summary_csv is not None:
        _parse_vela_summary(summary_csv, report)
    else:
        log.warning("Vela summary CSV not found in %s", output_dir)

    if per_layer_csv.exists():
        _parse_vela_per_layer(per_layer_csv, report)
    else:
        log.warning("Vela per-layer CSV not found at %s", per_layer_csv)

    log.info("Phase 5 complete:\n%s", report.summary())
    return report


def _vela_cascading_metrics(report: VelaReport) -> dict:
    """
    Extract cascading-relevant aggregates from a Vela summary row.

    All values are pulled from VelaReport.raw_summary (the parsed Vela summary
    CSV). Missing keys default to 0 so a comparison can be produced even when
    Vela is not available for one side.
    """
    raw = report.raw_summary or {}

    def _f(key: str, default: float = 0.0) -> float:
        try:
            return float(raw.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    return {
        # peak SRAM allocation (KiB, already in KiB in the Vela CSV)
        "sram_peak_kb":        _f("sram_memory_used"),
        # total SRAM traffic per inference (MB)
        "sram_bw_mb":          _f("sram_total_bytes") / (1024.0 * 1024.0),
        # total DRAM traffic per inference (MB)
        "dram_bw_mb":          _f("dram_total_bytes") / (1024.0 * 1024.0),
        # weight reads from DRAM (KiB)
        "weight_dram_kb":      _f("dram_weight_read_bytes") / 1024.0,
        # wall-clock estimate per inference (ms)
        "inference_ms":        _f("inference_time") * 1000.0,
        # number of operator passes after Vela's fusing — proxy for cascade groups
        "passes_after_fusing": int(_f("passes_after_fusing")),
    }


def compare_vela_reports(before: VelaReport, after: VelaReport) -> dict:
    """
    Build a side-by-side comparison of two Vela reports.

    Intended for measuring the impact of a static transform on NPU performance:
    the same calibration data is used to export both Keras models to INT8 TFLite,
    both are compiled with the same Vela configuration, and the resulting reports
    are reduced to a set of comparable metrics covering throughput, MAC efficiency,
    and cascading footprint (SRAM peak, SRAM/DRAM bandwidth, weight DRAM traffic,
    post-fusing pass count).
    """
    def _pct_delta(b: float, a: float) -> float:
        if b == 0:
            return 0.0
        return (a - b) / b * 100.0

    def _summary(r: VelaReport) -> dict:
        cm = _vela_cascading_metrics(r)
        return {
            "tflite_path":         r.tflite_path,
            "npu_layers":          r.npu_layers,
            "cpu_layers":          r.cpu_layers,
            "total_cycles":        r.total_cycles,
            "mean_util_pct":       r.mean_util_pct,
            "job_count":           r.job_count,
            "memory_bound":        len(r.memory_bound_layers),
            # cascading metrics
            "sram_peak_kb":        cm["sram_peak_kb"],
            "sram_bw_mb":          cm["sram_bw_mb"],
            "dram_bw_mb":          cm["dram_bw_mb"],
            "weight_dram_kb":      cm["weight_dram_kb"],
            "inference_ms":        cm["inference_ms"],
            "passes_after_fusing": cm["passes_after_fusing"],
            # cascade structure
            "cascade_groups":      r.cascade_groups,
            "cascaded_ops":        r.cascaded_ops,
            "uncascaded_ops":      r.uncascaded_ops,
            "cascades": [
                {
                    "index": c.index,
                    "start": c.start,
                    "end": c.end,
                    "mem_usage_bytes": c.mem_usage_bytes,
                    "op_count": c.op_count,
                }
                for c in r.cascades
            ],
        }

    before_s = _summary(before)
    after_s  = _summary(after)

    return {
        "before": before_s,
        "after":  after_s,
        "delta": {
            "npu_layers":          after.npu_layers    - before.npu_layers,
            "cpu_layers":          after.cpu_layers    - before.cpu_layers,
            "total_cycles":        after.total_cycles  - before.total_cycles,
            "cycles_pct":          _pct_delta(before.total_cycles, after.total_cycles),
            "mean_util_pct":       after.mean_util_pct - before.mean_util_pct,
            "job_count":           after.job_count     - before.job_count,
            "memory_bound":        len(after.memory_bound_layers) - len(before.memory_bound_layers),
            "sram_peak_kb":        after_s["sram_peak_kb"]        - before_s["sram_peak_kb"],
            "sram_peak_pct":       _pct_delta(before_s["sram_peak_kb"],   after_s["sram_peak_kb"]),
            "sram_bw_mb":          after_s["sram_bw_mb"]          - before_s["sram_bw_mb"],
            "sram_bw_pct":         _pct_delta(before_s["sram_bw_mb"],     after_s["sram_bw_mb"]),
            "dram_bw_mb":          after_s["dram_bw_mb"]          - before_s["dram_bw_mb"],
            "dram_bw_pct":         _pct_delta(before_s["dram_bw_mb"],     after_s["dram_bw_mb"]),
            "weight_dram_kb":      after_s["weight_dram_kb"]      - before_s["weight_dram_kb"],
            "weight_dram_pct":     _pct_delta(before_s["weight_dram_kb"], after_s["weight_dram_kb"]),
            "inference_ms":        after_s["inference_ms"]        - before_s["inference_ms"],
            "inference_pct":       _pct_delta(before_s["inference_ms"],   after_s["inference_ms"]),
            "passes_after_fusing": after_s["passes_after_fusing"] - before_s["passes_after_fusing"],
            "cascade_groups":      after.cascade_groups - before.cascade_groups,
            "cascaded_ops":        after.cascaded_ops   - before.cascaded_ops,
            "uncascaded_ops":      after.uncascaded_ops - before.uncascaded_ops,
        },
    }


def _parse_vela_stdout(stdout: str, report: VelaReport) -> None:
    """
    Extract CPU and NPU operator counts from Vela stdout.

    Vela prints lines such as:
        CPU operators = 0 (0.0%)
        NPU operators = 137 (100.0%)
    The summary CSV does not break operators down by placement, so this is the
    only reliable source for those counts.
    """
    import re
    cpu_re = re.compile(r"^\s*CPU operators\s*=\s*(\d+)")
    npu_re = re.compile(r"^\s*NPU operators\s*=\s*(\d+)")
    for line in stdout.splitlines():
        m = cpu_re.match(line)
        if m:
            report.cpu_layers = int(m.group(1))
            continue
        m = npu_re.match(line)
        if m:
            report.npu_layers = int(m.group(1))


def _parse_vela_cascades(stdout: str, report: VelaReport) -> None:
    """
    Extract cascade-group information from Vela's --verbose-schedule output.

    Vela emits two pieces of cascade information:

    1. Per scheduled operator, a line of the form::

           Assigned Cascade = <id>

       Cascade id ``0`` means the operator is *not* part of any cascade group;
       any positive id marks membership of that group.

    2. After the per-op block, a section starting with ``Cascades:`` whose
       entries take the form::

           <index>: <start_op> -> <end_op>, size: <mem_usage_bytes>

       For a network where the scheduler chose not to cascade (every op at
       id ``0``), this section is empty.

    The parser populates:
      - ``report.cascade_groups``      - count of distinct non-zero cascade ids
      - ``report.cascaded_ops``        - number of ops with non-zero cascade id
      - ``report.uncascaded_ops``      - number of ops with cascade id == 0
      - ``report.cascades``            - list[VelaCascade], one per group
    """
    import re
    from collections import Counter

    assigned_re = re.compile(r"^\s*Assigned Cascade\s*=\s*(\d+)\s*$")
    section_re  = re.compile(r"^\s*Cascades:\s*$")
    entry_re    = re.compile(
        r"^\s*(?P<idx>\d+)\s*:\s*"
        r"(?P<start>\S+)\s*->\s*(?P<end>\S+)\s*,\s*"
        r"size\s*:\s*(?P<mem>\d+)\s*$"
    )

    cascade_counts: Counter = Counter()
    in_cascades_section = False
    cascades: List[VelaCascade] = []

    for line in stdout.splitlines():
        if in_cascades_section:
            m = entry_re.match(line)
            if m:
                cascades.append(VelaCascade(
                    index=int(m.group("idx")),
                    start=m.group("start"),
                    end=m.group("end"),
                    mem_usage_bytes=int(m.group("mem")),
                ))
                continue
            # First non-matching, non-blank line ends the section.
            if line.strip():
                in_cascades_section = False
            else:
                continue

        if section_re.match(line):
            in_cascades_section = True
            continue

        m = assigned_re.match(line)
        if m:
            cascade_counts[int(m.group(1))] += 1

    report.uncascaded_ops = int(cascade_counts.get(0, 0))
    report.cascaded_ops   = int(sum(n for cid, n in cascade_counts.items() if cid != 0))
    report.cascade_groups = int(sum(1 for cid in cascade_counts if cid != 0))

    # Attach per-cascade op counts from the assigned-cascade tally.
    for cascade in cascades:
        cascade.op_count = int(cascade_counts.get(cascade.index, 0))
    report.cascades = cascades


def _parse_vela_summary(csv_path: Path, report: VelaReport) -> None:
    """
    Parse the Vela summary CSV (one aggregate row).

    The file is named ``<stem>_summary_<system_config>.csv`` and contains a
    single data row of network-wide metrics. Only the fields required by the
    pipeline are extracted; the full row is preserved in ``raw_summary`` for
    callers that need additional values.
    """
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            log.warning("Vela summary CSV %s is empty", csv_path)
            return

        row = rows[0]
        report.raw_summary = dict(row)
        report.total_cycles = int(float(row.get("cycles_total", 0) or 0))

    except Exception as e:
        log.warning("Could not parse Vela summary CSV: %s", e)


def _parse_vela_per_layer(csv_path: Path, report: VelaReport) -> None:
    """
    Parse Vela's per-layer CSV (emitted with --verbose-performance).

    Populates report.layer_stats with one VelaLayerStat per scheduled NPU
    operator, computes the cycle-weighted mean MAC utilization, and flags
    memory-bound operators (Util%(MAC) < 50 with non-zero MACs).

    The CSV only contains NPU subgraphs, so on_npu is True for every row;
    CPU layer counts are sourced separately from stdout.
    """
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        util_num = 0.0
        util_den = 0.0
        for row in rows:
            cycles = int(float(row.get("Op Cycles", 0) or 0))
            util_pct = float(row.get("Util% (MAC)", 0) or 0)
            macs = int(float(row.get("MAC Count", 0) or 0))

            stat = VelaLayerStat(
                name=row.get("Name", ""),
                on_npu=True,
                operator=row.get("NNG Operator", ""),
                cycles=cycles,
                util_pct=util_pct,
                memory_bound=(util_pct < 50.0 and macs > 0),
            )
            report.layer_stats.append(stat)
            if stat.memory_bound:
                report.memory_bound_layers.append(stat.name)

            if macs > 0 and cycles > 0:
                util_num += util_pct * cycles
                util_den += cycles

        report.mean_util_pct = float(util_num / util_den) if util_den > 0 else 0.0

    except Exception as e:
        log.warning("Could not parse Vela per-layer CSV: %s", e)


# ---------------------------------------------------------------------------
# Rule update from Vela report (feeds next iteration)
# ---------------------------------------------------------------------------

def update_rules_from_vela(
    vela_report: VelaReport,
    current_rules: dict,
) -> dict:
    """
    Adjust rule parameters based on measured Vela performance.

    Current adjustments:
    - If mean MAC utilization < 50%: recommend 256-MAC config (model is memory-bound)
    - If memory-bound layers exist: increase sparsity target in RULE-L03/M03
    - If CPU fallback layers exist: add them to RULE-L05 known_fallbacks
    - If job_count > 1: flag RULE-M02 attention risk confirmed
    - If memory pressure or fallback boundaries exist: recommend stronger NPU cascading

    Returns updated rules dict (does not mutate input).
    """
    from copy import deepcopy
    updated = deepcopy(current_rules)

    # Memory-bound → increase sparsity target
    if vela_report.memory_bound_layers:
        for rule_id in ("RULE-L03", "RULE-M03"):
            if rule_id in updated:
                current_target = updated[rule_id].params.get("target_sparsity", 0.70)
                new_target = min(current_target + 0.05, 0.85)
                updated[rule_id].params["target_sparsity"] = new_target
                log.info(
                    "Rule update: %s target_sparsity %.2f → %.2f "
                    "(memory-bound layers detected)",
                    rule_id, current_target, new_target,
                )

    # Low mean utilization → recommend 256 MACs
    if vela_report.mean_util_pct < 50 and vela_report.npu_layers > 0:
        updated["RULE-DEPLOY-MAC"] = {
            "id": "RULE-DEPLOY-MAC",
            "generated": True,
            "recommendation": (
                f"Mean MAC utilization={vela_report.mean_util_pct:.1f}% < 50%. "
                "Model is memory-bound — 256-MAC configuration is sufficient. "
                "512-MAC adds area/power cost with negligible throughput gain."
            ),
        }
        log.info("Rule update: generated RULE-DEPLOY-MAC (low MAC utilization)")

    # CPU fallbacks → update RULE-L05
    if vela_report.cpu_layers > 0:
        fallback_ops = list({s.operator for s in vela_report.layer_stats if not s.on_npu})
        if "RULE-L05" in updated:
            updated["RULE-L05"].params["known_fallbacks"] = fallback_ops
            log.info("Rule update: RULE-L05 known_fallbacks = %s", fallback_ops)

    # Multiple jobs → confirm RULE-M02
    if vela_report.job_count > 1:
        if "RULE-M02" in updated:
            updated["RULE-M02"].params["confirmed_job_boundaries"] = (
                [s.name for s in vela_report.layer_stats if not s.on_npu]
            )
            log.info(
                "Rule update: RULE-M02 confirmed %d job boundaries",
                vela_report.job_count,
            )

    # NPU cascading evidence (RULE-M06)
    # The implementation of RULE-M06 lives in
    # optimizer.transforms.apply_cascading_optimizations and runs in Phase 2.
    # Here we only attach Vela-derived evidence to the rule and emit a deploy
    # advisory when memory pressure or fallback boundaries remain after
    # transformation.
    if (
        vela_report.memory_bound_layers
        or vela_report.cpu_layers > 0
        or vela_report.job_count > 1
    ):
        if "RULE-M06" in updated:
            updated["RULE-M06"].params["measured"] = {
                "memory_bound_layers": list(vela_report.memory_bound_layers),
                "job_count": int(vela_report.job_count),
                "cpu_fallback_layers": [
                    s.name for s in vela_report.layer_stats if not s.on_npu
                ],
            }
            log.info(
                "Rule update: RULE-M06 evidence recorded (memory_bound=%d, cpu_layers=%d, jobs=%d)",
                len(vela_report.memory_bound_layers),
                vela_report.cpu_layers,
                vela_report.job_count,
            )

        updated["RULE-DEPLOY-CASCADE"] = {
            "id": "RULE-DEPLOY-CASCADE",
            "generated": True,
            "recommendation": (
                "Residual memory pressure or fallback boundaries detected after RULE-M06 "
                "activation fusion. Investigate remaining CPU operators and consider "
                "structural changes (Conv/Depthwise/pointwise sequencing) to widen cascade "
                "regions further."
            ),
            "evidence": {
                "memory_bound_layers": list(vela_report.memory_bound_layers),
                "cpu_layers": int(vela_report.cpu_layers),
                "job_count": int(vela_report.job_count),
            },
        }
        log.info("Rule update: generated RULE-DEPLOY-CASCADE")

    return updated
