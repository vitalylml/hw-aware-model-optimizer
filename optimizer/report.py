"""
Report generation utilities.

Renders AnalysisReport and VelaReport as:
  - structured console output (via logging)
  - plain-text summary
  - HTML report saved to disk
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Console / text report
# ---------------------------------------------------------------------------

def print_analysis_report(report) -> None:
    """Pretty-print an AnalysisReport to the logger."""
    lines = [
        "",
        "╔══════════════════════════════════════════════════════╗",
        "║          ETHOS-U65 MODEL ANALYSIS REPORT             ║",
        "╚══════════════════════════════════════════════════════╝",
        f"  Model         : {report.model_path}",
        f"  Total params  : {report.total_params:,}",
        f"  Global sparsity: {report.global_sparsity:.1%}",
        f"  Transformer   : {report.is_transformer}",
        f"  DRAM risk     : {report.has_dram_risk}",
        f"  Violations    : {len(report.violations)}",
        f"  Warnings      : {len(report.warnings)}",
        "",
    ]

    if report.violations:
        lines.append("  VIOLATIONS (must fix before deployment):")
        for f in report.violations:
            lines.append(f"    [{f.rule_id}] {f.layer_name} ({f.layer_type})")
            lines.append(f"      ↳ {f.detail}")
            if f.suggested_action:
                lines.append(f"      ✦ {f.suggested_action}")

    if report.warnings:
        lines.append("")
        lines.append("  WARNINGS (strongly recommended):")
        for f in report.warnings:
            lines.append(f"    [{f.rule_id}] {f.layer_name} ({f.layer_type})")
            lines.append(f"      ↳ {f.detail}")

    if report.job_boundaries:
        lines.append("")
        lines.append("  EXPECTED NPU JOB BOUNDARIES:")
        for jb in report.job_boundaries:
            lines.append(f"    • {jb}")

    lines.append("")
    log.info("\n".join(lines))


def print_vela_report(report) -> None:
    """Pretty-print a VelaReport to the logger."""
    total_npu = max(1, report.cascaded_ops + report.uncascaded_ops)
    cascade_cov_pct = report.cascaded_ops / total_npu * 100.0
    lines = [
        "",
        "+======================================================+",
        "|              VELA COMPILATION REPORT                 |",
        "+======================================================+",
        f"  Input tflite     : {report.tflite_path}",
        f"  Output tflite    : {report.vela_tflite_path}",
        f"  NPU layers       : {report.npu_layers}",
        f"  CPU fallbacks    : {report.cpu_layers}",
        f"  Total cycles     : {report.total_cycles:,}",
        f"  Mean MAC util    : {report.mean_util_pct:.1f}%",
        f"  NPU job count    : {report.job_count}",
        f"  Cascade groups   : {report.cascade_groups}",
        f"  Cascade coverage : {report.cascaded_ops}/{report.cascaded_ops + report.uncascaded_ops} "
        f"ops ({cascade_cov_pct:.1f}%)",
        "",
    ]

    if report.memory_bound_layers:
        lines.append("  MEMORY-BOUND LAYERS (increase sparsity/clustering):")
        for name in report.memory_bound_layers[:10]:
            lines.append(f"    - {name}")
        if len(report.memory_bound_layers) > 10:
            lines.append(f"    ... and {len(report.memory_bound_layers) - 10} more")

    if report.cpu_layers > 0:
        fallback_names = [s.name for s in report.layer_stats if not s.on_npu]
        lines.append("")
        lines.append("  CPU FALLBACK LAYERS (RULE-L05 violations confirmed):")
        for name in fallback_names:
            lines.append(f"    - {name}")

    if report.cascades:
        lines.append("")
        lines.append("  CASCADE GROUPS (Vela scheduler):")
        for c in report.cascades[:10]:
            lines.append(
                f"    - #{c.index}  {c.start} -> {c.end}  "
                f"({c.op_count} ops, {c.mem_usage_bytes:,} B SRAM)"
            )
        if len(report.cascades) > 10:
            lines.append(f"    ... and {len(report.cascades) - 10} more")
    elif report.uncascaded_ops > 0:
        lines.append("")
        lines.append(
            "  Vela did not create any cascade groups; all NPU operators "
            "execute independently."
        )

    lines.append("")
    log.info("\n".join(lines))


def print_vela_comparison(before, after) -> None:
    """Print a side-by-side comparison of two Vela reports (pre- vs post-transform)."""
    from optimizer.exporter import _vela_cascading_metrics

    def _fmt(value, fmt: str) -> str:
        return fmt.format(value)

    def _row(label: str, b, a, fmt: str = "{:,}") -> str:
        b_s = _fmt(b, fmt)
        a_s = _fmt(a, fmt)
        if isinstance(b, (int, float)) and isinstance(a, (int, float)) and b not in (0, 0.0):
            d_pct = (a - b) / b * 100.0
            sign = "+" if d_pct >= 0 else ""
            delta = f"({sign}{d_pct:.1f}%)"
        else:
            delta = ""
        return f"  {label:<22}  {b_s:>14}    ->    {a_s:>14}  {delta}"

    cm_b = _vela_cascading_metrics(before)
    cm_a = _vela_cascading_metrics(after)

    lines = [
        "",
        "+==========================================================+",
        "|      VELA PERFORMANCE COMPARISON (before / after)        |",
        "+==========================================================+",
        f"  Before tflite : {before.tflite_path}",
        f"  After  tflite : {after.tflite_path}",
        "",
        f"  {'metric':<22}  {'before':>14}          {'after':>14}",
        "  -- Throughput / efficiency --",
        _row("NPU layers",          before.npu_layers,    after.npu_layers),
        _row("CPU fallbacks",       before.cpu_layers,    after.cpu_layers),
        _row("Total cycles",        before.total_cycles,  after.total_cycles),
        _row("Inference ms",        cm_b["inference_ms"], cm_a["inference_ms"], fmt="{:.3f}"),
        _row("Mean MAC util %",     before.mean_util_pct, after.mean_util_pct,  fmt="{:.1f}"),
        _row("NPU job count",       before.job_count,     after.job_count),
        _row("Memory-bound",
             len(before.memory_bound_layers),
             len(after.memory_bound_layers)),
        "  -- Cascading footprint --",
        _row("Passes after fusing", cm_b["passes_after_fusing"], cm_a["passes_after_fusing"]),
        _row("SRAM peak (KiB)",     cm_b["sram_peak_kb"],   cm_a["sram_peak_kb"],   fmt="{:,.1f}"),
        _row("SRAM bandwidth (MB)", cm_b["sram_bw_mb"],     cm_a["sram_bw_mb"],     fmt="{:,.2f}"),
        _row("DRAM bandwidth (MB)", cm_b["dram_bw_mb"],     cm_a["dram_bw_mb"],     fmt="{:,.2f}"),
        _row("Weight DRAM (KiB)",   cm_b["weight_dram_kb"], cm_a["weight_dram_kb"], fmt="{:,.1f}"),
        "  -- Cascade structure (Vela scheduler) --",
        _row("Cascade groups",      before.cascade_groups,  after.cascade_groups),
        _row("Cascaded ops",        before.cascaded_ops,    after.cascaded_ops),
        _row("Uncascaded ops",      before.uncascaded_ops,  after.uncascaded_ops),
        "",
    ]

    def _render_cascades(label: str, cascades) -> list[str]:
        if not cascades:
            return [f"  Cascade groups ({label}): none"]
        out = [f"  Cascade groups ({label}):"]
        for c in cascades[:5]:
            out.append(
                f"    - #{c.index}  {c.start} -> {c.end}  "
                f"({c.op_count} ops, {c.mem_usage_bytes:,} B SRAM)"
            )
        if len(cascades) > 5:
            out.append(f"    ... and {len(cascades) - 5} more")
        return out

    if before.cascades or after.cascades:
        lines.extend(_render_cascades("before", before.cascades))
        lines.extend(_render_cascades("after",  after.cascades))
        lines.append("")

    log.info("\n".join(lines))


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def save_html_report(
    analysis_report,
    vela_report=None,
    output_path: str | Path = "optimization_report.html",
) -> Path:
    """
    Save a self-contained HTML report combining analysis and Vela results.
    """
    output_path = Path(output_path)

    findings_rows = ""
    for f in analysis_report.findings:
        sev_color = {
            "violation": "#FCEBEB",
            "warning": "#FAEEDA",
            "info": "#E6F1FB",
        }.get(f.severity, "#F1EFE8")
        sev_text_color = {
            "violation": "#A32D2D",
            "warning": "#854F0B",
            "info": "#0C447C",
        }.get(f.severity, "#2C2C2A")
        findings_rows += f"""
        <tr style="background:{sev_color}">
          <td style="color:{sev_text_color};font-weight:500">{f.rule_id}</td>
          <td style="color:{sev_text_color}">{f.severity.upper()}</td>
          <td>{f.layer_name}</td>
          <td>{f.layer_type}</td>
          <td>{f.detail}</td>
          <td style="color:#378ADD">{f.suggested_action}</td>
        </tr>"""

    layer_rows = ""
    for layer in analysis_report.layers:
        layer_rows += f"""
        <tr>
          <td>{layer.name}</td>
          <td><span style="background:#E6F1FB;color:#0C447C;padding:2px 8px;border-radius:20px;font-size:12px">{layer.layer_type}</span></td>
          <td style="font-family:monospace;font-size:12px">{layer.output_shape or '—'}</td>
          <td>{layer.param_count:,}</td>
          <td>{layer.sparsity:.1%}</td>
          <td>{layer.unique_weights:,}</td>
        </tr>"""

    vela_section = ""
    if vela_report:
        vela_section = f"""
        <h2>Vela compilation results</h2>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px">
          <div class="metric"><div class="ml">NPU layers</div><div class="mv">{vela_report.npu_layers}</div></div>
          <div class="metric"><div class="ml">CPU fallbacks</div><div class="mv" style="color:#A32D2D">{vela_report.cpu_layers}</div></div>
          <div class="metric"><div class="ml">Total cycles</div><div class="mv">{vela_report.total_cycles:,}</div></div>
          <div class="metric"><div class="ml">Mean MAC util</div><div class="mv">{vela_report.mean_util_pct:.1f}%</div></div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Ethos-U65 Optimization Report</title>
<style>
  body {{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;padding:32px;color:#2C2C2A;background:#F1EFE8;line-height:1.6}}
  h1   {{font-size:22px;font-weight:500;margin-bottom:4px}}
  h2   {{font-size:16px;font-weight:500;margin:24px 0 12px;color:#2C2C2A}}
  .sub {{font-size:13px;color:#5F5E5A;margin-bottom:24px}}
  .metrics {{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}}
  .metric {{background:white;border-radius:10px;padding:14px 16px;border:0.5px solid #D3D1C7}}
  .ml  {{font-size:11px;color:#888780}}
  .mv  {{font-size:20px;font-weight:500;margin-top:2px}}
  table {{width:100%;border-collapse:collapse;background:white;border-radius:10px;overflow:hidden;border:0.5px solid #D3D1C7;margin-bottom:24px}}
  th   {{background:#F1EFE8;font-size:12px;font-weight:500;text-align:left;padding:10px 14px;border-bottom:0.5px solid #D3D1C7}}
  td   {{font-size:13px;padding:9px 14px;border-bottom:0.5px solid #F1EFE8}}
  tr:last-child td {{border-bottom:none}}
</style>
</head>
<body>
<h1>Ethos-U65 Model Optimization Report</h1>
<div class="sub">Model: {analysis_report.model_path}</div>

<h2>Model summary</h2>
<div class="metrics">
  <div class="metric"><div class="ml">Total parameters</div><div class="mv">{analysis_report.total_params:,}</div></div>
  <div class="metric"><div class="ml">Global sparsity</div><div class="mv">{analysis_report.global_sparsity:.1%}</div></div>
  <div class="metric"><div class="ml">Violations</div><div class="mv" style="color:#A32D2D">{len(analysis_report.violations)}</div></div>
  <div class="metric"><div class="ml">Warnings</div><div class="mv" style="color:#854F0B">{len(analysis_report.warnings)}</div></div>
</div>

{vela_section}

<h2>Findings ({len(analysis_report.findings)} total)</h2>
<table>
  <tr><th>Rule</th><th>Severity</th><th>Layer</th><th>Type</th><th>Detail</th><th>Action</th></tr>
  {findings_rows}
</table>

<h2>Layer inventory ({len(analysis_report.layers)} layers)</h2>
<table>
  <tr><th>Name</th><th>Type</th><th>Output shape</th><th>Params</th><th>Sparsity</th><th>Unique weights</th></tr>
  {layer_rows}
</table>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    log.info("HTML report saved to %s", output_path)
    return output_path
