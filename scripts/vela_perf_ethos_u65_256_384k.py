"""
Vela performance extraction for Ethos-U65-256, Dedicated_Sram 384 KB.

Usage
-----
    python scripts/vela_perf_ethos_u65_256_384k.py <model.tflite>

Returns a JSON dict (printed to stdout) with structured performance metrics.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

HW_ID = "Ethos-u65-256_Dedicated_Sram_384KB"

_VELA_ARGS = [
    "--system-config",      "Ethos_U65_High_End",
    "--memory-mode",        "Dedicated_Sram_384KB",
    "--accelerator-config", "ethos-u65-256",
    "--optimise",           "Performance",
    "--verbose-cycle-estimate",
]

# On Windows the vela.exe console-script wrapper is unreliable; invoke via
# the current Python interpreter's -m entry point instead.
_VELA_CMD = [sys.executable, "-m", "ethosu.vela"]


def run_vela(tflite_path: Path, output_dir: Path) -> str:
    """Run Vela and return stdout. Raises RuntimeError on failure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = _VELA_CMD + [str(tflite_path), "--output-dir", str(output_dir)] + _VELA_ARGS
    log.debug("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Vela failed (exit {exc.returncode}):\n{exc.stdout}\n{exc.stderr}"
        ) from exc
    except FileNotFoundError:
        raise RuntimeError("ethosu.vela not found. Install with: pip install ethos-u-vela")
    return result.stdout


def _parse_batch_inference_time(stdout: str) -> float:
    """
    Extract batch inference time in microseconds from --verbose-cycle-estimate output.

    Vela prints a line such as:
        Batch Inference time                 0.04 ms, 25895.33 inferences/s (batch size 1)
    or (older versions):
        Batch Inference time  =  1234.56 us, 1234560.00 cycles @ 1000 MHz
    Falls back to deriving from total cycles at 1 GHz if the line is absent.
    """
    pattern = re.compile(
        r"[Bb]atch\s+[Ii]nference\s+time\s*[=:]?\s*([\d.]+)\s*(ms|us)",
        re.IGNORECASE,
    )
    for line in stdout.splitlines():
        m = pattern.search(line)
        if m:
            value = float(m.group(1))
            unit = m.group(2).lower()
            return value * 1000.0 if unit == "ms" else value
    return 0.0


def _parse_total_cycles(stdout: str) -> int:
    """
    Extract total cycle count from Vela stdout.

    Vela prints a line such as:
        Total cycles  =  1234560
    or (from the cycle-estimate table):
        Total          1234560
    """
    patterns = [
        re.compile(r"[Tt]otal\s+cycles\s*[=:]\s*([\d,]+)"),
        re.compile(r"^\s*Total\s+([\d,]+)\s*$"),
    ]
    for line in stdout.splitlines():
        for pat in patterns:
            m = pat.search(line)
            if m:
                return int(m.group(1).replace(",", ""))
    return 0


def _parse_npu_utilisation(stdout: str) -> float:
    """
    Extract NPU utilisation percentage.

    Vela prints a line such as:
        NPU active  =  85.3%
    or:
        MAC utilisation   85.3 %
    """
    patterns = [
        re.compile(r"NPU\s+active\s*[=:]\s*([\d.]+)\s*%", re.IGNORECASE),
        re.compile(r"MAC\s+utilis[ae]tion\s+([\d.]+)\s*%", re.IGNORECASE),
    ]
    for line in stdout.splitlines():
        for pat in patterns:
            m = pat.search(line)
            if m:
                return float(m.group(1))
    return 0.0


def _parse_operator_counts(stdout: str) -> tuple[int, int]:
    """Return (npu_layers, cpu_layers) from Vela stdout."""
    cpu_re = re.compile(r"^\s*CPU operators\s*=\s*(\d+)")
    npu_re = re.compile(r"^\s*NPU operators\s*=\s*(\d+)")
    npu, cpu = 0, 0
    for line in stdout.splitlines():
        m = cpu_re.match(line)
        if m:
            cpu = int(m.group(1))
            continue
        m = npu_re.match(line)
        if m:
            npu = int(m.group(1))
    return npu, cpu


def measure(tflite_path: str | Path, output_dir: str | Path | None = None) -> dict:
    """
    Run Vela on *tflite_path* and return structured performance metrics.

    Parameters
    ----------
    tflite_path : path to the INT8 .tflite file
    output_dir  : directory for Vela output artefacts; defaults to
                  <tflite_stem>_vela_256_384k/ next to the input file

    Returns
    -------
    dict with keys:
        hw_id, tflite_path, batch_inference_time_us, total_cycles,
        npu_utilisation_pct, npu_layers, cpu_layers
    """
    tflite_path = Path(tflite_path)
    if output_dir is None:
        output_dir = tflite_path.parent / f"{tflite_path.stem}_vela_256_384k"

    stdout = run_vela(tflite_path, Path(output_dir))

    npu_layers, cpu_layers = _parse_operator_counts(stdout)
    batch_us = _parse_batch_inference_time(stdout)
    total_cycles = _parse_total_cycles(stdout)
    npu_util = _parse_npu_utilisation(stdout)

    # If Vela did not emit a batch-inference-time line, derive from cycles at 1 GHz.
    if batch_us == 0.0 and total_cycles > 0:
        batch_us = total_cycles / 1_000.0   # cycles / (1e9 Hz) * 1e6 µs

    return {
        "hw_id":                  HW_ID,
        "tflite_path":            str(tflite_path),
        "batch_inference_time_us": batch_us,
        "total_cycles":           total_cycles,
        "npu_utilisation_pct":    npu_util,
        "npu_layers":             npu_layers,
        "cpu_layers":             cpu_layers,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Vela on a .tflite file for Ethos-U65-256 / Dedicated_Sram 384 KB "
            "and print structured performance metrics as JSON."
        )
    )
    parser.add_argument("tflite", help="Path to the INT8 .tflite file")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for Vela output artefacts (default: <stem>_vela_256_384k/)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    metrics = measure(args.tflite, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    _cli()
