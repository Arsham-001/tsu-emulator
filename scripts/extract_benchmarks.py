#!/usr/bin/env python
"""
Extract and summarize benchmark results for TSU.

Process:
1. Reads any generated benchmark output files in `visual_output/`:
   - benchmark_report.txt (human-readable)
   - benchmark_results.json (machine-readable) if present
2. Produces a concise summary in BENCHMARK_SUMMARY.md
3. Optionally updates README.md between markers:
   <!-- BENCHMARK_SUMMARY_START --> ... <!-- BENCHMARK_SUMMARY_END -->

Design Goals:
- Idempotent: safe to run multiple times
- Fails gracefully if benchmarks haven't run yet
- Minimal parsing assumptions

Extendability:
- Add additional metrics parsing (KL divergence, ESS, timing) when format stabilizes.
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
VIS_DIR = ROOT / "visual_output"
SUMMARY_FILE = ROOT / "BENCHMARK_SUMMARY.md"
README_FILE = ROOT / "README.md"

START_MARKER = "<!-- BENCHMARK_SUMMARY_START -->"
END_MARKER = "<!-- BENCHMARK_SUMMARY_END -->"


def load_report_text() -> str | None:
    txt_path = VIS_DIR / "benchmark_report.txt"
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8")
    return None


def load_results_json() -> dict | None:
    json_path = VIS_DIR / "benchmark_results.json"
    if json_path.exists():
        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def parse_metrics(report_text: str | None, results_json: dict | None) -> dict:
    """
    Extract a few key metrics heuristically.
    Fallback to simple placeholders if unavailable.
    """
    metrics = {
        "gaussian_kl": "n/a",
        "multimodal_modes": "n/a",
        "ising_gap": "n/a",
        "regression_coverage": "n/a",
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    if results_json:
        # Heuristic keys (adjust when stable)
        # Example expected structure (pseudo):
        # {
        #   "sampling": {"gaussian": {"kl": 0.0023}, "multimodal": {"modes_found": 3}},
        #   "optimization": {"ising": {"gap": 0.0}},
        #   "ml": {"regression": {"coverage": 1.0}}
        # }
        sampling = results_json.get("sampling", {})
        gauss = sampling.get("gaussian", {})
        metrics["gaussian_kl"] = gauss.get("kl", metrics["gaussian_kl"])
        multi = sampling.get("multimodal", {})
        metrics["multimodal_modes"] = multi.get("modes_found", metrics["multimodal_modes"])

        opt = results_json.get("optimization", {})
        ising = opt.get("ising", {})
        metrics["ising_gap"] = ising.get("gap", metrics["ising_gap"])

        ml = results_json.get("ml", {})
        reg = ml.get("regression", {})
        cov = reg.get("coverage", None)
        if cov is not None:
            # Format as percentage if fraction
            try:
                cov_val = float(cov)
                metrics["regression_coverage"] = f"{cov_val*100:.1f}%"
            except Exception:
                metrics["regression_coverage"] = str(cov)

    # Fallback attempt: parse text for hints if JSON missing data
    if report_text and metrics["gaussian_kl"] == "n/a":
        kl_match = re.search(r"Gaussian.*?KL\s*[:=]\s*([\d\.eE-]+)", report_text)
        if kl_match:
            metrics["gaussian_kl"] = kl_match.group(1)

    return metrics


def build_markdown(metrics: dict) -> str:
    lines = []
    lines.append("# Benchmark Summary")
    lines.append("")
    lines.append(f"Last updated (UTC): `{metrics['timestamp']}`")
    lines.append("")
    lines.append("Key metrics (quick mode or last run):")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Gaussian KL divergence | {metrics['gaussian_kl']} |")
    lines.append(f"| Multimodal modes found | {metrics['multimodal_modes']} |")
    lines.append(f"| Ising optimality gap | {metrics['ising_gap']} |")
    lines.append(f"| Regression coverage (95% CI) | {metrics['regression_coverage']} |")
    lines.append("")
    lines.append("Run locally:")
    lines.append("```bash")
    lines.append("python -m tsu.benchmarks.runner --quick")
    lines.append("```")
    lines.append("")
    lines.append("Full benchmark details: see `visual_output/` artifacts or run full mode without `--quick`.")
    return "\n".join(lines)


def write_summary(markdown: str):
    SUMMARY_FILE.write_text(markdown, encoding="utf-8")


def update_readme(markdown: str):
    if not README_FILE.exists():
        return
    original = README_FILE.read_text(encoding="utf-8")
    if START_MARKER not in original or END_MARKER not in original:
        # Do nothing if markers not present
        return

    pattern = re.compile(
        rf"{START_MARKER}.*?{END_MARKER}",
        re.DOTALL,
    )
    replacement = f"{START_MARKER}\n{markdown}\n{END_MARKER}"
    updated = pattern.sub(replacement, original)
    if updated != original:
        README_FILE.write_text(updated, encoding="utf-8")


def main():
    report_text = load_report_text()
    results_json = load_results_json()
    metrics = parse_metrics(report_text, results_json)
    md = build_markdown(metrics)
    write_summary(md)
    update_readme(md)
    print("Benchmark summary generated.")
    print(f" -> {SUMMARY_FILE}")
    print("README updated (if markers present).")


if __name__ == "__main__":
    main()
