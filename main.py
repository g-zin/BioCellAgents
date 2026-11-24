"""BioCellAgents — Computational Pathology & Multi-Agent CLI Entry Point.

Supports:
  - PanNuke fold patches (``--fold_dir data/raw/fold3 --patch_index 0``)
  - Custom H&E tissue images (``--image path/to/patch.png``)
  - Full 3-agent CrewAI mode (default when an LLM API key is configured)
  - Fast 100% offline deterministic report mode (``--deterministic``, 0% hallucination)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()


# --- CrewAI <-> Groq compatibility shim ------------------------------------
# CrewAI marks stable prompt prefixes with a `cache_breakpoint` flag (an
# OpenAI prompt-caching feature). Groq rejects this property, and litellm's
# drop_params does not strip message-level properties, so we remove it from
# every message before it reaches the provider.
def _strip_cache_breakpoint(messages: Any) -> Any:
    if not messages:
        return messages
    cleaned = []
    for msg in messages:
        if isinstance(msg, dict) and "cache_breakpoint" in msg:
            msg = {k: v for k, v in msg.items() if k != "cache_breakpoint"}
        cleaned.append(msg)
    return cleaned


def _configure_litellm() -> None:
    import litellm
    import litellm.main

    litellm.drop_params = True
    litellm.cache = None

    if getattr(litellm.main.completion, "_biocell_patched", False):
        return

    _orig_completion = litellm.main.completion

    def _patched_completion(*args: Any, **kwargs: Any) -> Any:
        if "messages" in kwargs:
            kwargs["messages"] = _strip_cache_breakpoint(kwargs["messages"])
        elif args and isinstance(args[0], list):
            args = (_strip_cache_breakpoint(args[0]),) + args[1:]
        return _orig_completion(*args, **kwargs)

    setattr(_patched_completion, "_biocell_patched", True)
    litellm.main.completion = _patched_completion
    litellm.completion = _patched_completion


def run_pipeline(
    fold_dir: str = "data/raw/fold1",
    patch_index: int = 0,
    *,
    image_path: Optional[str] = None,
    deterministic: bool = False,
    output_path: str = "reports/pathology_report.md",
    fetch_literature: bool = True,
) -> Any:
    """Run the BioCellAgents pathology pipeline (3-agent CrewAI or deterministic)."""
    from src.evals.report_fidelity_eval import evaluate_report_fidelity
    from src.pipeline import (
        build_precomputed_payload,
        build_precomputed_payload_from_image,
        load_image_file,
        serialize_precomputed_payload,
    )
    from src.reporting import render_deterministic_report

    mode_label = "Deterministic Script Engine" if deterministic else "3-Agent Tool-Using CrewAI"
    print(f"Initializing BioCellAgents Execution Engine ({mode_label})...")
    if image_path:
        print(f"   - Target Image          : {image_path}")
    else:
        print(f"   - Target Fold Directory : {fold_dir}")
        print(f"   - Target Patch Index    : {patch_index}")

    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Deterministic vision + calibration + spatial graph pre-computation
    print("\n--- Pre-Computation Phase ---")
    if image_path:
        image = load_image_file(image_path)
        payload = build_precomputed_payload_from_image(
            image=image,
            source_name=Path(image_path).name,
            fetch_literature=fetch_literature,
        )
    else:
        payload = build_precomputed_payload(
            fold_dir=fold_dir,
            patch_index=patch_index,
            fetch_literature=fetch_literature,
        )
    precomputed_results = serialize_precomputed_payload(payload)
    print("--- Pre-Computation Complete ---\n")

    # 2. Deterministic offline report path (zero LLM / zero hallucination)
    if deterministic:
        report_content = render_deterministic_report(payload)
        audit = evaluate_report_fidelity(report_content, payload)
        report_path.write_text(report_content, encoding="utf-8")
        print("=" * 60)
        print("DETERMINISTIC COMPUTATIONAL PATHOLOGY ANALYSIS COMPLETE")
        print(f"Fidelity Audit Score : {audit['score']:.1f}/100 (Passed: {audit['passed']})")
        print(f"Report saved at      : {report_path.resolve()}")
        print("=" * 60)
        return {"report": report_content, "payload": payload, "fidelity_audit": audit}

    # 3. Genuinely agentic 3-agent CrewAI path
    _configure_litellm()
    from src.crew import PathologyCrew

    inputs = {
        "fold_dir": fold_dir if not image_path else str(Path(image_path).parent),
        "patch_index": patch_index,
        "precomputed_results": precomputed_results,
    }

    crew_instance = PathologyCrew().crew()
    result = crew_instance.kickoff(inputs=inputs)

    report_content = str(result.raw) if hasattr(result, "raw") else str(result)
    report_path.write_text(report_content, encoding="utf-8")
    audit = evaluate_report_fidelity(report_content, payload)

    print("\n" + "=" * 60)
    print("COMPUTATIONAL PATHOLOGY ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Fidelity Audit Score : {audit['score']:.1f}/100 (Passed: {audit['passed']})")
    print(f"Report saved at      : {report_path.resolve()}\n")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for BioCellAgents."""
    parser = argparse.ArgumentParser(
        description="Run BioCellAgents Computational Pathology & Multi-Agent Engine"
    )
    parser.add_argument(
        "--fold_dir",
        type=str,
        default="data/raw/fold1",
        help="Path to PanNuke fold directory (ignored when --image is provided)",
    )
    parser.add_argument(
        "--patch_index",
        type=int,
        default=0,
        help="Index of image patch to analyze",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional path to a standalone H&E image (PNG/JPG/TIFF)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Generate the 11-section RUO report deterministically in Python (no LLM API key needed)",
    )
    parser.add_argument(
        "--no-literature",
        action="store_true",
        help="Skip PubMed/Europe PMC pre-retrieval (useful for offline runs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/pathology_report.md",
        help="Output Markdown report path",
    )
    return parser


if __name__ == "__main__":
    cli_args = build_arg_parser().parse_args()
    run_pipeline(
        fold_dir=cli_args.fold_dir,
        patch_index=cli_args.patch_index,
        image_path=cli_args.image,
        deterministic=cli_args.deterministic,
        output_path=cli_args.output,
        fetch_literature=not cli_args.no_literature,
    )
