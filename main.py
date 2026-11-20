import argparse
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()


# --- CrewAI <-> Groq compatibility shim ------------------------------------
# CrewAI marks stable prompt prefixes with a `cache_breakpoint` flag (an
# OpenAI prompt-caching feature). Groq rejects this property, and litellm's
# drop_params does not strip message-level properties, so we remove it from
# every message before it reaches the provider.
def _strip_cache_breakpoint(messages):
    if not messages:
        return messages
    cleaned = []
    for m in messages:
        if isinstance(m, dict) and "cache_breakpoint" in m:
            m = {k: v for k, v in m.items() if k != "cache_breakpoint"}
        cleaned.append(m)
    return cleaned


def _configure_litellm() -> None:
    import litellm
    import litellm.main

    litellm.drop_params = True
    litellm.cache = None

    if getattr(litellm.main.completion, "_biocell_patched", False):
        return

    _orig_completion = litellm.main.completion

    def _patched_completion(*args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = _strip_cache_breakpoint(kwargs["messages"])
        elif args and isinstance(args[0], list):
            args = (_strip_cache_breakpoint(args[0]),) + args[1:]
        return _orig_completion(*args, **kwargs)

    setattr(_patched_completion, "_biocell_patched", True)
    litellm.main.completion = _patched_completion
    litellm.completion = _patched_completion


def run_pipeline(fold_dir: str = "data/raw/fold1", patch_index: int = 0):
    """Triggers the multi-agent pathology analysis pipeline for a specified tissue patch."""
    _configure_litellm()
    from src.crew import PathologyCrew
    from src.pipeline import build_precomputed_results

    print("Initializing BioCellAgents Multi-Agent Execution Engine...")
    print(f"   - Target Fold Directory : {fold_dir}")
    print(f"   - Target Patch Index   : {patch_index}")

    # Ensure output report folder exists
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Run pre-computation pipeline (organ classification + cell detection + spatial analysis)
    print("\n--- Pre-Computation Phase ---")
    precomputed_results = build_precomputed_results(fold_dir, patch_index)
    print("--- Pre-Computation Complete ---\n")

    inputs = {
        "fold_dir": fold_dir,
        "patch_index": patch_index,
        "precomputed_results": precomputed_results,
    }

    try:
        # Assemble and execute CrewAI workflow
        crew_instance = PathologyCrew().crew()
        result = crew_instance.kickoff(inputs=inputs)

        # Write final markdown output to disk
        report_path = reports_dir / "pathology_report.md"
        report_content = str(result.raw) if hasattr(result, "raw") else str(result)
        report_path.write_text(report_content, encoding="utf-8")

        print("\n" + "=" * 60)
        print("COMPUTATIONAL PATHOLOGY ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Report saved at: {report_path.resolve()}\n")
        return result

    except Exception as e:
        print(f"\nExecution failed: {str(e)}")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BioCellAgents Multi-Agent Pathology Engine")
    parser.add_argument(
        "--fold_dir", type=str, default="data/raw/fold1", help="Path to PanNuke fold directory"
    )
    parser.add_argument(
        "--patch_index", type=int, default=0, help="Index of image patch to analyze"
    )
    args = parser.parse_args()

    run_pipeline(fold_dir=args.fold_dir, patch_index=args.patch_index)
