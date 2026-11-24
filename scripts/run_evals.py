"""CLI wrapper for the BioCellAgents Dual-Layer Eval & Benchmark Harness.

Usage:
    python scripts/run_evals.py --mode synthetic --device cpu --fail-on-error
    python scripts/run_evals.py --mode real --device cuda --output-dir results/evals
    python scripts/run_evals.py --fidelity-only --fail-on-error
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evals.runner import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
