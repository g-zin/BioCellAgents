"""Hyperparameter tuning for BioCellAgents models.

Provides grid search and random search over hyperparameters
for all 3 model types (segmentation, cell classification, organ classification).
"""
import itertools
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TuningResult:
    """Result of a single hyperparameter trial."""
    params: Dict[str, Any]
    score: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    is_best: bool = False


class HyperparameterTuner:
    """Grid/random search hyperparameter tuner.

    Usage:
        tuner = HyperparameterTuner(
            param_grid={
                "learning_rate": [1e-5, 1e-4, 5e-4],
                "batch_size": [8, 16, 32],
                "weight_decay": [0.0, 1e-4],
            },
            search_strategy="random",
            max_trials=20,
        )
        best = tuner.run(train_fn, metric="val_dice", maximize=True)
    """

    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        search_strategy: str = "random",
        max_trials: int = 50,
        random_seed: int = 42,
    ):
        self.param_grid = param_grid
        self.search_strategy = search_strategy
        self.max_trials = max_trials
        self.random_seed = random_seed
        self.results: List[TuningResult] = []

    def _generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate parameter combinations based on strategy."""
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]

        if self.search_strategy == "grid":
            all_combos = list(itertools.product(*values))
            return [dict(zip(keys, combo)) for combo in all_combos]
        elif self.search_strategy == "random":
            random.seed(self.random_seed)
            combos = []
            for _ in range(self.max_trials):
                combo = {k: random.choice(v) for k, v in zip(keys, values)}
                combos.append(combo)
            return combos
        else:
            raise ValueError(f"Unknown strategy: {self.search_strategy}")

    def run(
        self,
        train_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        metric: str = "val_dice",
        maximize: bool = True,
        verbose: bool = True,
    ) -> TuningResult:
        """Execute the hyperparameter search.

        Args:
            train_fn: Callable that takes a params dict and returns metrics dict.
            metric: Metric name to optimize.
            maximize: True if higher is better.
            verbose: Print progress.

        Returns:
            Best TuningResult.
        """
        combinations = self._generate_combinations()
        total = min(len(combinations), self.max_trials)
        if verbose:
            print(f"Starting {self.search_strategy} search: {total} trials")
            print(f"   Optimizing: {metric} ({'maximize' if maximize else 'minimize'})")

        for i, params in enumerate(combinations[:total], 1):
            if verbose:
                print(f"\n   Trial {i}/{total}: {params}")

            try:
                metrics = train_fn(params)
                score = metrics.get(metric, 0.0)
            except Exception as e:
                if verbose:
                    print(f"   Trial {i} failed: {e}")
                score = float("-inf") if maximize else float("inf")
                metrics = {"error": str(e)}

            self.results.append(TuningResult(params=params, score=score, metrics=metrics))

            if verbose:
                print(f"   Score ({metric}): {score:.4f}")

        # Determine best
        if maximize:
            best = max(self.results, key=lambda r: r.score)
        else:
            best = min(self.results, key=lambda r: r.score)
        best.is_best = True

        if verbose:
            print(f"\n{'='*50}")
            print(f"Best params: {best.params}")
            print(f"   Best {metric}: {best.score:.4f}")
            print(f"{'='*50}")

        return best

    def save_results(self, output_path: str) -> Path:
        """Save tuning results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "strategy": self.search_strategy,
            "param_grid": self.param_grid,
            "num_trials": len(self.results),
            "results": [
                {
                    "params": r.params,
                    "score": r.score,
                    "metrics": r.metrics,
                    "is_best": r.is_best,
                }
                for r in self.results
            ],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return output_path

    def get_top_n(self, n: int = 5) -> List[TuningResult]:
        """Get top N results sorted by score."""
        maximize = any(r.score > 0 for r in self.results)
        if maximize:
            return sorted(self.results, key=lambda r: r.score, reverse=True)[:n]
        return sorted(self.results, key=lambda r: r.score)[:n]