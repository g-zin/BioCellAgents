"""Shared evaluation metrics and result saving for BioCellAgents."""
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def compute_metrics(
    predictions: List[int],
    ground_truth: List[int],
    class_names: List[str],
) -> Dict:
    """Compute accuracy, per-class precision/recall/F1, and confusion matrix.
    
    Args:
        predictions: List of predicted class indices
        ground_truth: List of ground truth class indices
        class_names: List of class name strings
    
    Returns:
        Dict with 'accuracy', 'per_class' (dict of class -> {precision, recall, f1}),
        and 'confusion_matrix' (2D list)
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

    acc = accuracy_score(ground_truth, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, labels=list(range(len(class_names))), zero_division=0
    )
    cm = confusion_matrix(ground_truth, predictions, labels=list(range(len(class_names)))).tolist()

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
        }

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1_score(ground_truth, predictions, average="macro", zero_division=0)), 4),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def save_results(results: Dict, output_path: str) -> None:
    """Save evaluation results to a JSON file.
    
    Args:
        results: Dict of metrics to save
        output_path: Path to output JSON file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {path}")
