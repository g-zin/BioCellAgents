"""Training metrics tracker for BioCellAgents.

Tracks loss curves, accuracy, dice scores, and other metrics across epochs.
Supports JSON/CSV export and optional TensorBoard integration.
"""
import json
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class MetricsTracker:
    """Tracks and persists training metrics across epochs.

    Usage:
        tracker = MetricsTracker(output_dir="checkpoints/segmentation")
        tracker.log_epoch(epoch=1, train_loss=0.5, val_loss=0.45, val_dice=0.72)
        tracker.save()
        best = tracker.get_best("val_dice", maximize=True)
    """

    def __init__(self, output_dir: str = "checkpoints", task_name: str = "model"):
        self.output_dir = Path(output_dir)
        self.task_name = task_name
        self.history: List[Dict[str, Any]] = []
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._created_at = datetime.now().isoformat()

    def log_epoch(self, epoch: int, **metrics: float) -> None:
        """Log metrics for a single epoch."""
        entry = {"epoch": epoch, "timestamp": datetime.now().isoformat()}
        entry.update(metrics)
        self.history.append(entry)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._metrics[key].append(float(value))

    def log_event(self, event: str, **kwargs) -> None:
        """Log a non-numeric event (e.g., checkpoint saved)."""
        entry = {"event": event, "timestamp": datetime.now().isoformat(), **kwargs}
        self.history.append(entry)

    def get_best(self, metric: str, maximize: bool = True) -> Optional[Dict[str, Any]]:
        """Get the epoch with the best value for a given metric."""
        candidates = [e for e in self.history if metric in e and isinstance(e[metric], (int, float))]
        if not candidates:
            return None
        if maximize:
            return max(candidates, key=lambda e: e[metric])
        return min(candidates, key=lambda e: e[metric])

    def get_series(self, metric: str) -> List[float]:
        """Get the time series for a metric."""
        return self._metrics.get(metric, [])

    def get_summary(self) -> Dict[str, Any]:
        """Generate a summary of all tracked metrics."""
        summary: Dict[str, Any] = {
            "task": self.task_name,
            "created_at": self._created_at,
            "total_epochs": len([e for e in self.history if "epoch" in e]),
            "metrics": {},
        }
        for metric, values in self._metrics.items():
            if not values:
                continue
            summary["metrics"][metric] = {
                "min": round(min(values), 6),
                "max": round(max(values), 6),
                "mean": round(sum(values) / len(values), 6),
                "final": round(values[-1], 6),
            }
        return summary

    def save(self, filename: Optional[str] = None) -> Path:
        """Save metrics history to JSON and CSV files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        base = filename or f"{self.task_name}_metrics"

        json_path = self.output_dir / f"{base}.json"
        data = {
            "task": self.task_name,
            "created_at": self._created_at,
            "summary": self.get_summary(),
            "history": self.history,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        csv_path = self.output_dir / f"{base}.csv"
        epoch_entries = [e for e in self.history if "epoch" in e]
        if epoch_entries:
            fieldnames = sorted(set(k for e in epoch_entries for k in e.keys()))
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(epoch_entries)

        return json_path

    def load(self, path: str) -> None:
        """Load metrics history from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.history = data.get("history", [])
        self._created_at = data.get("created_at", "")
        self.task_name = data.get("task", self.task_name)
        self._metrics.clear()
        for entry in self.history:
            for key, value in entry.items():
                if isinstance(value, (int, float)) and key != "epoch":
                    self._metrics[key].append(float(value))

    def to_tensorboard(self, log_dir: Optional[str] = None) -> None:
        """Write metrics to TensorBoard (if installed)."""
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print("TensorBoard not installed. Skipping TB logging.")
            return

        tb_dir = Path(log_dir) if log_dir else self.output_dir / "tensorboard"
        writer = SummaryWriter(log_dir=str(tb_dir))
        for entry in self.history:
            if "epoch" not in entry:
                continue
            epoch = entry["epoch"]
            for key, value in entry.items():
                if isinstance(value, (int, float)) and key != "epoch":
                    writer.add_scalar(f"metrics/{key}", value, epoch)
        writer.close()