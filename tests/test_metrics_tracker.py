"""Unit tests for BioCellAgents MetricsTracker and logging utilities."""
import json
import tempfile
from pathlib import Path

import pytest


class TestMetricsTracker:
    """Tests for MetricsTracker."""

    def test_log_and_retrieve(self):
        from src.utils.metrics_tracker import MetricsTracker
        tracker = MetricsTracker(task_name="test")
        tracker.log_epoch(epoch=1, train_loss=0.5, val_loss=0.45, val_dice=0.72)
        tracker.log_epoch(epoch=2, train_loss=0.4, val_loss=0.42, val_dice=0.75)
        tracker.log_epoch(epoch=3, train_loss=0.35, val_loss=0.40, val_dice=0.78)
        assert len(tracker.history) == 3
        assert tracker.get_series("val_dice") == [0.72, 0.75, 0.78]

    def test_get_best_maximize(self):
        from src.utils.metrics_tracker import MetricsTracker
        tracker = MetricsTracker(task_name="test")
        tracker.log_epoch(epoch=1, val_dice=0.70)
        tracker.log_epoch(epoch=2, val_dice=0.85)
        tracker.log_epoch(epoch=3, val_dice=0.80)
        best = tracker.get_best("val_dice", maximize=True)
        assert best["epoch"] == 2
        assert best["val_dice"] == 0.85

    def test_get_best_minimize(self):
        from src.utils.metrics_tracker import MetricsTracker
        tracker = MetricsTracker(task_name="test")
        tracker.log_epoch(epoch=1, val_loss=0.5)
        tracker.log_epoch(epoch=2, val_loss=0.3)
        tracker.log_epoch(epoch=3, val_loss=0.4)
        best = tracker.get_best("val_loss", maximize=False)
        assert best["epoch"] == 2
        assert best["val_loss"] == 0.3

    def test_get_best_missing_metric(self):
        from src.utils.metrics_tracker import MetricsTracker
        tracker = MetricsTracker(task_name="test")
        tracker.log_epoch(epoch=1, val_loss=0.5)
        assert tracker.get_best("nonexistent") is None

    def test_summary(self):
        from src.utils.metrics_tracker import MetricsTracker
        tracker = MetricsTracker(task_name="test")
        tracker.log_epoch(epoch=1, loss=0.5)
        tracker.log_epoch(epoch=2, loss=0.3)
        summary = tracker.get_summary()
        assert summary["task"] == "test"
        assert summary["total_epochs"] == 2
        assert summary["metrics"]["loss"]["min"] == 0.3
        assert summary["metrics"]["loss"]["max"] == 0.5
        assert summary["metrics"]["loss"]["mean"] == 0.4

    def test_save_and_load(self):
        from src.utils.metrics_tracker import MetricsTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MetricsTracker(output_dir=tmpdir, task_name="test")
            tracker.log_epoch(epoch=1, loss=0.5, dice=0.7)
            tracker.log_epoch(epoch=2, loss=0.4, dice=0.8)
            json_path = tracker.save()
            assert json_path.exists()
            csv_path = json_path.with_suffix(".csv")
            assert csv_path.exists()
            tracker2 = MetricsTracker(task_name="test")
            tracker2.load(str(json_path))
            assert len(tracker2.history) == 2
            assert tracker2.get_series("loss") == [0.5, 0.4]

    def test_log_event(self):
        from src.utils.metrics_tracker import MetricsTracker
        tracker = MetricsTracker(task_name="test")
        tracker.log_event("checkpoint_saved", path="model.pt")
        assert len(tracker.history) == 1
        assert tracker.history[0]["event"] == "checkpoint_saved"


class TestLoggingConfig:
    """Tests for logging configuration."""

    def test_setup_logger(self):
        from src.utils.logging_config import setup_logger
        import shutil
        tmpdir = tempfile.mkdtemp()
        try:
            logger = setup_logger(name="test_logger", log_dir=tmpdir)
            logger.info("Test message")
            log_file = Path(tmpdir) / "test_logger.log"
            assert log_file.exists()
            content = log_file.read_text()
            assert "Test message" in content
        finally:
            # Close handlers to release file handles (Windows-safe cleanup)
            for handler in logger.handlers:
                handler.close()
            logger.handlers.clear()
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_get_logger_returns_same(self):
        from src.utils.logging_config import get_logger
        logger1 = get_logger("test_same")
        logger2 = get_logger("test_same")
        assert logger1 is logger2

    def test_json_format(self):
        from src.utils.logging_config import setup_logger
        import logging
        import shutil
        tmpdir = tempfile.mkdtemp()
        try:
            logger = setup_logger(name="json_test", log_dir=tmpdir, level=logging.DEBUG)
            logger.info("JSON test", extra={"epoch": 5, "loss": 0.123})
            log_file = Path(tmpdir) / "json_test.log"
            content = log_file.read_text().strip()
            last_line = content.split("\n")[-1]
            parsed = json.loads(last_line)
            assert parsed["epoch"] == 5
            assert parsed["loss"] == 0.123
        finally:
            for handler in logger.handlers:
                handler.close()
            logger.handlers.clear()
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_training_logger(self):
        from src.utils.logging_config import TrainingLogger
        import shutil
        tmpdir = tempfile.mkdtemp()
        try:
            tlog = TrainingLogger(task_name="seg_test", log_dir=tmpdir)
            tlog.log_epoch_start(epoch=1, total_epochs=10)
            tlog.log_batch(step=100, loss=0.5, lr=1e-4)
            tlog.log_epoch_end(epoch=1, train_loss=0.5, val_loss=0.45, val_metric=0.72)
            tlog.log_checkpoint("model.pt", is_best=True)
        finally:
            for handler in tlog.logger.handlers:
                handler.close()
            tlog.logger.handlers.clear()
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_inference_logger(self):
        from src.utils.logging_config import InferenceLogger
        import shutil
        tmpdir = tempfile.mkdtemp()
        try:
            ilog = InferenceLogger(model_name="infer_test", log_dir=tmpdir)
            with ilog.track("test_operation"):
                pass
            ilog.log_result(prediction="Liver", confidence=0.95)
        finally:
            for handler in ilog.logger.handlers:
                handler.close()
            ilog.logger.handlers.clear()
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_gpu_memory_info(self):
        from src.utils.logging_config import get_gpu_memory_info
        info = get_gpu_memory_info()
        assert isinstance(info, dict)