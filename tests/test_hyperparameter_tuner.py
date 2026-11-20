"""Offline unit tests for src.inference.hyperparameter_tuner."""

import json

import pytest

from src.inference.hyperparameter_tuner import HyperparameterTuner


class TestHyperparameterTuner:
    """Behavior tests for grid/random search, failure handling, and persistence."""

    def test_grid_search_maximize_and_top_n(self):
        tuner = HyperparameterTuner(
            param_grid={"lr": [1e-4, 1e-3], "bs": [8, 16]},
            search_strategy="grid",
            max_trials=10,
        )

        def train_fn(params):
            return {"val_dice": params["lr"] * 100 + params["bs"] * 0.01}

        best = tuner.run(train_fn, metric="val_dice", maximize=True, verbose=False)
        assert best.is_best is True
        assert best.params == {"lr": 1e-3, "bs": 16}
        assert len(tuner.results) == 4

        top2 = tuner.get_top_n(2)
        assert len(top2) == 2
        assert top2[0].score >= top2[1].score

    def test_random_search_and_failing_trial_handling(self, tmp_path):
        tuner = HyperparameterTuner(
            param_grid={"lr": [0.1, 0.2, 0.3]},
            search_strategy="random",
            max_trials=3,
            random_seed=123,
        )

        call_count = {"n": 0}

        def flaky_fn(params):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("OOM")
            return {"val_loss": params["lr"]}

        best = tuner.run(flaky_fn, metric="val_loss", maximize=False, verbose=False)
        assert best.is_best is True
        assert best.score < float("inf")
        assert tuner.results[0].score == float("inf")
        assert "error" in tuner.results[0].metrics

        out_path = tmp_path / "tuning" / "results.json"
        saved_path = tuner.save_results(out_path)
        assert saved_path.exists()
        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        assert payload["num_trials"] == 3

    def test_unknown_strategy_raises(self):
        tuner = HyperparameterTuner(param_grid={"lr": [1e-3]}, search_strategy="bayes")
        with pytest.raises(ValueError, match="Unknown strategy"):
            tuner.run(lambda p: {"val_dice": 1.0}, verbose=False)

    def test_minimize_positive_metric_and_get_top_n_order(self):
        tuner = HyperparameterTuner(param_grid={"lr": [0.1, 0.5, 0.9]}, search_strategy="grid")
        best = tuner.run(
            lambda p: {"val_loss": p["lr"]}, metric="val_loss", maximize=False, verbose=False
        )
        assert best.score == pytest.approx(0.1)
        top3 = tuner.get_top_n(3)
        assert [r.score for r in top3] == pytest.approx([0.1, 0.5, 0.9])
        assert tuner.get_top_n(1)[0].score == pytest.approx(0.1)

    def test_invalid_max_trials_and_empty_param_lists_raise(self):
        tuner_zero = HyperparameterTuner(param_grid={"lr": [0.1]}, max_trials=0)
        with pytest.raises(ValueError, match="max_trials"):
            tuner_zero.run(lambda p: {"val_dice": 1.0}, verbose=False)

        tuner_empty_vals = HyperparameterTuner(param_grid={"lr": []}, search_strategy="random")
        with pytest.raises(ValueError, match="must not be empty"):
            tuner_empty_vals.run(lambda p: {"val_dice": 1.0}, verbose=False)
