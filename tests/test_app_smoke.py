"""Offline smoke, chart-builder, and headless Streamlit AppTest tests for biocell_app and app."""

from __future__ import annotations

import importlib
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import numpy as np
import plotly.graph_objects as go


def test_app_and_biocell_app_import_headless(monkeypatch):
    """Both app and biocell_app should import cleanly without CUDA, data, or API keys."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    app_mod = importlib.import_module("app")
    biocell_mod = importlib.import_module("biocell_app")

    assert callable(app_mod.main)
    assert callable(biocell_mod.main)
    assert app_mod.main is biocell_mod.main
    assert len(biocell_mod.ORGAN_TYPES) == 19
    assert len(biocell_mod.CELL_TYPES) == 5


def test_biocell_app_chart_builders_and_okabe_ito_palette():
    """Chart builders produce valid dark-mode Plotly figures and support Okabe-Ito palette."""
    import biocell_app

    pie_fig = biocell_app.build_cell_pie(
        {"Neoplastic": 12, "Inflammatory": 5, "Connective": 0}, colorblind_safe=True
    )
    assert isinstance(pie_fig, go.Figure)
    assert len(pie_fig.data) == 1
    assert list(pie_fig.data[0].labels) == ["Neoplastic", "Inflammatory"]

    bar_fig = biocell_app.build_organ_bar(
        [
            {"organ": "Breast", "confidence": 0.72},
            {"organ": "Lung", "confidence": 0.18},
            {"organ": "Colon", "confidence": 0.05},
        ]
    )
    assert isinstance(bar_fig, go.Figure)
    assert len(bar_fig.data) == 1

    scatter_fig = biocell_app.build_spatial_scatter(
        [
            {"x": 10.0, "y": 20.0, "class": "Neoplastic"},
            {"x": 30.0, "y": 40.0, "class": "Inflammatory"},
        ],
        colorblind_safe=True,
    )
    assert isinstance(scatter_fig, go.Figure)
    assert len(scatter_fig.data) == 2


def test_load_patch_data_edge_cases_and_bytes_labels(tmp_path):
    """load_patch_data handles empty arrays, negative/oob indices, float scaling, and bytes labels."""
    import biocell_app
    from src.pipeline import load_patch

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    np.save(empty_dir / "images.npy", np.zeros((0, 32, 32, 3), dtype=np.uint8))
    biocell_app.load_patch_data.clear()
    img, tissue, n = biocell_app.load_patch_data(str(empty_dir), 0)
    assert img is None
    assert tissue is None
    assert n == 0

    fold_dir = tmp_path / "fold"
    fold_dir.mkdir()
    imgs = np.zeros((2, 16, 16, 3), dtype=np.float32)
    imgs[0] = 0.5
    imgs[1] = 1.0
    np.save(fold_dir / "images.npy", imgs)
    np.save(fold_dir / "tissues.npy", np.array([b"Breast", b"  'Liver'  "], dtype=object))

    biocell_app.load_patch_data.clear()
    img0, tissue0, n0 = biocell_app.load_patch_data(str(fold_dir), -99)
    assert n0 == 2
    assert img0 is not None and img0.dtype == np.uint8
    assert tissue0 == "Breast"

    img1, tissue1, n1 = biocell_app.load_patch_data(str(fold_dir), 99)
    assert n1 == 2
    assert img1 is not None and int(img1.max()) == 255
    assert tissue1 == "Liver"

    p_img, p_tissue = load_patch(str(fold_dir), 0)
    assert p_img.dtype == np.uint8
    assert p_tissue == "Breast"


def test_partial_dicts_and_chat_none_content(monkeypatch):
    """Chart builders handle missing keys and run_chat_query handles None LLM content."""
    import biocell_app

    bar_fig = biocell_app.build_organ_bar([{"organ": "Breast"}])
    assert isinstance(bar_fig, go.Figure)
    assert list(bar_fig.data[0].x) == [0.0]

    scatter_fig = biocell_app.build_spatial_scatter([{"class": "Neoplastic"}])
    assert isinstance(scatter_fig, go.Figure)
    assert list(scatter_fig.data[0].x) == [0.0]
    assert list(scatter_fig.data[0].y) == [0.0]

    class _FakeLiteLLM:
        drop_params = False

        @staticmethod
        def completion(**kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])

    monkeypatch.setitem(sys.modules, "litellm", _FakeLiteLLM)
    assert biocell_app.run_chat_query("hello", {}, []) == ""


def test_run_crewai_pipeline_litellm_patch_idempotent(monkeypatch):
    """Repeated calls to run_crewai_pipeline must not re-wrap litellm.main.completion."""
    import biocell_app

    orig_called = []

    def _orig_completion(*args, **kwargs):
        orig_called.append(kwargs.get("messages"))
        return "ok"

    fake_main = SimpleNamespace(completion=_orig_completion)
    fake_litellm = ModuleType("litellm")
    setattr(fake_litellm, "main", fake_main)
    setattr(fake_litellm, "completion", _orig_completion)
    setattr(fake_litellm, "drop_params", False)
    setattr(fake_litellm, "cache", None)

    class _FakeCrewObj:
        task_callback = None

        def kickoff(self, inputs):
            return SimpleNamespace(raw="report")

    class _FakePathologyCrew:
        def crew(self):
            return _FakeCrewObj()

    class _FakeLitTool:
        def _run(self, **kwargs):
            return {"status": "empty", "results": []}

    fake_crew_mod = ModuleType("src.crew")
    setattr(fake_crew_mod, "PathologyCrew", _FakePathologyCrew)
    fake_pubmed_mod = ModuleType("src.tools.pubmed_search_tool")
    setattr(fake_pubmed_mod, "LiteratureSearchTool", _FakeLitTool)

    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    monkeypatch.setitem(sys.modules, "litellm.main", fake_main)
    monkeypatch.setitem(sys.modules, "src.crew", fake_crew_mod)
    monkeypatch.setitem(sys.modules, "src.tools.pubmed_search_tool", fake_pubmed_mod)

    biocell_app.run_crewai_pipeline("fold", 0, {"organ": {"predicted_organ": "Breast"}})
    first_patched = fake_main.completion
    assert getattr(first_patched, "_biocell_patched", False) is True

    biocell_app.run_crewai_pipeline("fold", 0, {"organ": {"predicted_organ": "Breast"}})
    assert fake_main.completion is first_patched


def test_biocell_app_main_renders_real_graph_metrics_and_grayscale_overlay(monkeypatch):
    """main() renders graph topology KPIs, hardware badge, and handles 2D grayscale overlay."""
    import re

    import biocell_app

    rendered_cards = {}
    rendered_images = []
    session_state = {}

    class _FakeSt:
        def __init__(self):
            self.session_state = session_state
            self.sidebar = nullcontext()

        def set_page_config(self, **kwargs):
            pass

        def markdown(self, text, **kwargs):
            if 'class="metric-card"' in text:
                lbl = re.search(r'class="metric-label">([^<]+)<', text)
                val = re.search(r'class="metric-value">([^<]+)<', text)
                if lbl and val:
                    rendered_cards[lbl.group(1)] = val.group(1)

        def selectbox(self, label, options, **kwargs):
            return options[0]

        def number_input(self, label, **kwargs):
            return 0

        def divider(self):
            pass

        def button(self, label, **kwargs):
            return label == "Run Analysis"

        def checkbox(self, label, **kwargs):
            return label == "Skip AI Report (faster)"

        def info(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            raise AssertionError(f"Unexpected st.error: {args}")

        def warning(self, *args, **kwargs):
            pass

        def spinner(self, *args, **kwargs):
            return nullcontext()

        def columns(self, n):
            count = n if isinstance(n, int) else len(n)

            class _Col:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def button(self, *args, **kwargs):
                    return False

            return [_Col() for _ in range(count)]

        def image(self, img, **kwargs):
            rendered_images.append(img)

        def plotly_chart(self, *args, **kwargs):
            pass

        def caption(self, *args, **kwargs):
            pass

        def chat_message(self, *args, **kwargs):
            return nullcontext()

        def chat_input(self, *args, **kwargs):
            return None

    monkeypatch.setattr(biocell_app, "st", _FakeSt())
    gray_img = np.full((32, 32), 128, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.int32)
    mask[5:15, 5:15] = 1

    monkeypatch.setattr(biocell_app, "load_patch_data", lambda f, i: (gray_img, "Breast", 1))
    monkeypatch.setattr(
        biocell_app,
        "run_organ_classification",
        lambda img: {
            "predicted_organ": "Breast",
            "confidence": 0.9,
            "confidence_score": 0.9,
            "top3": [{"organ": "Breast", "confidence": 0.9}],
            "top_3_predictions": [{"organ": "Breast", "confidence": 0.9}],
            "temperature": 2.0,
            "time": 0.01,
        },
    )
    monkeypatch.setattr(
        biocell_app,
        "run_cell_analysis",
        lambda img: {
            "total_cells": 3,
            "total_cell_count": 3,
            "class_counts": {"Neoplastic": 2, "Inflammatory": 1},
            "centroids": [
                {"id": 1, "x": 10.0, "y": 10.0, "class": "Neoplastic"},
                {"id": 2, "x": 20.0, "y": 10.0, "class": "Neoplastic"},
                {"id": 3, "x": 25.0, "y": 25.0, "class": "Inflammatory"},
            ],
            "detected": [{"centroid": [10.0, 10.0], "class_name": "Neoplastic"}],
            "graph_metrics": {"total_nodes": 3, "total_edges": 1, "connected_components": 2},
            "validity": {"is_valid": False, "min_total_cells": 30, "warnings": ["low"]},
            "mask": mask,
            "time": 0.01,
        },
    )

    biocell_app.main()
    assert rendered_cards["Graph Nodes"] == "3"
    assert rendered_cards["Graph Edges"] == "1"
    assert rendered_cards["Avg Degree"] == "0.67"
    assert rendered_cards["Components"] == "2"
    assert len(rendered_images) == 2
    assert rendered_images[1].shape == (32, 32, 3)


def test_streamlit_apptest_headless_boot():
    """Headless Streamlit AppTest smoke test verifies zero uncaught exceptions on app startup."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=20)
    at.run()
    assert not at.exception, f"AppTest raised exception: {at.exception}"
    assert len(at.tabs) == 4
