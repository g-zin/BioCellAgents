"""Offline smoke and chart-builder unit tests for biocell_app and app."""

import importlib

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


def test_biocell_app_chart_builders():
    """Chart builder functions in biocell_app should produce valid Plotly figures."""
    import biocell_app

    pie_fig = biocell_app.build_cell_pie({"Neoplastic": 12, "Inflammatory": 5, "Connective": 0})
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
        ]
    )
    assert isinstance(scatter_fig, go.Figure)
    assert len(scatter_fig.data) == 2
