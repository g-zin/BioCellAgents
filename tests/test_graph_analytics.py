"""Offline unit tests for src.tools.graph_analytics."""

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from src.tools.graph_analytics import (
    CellGraphAnalyticsTool,
    SpatialGraphAnalytics,
    load_he_patch,
)


class TestSpatialGraphAnalytics:
    """Behavior tests for spatial proximity graph construction and TME metrics."""

    def test_build_graph_fewer_than_two_cells(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        analyzer = SpatialGraphAnalytics(max_distance_px=50.0)

        res_empty = analyzer.build_graph_and_compute_metrics([], patch_index=0)
        assert res_empty["total_nodes"] == 0
        assert res_empty["total_edges"] == 0
        assert res_empty["avg_clustering_coefficient"] == 0.0
        assert res_empty["connected_components"] == 0
        assert res_empty["interaction_matrix"] == {}
        assert Path(res_empty["graph_image_path"]).exists()

        res_one = analyzer.build_graph_and_compute_metrics(
            [{"id": 1, "x": 10.0, "y": 20.0, "class": "Neoplastic"}],
            patch_index=1,
        )
        assert res_one["total_nodes"] == 1
        assert res_one["total_edges"] == 0
        assert res_one["connected_components"] == 1
        assert Path(res_one["graph_image_path"]).exists()

    def test_distance_threshold_boundary(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        analyzer = SpatialGraphAnalytics(max_distance_px=50.0)

        within = [
            {"id": 1, "x": 0.0, "y": 0.0, "class": "Neoplastic"},
            {"id": 2, "x": 30.0, "y": 40.0, "class": "Inflammatory"},
        ]
        res_in = analyzer.build_graph_and_compute_metrics(within, patch_index=2)
        assert res_in["total_edges"] == 1
        assert res_in["avg_neighbor_distance_px"] == 50.0
        assert res_in["connected_components"] == 1

        outside = [
            {"id": 1, "x": 0.0, "y": 0.0, "class": "Neoplastic"},
            {"id": 2, "x": 30.01, "y": 40.0, "class": "Inflammatory"},
        ]
        res_out = analyzer.build_graph_and_compute_metrics(outside, patch_index=3)
        assert res_out["total_edges"] == 0
        assert res_out["connected_components"] == 2

    def test_triangle_topology_density_and_interaction_matrix(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        analyzer = SpatialGraphAnalytics(max_distance_px=50.0)
        centroids = [
            {"id": 1, "x": 0.0, "y": 0.0, "class": "Neoplastic"},
            {"id": 2, "x": 30.0, "y": 0.0, "class": "Inflammatory"},
            {"id": 3, "x": 0.0, "y": 40.0, "class": "Neoplastic"},
        ]
        res = analyzer.build_graph_and_compute_metrics(centroids, patch_index=4)
        assert res["total_nodes"] == 3
        assert res["total_edges"] == 3
        assert res["avg_clustering_coefficient"] == 1.0
        assert res["connected_components"] == 1
        assert res["cellular_density_per_100x100px"] == round((3 / (256 * 256)) * 10000, 2)
        assert res["interaction_matrix"] == {
            "Inflammatory <-> Neoplastic": 2,
            "Neoplastic <-> Neoplastic": 1,
        }

    def test_immune_infiltration_metrics_exact_and_missing_classes(self):
        analyzer = SpatialGraphAnalytics()

        g_only_tumor = nx.Graph()
        g_only_tumor.add_node(1, pos=(0.0, 0.0), cell_class="Neoplastic")
        m_missing = analyzer.compute_immune_infiltration_metrics(g_only_tumor)
        assert m_missing["mean_immune_to_tumor_dist_px"] == 0.0
        assert m_missing["closest_immune_to_tumor_dist_px"] == 0.0
        assert m_missing["inflammatory_count"] == 0
        assert m_missing["neoplastic_count"] == 1

        g_both = nx.Graph()
        g_both.add_node(1, pos=(0.0, 0.0), cell_class="Neoplastic")
        g_both.add_node(2, pos=(100.0, 0.0), cell_class="Neoplastic")
        g_both.add_node(3, pos=(3.0, 4.0), cell_class="Inflammatory")
        g_both.add_node(4, pos=(100.0, 10.0), cell_class="Inflammatory")
        m_exact = analyzer.compute_immune_infiltration_metrics(g_both)
        assert m_exact["closest_immune_to_tumor_dist_px"] == pytest.approx(5.0)
        assert m_exact["mean_immune_to_tumor_dist_px"] == pytest.approx(7.5)
        assert m_exact["inflammatory_count"] == 2
        assert m_exact["neoplastic_count"] == 2

    def test_load_he_patch_missing_and_normalization(self, tmp_path):
        assert load_he_patch(str(tmp_path), 0) is None

        arr = np.linspace(0.0, 1.0, 2 * 16 * 16 * 3, dtype=np.float32).reshape(2, 16, 16, 3)
        np.save(tmp_path / "images.npy", arr)
        patch = load_he_patch(str(tmp_path), 1)
        assert patch is not None
        assert patch.dtype == np.uint8
        assert patch.shape == (16, 16, 3)
        assert int(patch.max()) <= 255
        assert int(patch.min()) >= 0

    def test_cell_graph_analytics_tool_primary_and_fallback(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        tool = CellGraphAnalyticsTool()

        err_res = tool._run(fold_dir=str(tmp_path / "missing"), patch_index=0)
        assert "error" in err_res

        dummy_img = np.zeros((32, 32, 3), dtype=np.uint8)
        monkeypatch.setattr(
            "src.tools.graph_analytics.load_he_patch", lambda fold_dir, patch_index: dummy_img
        )

        class _FakeDetector:
            def detect_and_classify(self, img):
                return [
                    {"cell_id": 1, "centroid": [10.0, 10.0], "class_name": "Neoplastic"}
                ], np.zeros((32, 32))

            def to_centroids(self, detected):
                return [{"id": 1, "x": 10.0, "y": 10.0, "class": "Neoplastic", "area_px": 10}]

            def class_counts(self, detected):
                return {"Neoplastic": 1}

        monkeypatch.setattr("src.tools.graph_analytics.TrainedCellDetector", _FakeDetector)
        ok_res = tool._run(fold_dir=str(tmp_path), patch_index=0)
        assert ok_res["total_cell_count"] == 1
        assert ok_res["class_counts"] == {"Neoplastic": 1}
        assert "validity_assessment" in ok_res
        assert ok_res["detection_source"] == "Trained DINOv2 segmentation + classification"

        class _BrokenDetector:
            def detect_and_classify(self, img):
                raise RuntimeError("missing checkpoint")

        monkeypatch.setattr("src.tools.graph_analytics.TrainedCellDetector", _BrokenDetector)
        monkeypatch.setattr(
            "src.tools.graph_analytics.parse_fold_masks",
            lambda fold_dir, patch_index: {
                "centroids": [
                    {"id": 1, "x": 5.0, "y": 5.0, "class": "Neoplastic", "area_px": 12},
                    {"id": 2, "x": 15.0, "y": 5.0, "class": "Inflammatory", "area_px": 14},
                ],
                "counts_by_class": {"Neoplastic": 1, "Inflammatory": 1},
                "total_cell_count": 2,
            },
        )
        fb_res = tool._run(fold_dir=str(tmp_path), patch_index=0)
        assert fb_res["total_cell_count"] == 2
        assert fb_res["detection_source"].startswith("Ground-truth masks (model fallback:")
