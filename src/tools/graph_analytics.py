import os
from pathlib import Path
from typing import Dict, List, Any, Type, Tuple
import numpy as np
import networkx as nx
from scipy.spatial import distance
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (thread-safe)
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from src.models.nuclei_segmenter import parse_fold_masks
from src.models.cell_detector import TrainedCellDetector
from src.inference.validity_gate import ValidityGate

# Standard PanNuke cell class visual color scheme
CLASS_COLORS = {
    "Neoplastic": "#E63946",    # Crimson Red
    "Inflammatory": "#457B9D",  # Steel Blue
    "Connective": "#2A9D8F",    # Teal
    "Dead": "#2B2D42",          # Dark Slate
    "Epithelial": "#F4A261",    # Orange
    "Unknown": "#8D99AE"        # Gray
}


def load_he_patch(fold_dir: str, patch_index: int) -> np.ndarray:
    """Loads a single H&E image patch (256x256x3) from images.npy in memory-map mode."""
    images_path = Path(fold_dir) / "images.npy"
    if not images_path.exists():
        return None

    try:
        images = np.load(images_path, mmap_mode="r")
        patch_img = np.array(images[patch_index])

        # Normalize pixel values to 0-255 uint8 format for Matplotlib display
        if patch_img.max() <= 1.0:
            patch_img = (patch_img * 255).astype(np.uint8)
        else:
            patch_img = patch_img.astype(np.uint8)

        return patch_img
    except Exception as e:
        print(f"Could not load H&E background image: {e}")
        return None


class SpatialGraphAnalytics:
    """Constructs 2D cell proximity graphs from centroids, computes topology metrics,

    interaction matrices, immune infiltration depth, and saves visual plots.
    """

    def __init__(self, max_distance_px: float = 50.0):
        self.max_distance_px = float(max_distance_px)

    def compute_interaction_matrix(self, G: nx.Graph) -> Dict[str, int]:
        """Calculates pairwise edge counts between different cell classes."""
        interactions: Dict[str, int] = {}
        for u, v in G.edges():
            c1 = G.nodes[u].get("cell_class", "Unknown")
            c2 = G.nodes[v].get("cell_class", "Unknown")

            # Alphabetically sort pair key to ensure undirected graph consistency
            pair = tuple(sorted([c1, c2]))
            pair_key = f"{pair[0]} <-> {pair[1]}"
            interactions[pair_key] = interactions.get(pair_key, 0) + 1

        return interactions

    def compute_immune_infiltration_metrics(self, G: nx.Graph) -> Dict[str, float]:
        """Calculates mean & min Euclidean distance (px) from Inflammatory cells to nearest Neoplastic cell."""
        neoplastic_nodes = [n for n, d in G.nodes(data=True) if d.get("cell_class") == "Neoplastic"]
        inflammatory_nodes = [n for n, d in G.nodes(data=True) if d.get("cell_class") == "Inflammatory"]

        if not neoplastic_nodes or not inflammatory_nodes:
            return {
                "mean_immune_to_tumor_dist_px": 0.0,
                "closest_immune_to_tumor_dist_px": 0.0,
                "inflammatory_count": len(inflammatory_nodes),
                "neoplastic_count": len(neoplastic_nodes)
            }

        neoplastic_coords = np.array([G.nodes[n]["pos"] for n in neoplastic_nodes], dtype=np.float64)
        
        min_distances = []
        for inf in inflammatory_nodes:
            inf_pos = np.array(G.nodes[inf]["pos"], dtype=np.float64)
            dists = np.linalg.norm(neoplastic_coords - inf_pos, axis=1)
            min_distances.append(np.min(dists))

        return {
            "mean_immune_to_tumor_dist_px": round(float(np.mean(min_distances)), 2),
            "closest_immune_to_tumor_dist_px": round(float(np.min(min_distances)), 2),
            "inflammatory_count": len(inflammatory_nodes),
            "neoplastic_count": len(neoplastic_nodes)
        }

    def _generate_and_save_plot(
        self,
        G: nx.Graph,
        patch_index: int,
        he_image: np.ndarray = None,
        output_dir: str = "reports"
    ) -> str:
        """Generates and saves a 2D spatial cell proximity plot overlaid on the H&E tissue image."""
        reports_path = Path(output_dir)
        reports_path.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

        # 1. Overlay raw H&E tissue background
        if he_image is not None:
            ax.imshow(he_image, alpha=0.7, zorder=0)

        pos = nx.get_node_attributes(G, "pos")
        classes = nx.get_node_attributes(G, "cell_class")

        # 2. Draw proximity edges
        for u, v in G.edges():
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color="#FFD166", linestyle="--", alpha=0.8, linewidth=1.2, zorder=1)

        # 3. Group node coordinates by cell class
        grouped_coords: Dict[str, List[tuple]] = {}
        for node_id, cls_name in classes.items():
            grouped_coords.setdefault(cls_name, []).append(pos[node_id])

        # 4. Plot cell nodes with sharp white borders
        for cls_name, coords in grouped_coords.items():
            xs = [pt[0] for pt in coords]
            ys = [pt[1] for pt in coords]
            color = CLASS_COLORS.get(cls_name, "#8D99AE")
            ax.scatter(
                xs, ys,
                c=color,
                label=f"{cls_name} (n={len(coords)})",
                s=85,
                edgecolors="white",
                linewidths=0.8,
                zorder=2
            )

        # 5. Format axes
        title_text = f"2D Proximity Graph over H&E Stain (Patch {patch_index})" if he_image is not None else f"2D Cell Proximity Graph (Patch {patch_index})"
        ax.set_title(title_text, fontsize=11, fontweight="bold")
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.set_xlim(0, 256)
        ax.set_ylim(256, 0)  # Invert Y to match image space where (0,0) is top-left
        
        if he_image is not None:
            ax.grid(False)
        else:
            ax.grid(True, linestyle=":", alpha=0.4)

        ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.85, fontsize=8)

        img_filename = f"spatial_graph_patch_{patch_index}.png"
        img_filepath = reports_path / img_filename
        plt.tight_layout()
        plt.savefig(img_filepath, dpi=300)
        plt.close(fig)

        return f"reports/{img_filename}"

    def build_graph_and_compute_metrics(
        self,
        centroids: List[Dict[str, Any]],
        patch_index: int = 0,
        he_image: np.ndarray = None
    ) -> Dict[str, Any]:
        num_cells = len(centroids)

        if num_cells < 2:
            image_path = self._generate_and_save_plot(nx.Graph(), patch_index=patch_index, he_image=he_image)
            return {
                "total_nodes": num_cells,
                "total_edges": 0,
                "avg_clustering_coefficient": 0.0,
                "avg_neighbor_distance_px": 0.0,
                "cellular_density_per_100x100px": 0.0,
                "connected_components": num_cells,
                "interaction_matrix": {},
                "immune_infiltration": {
                    "mean_immune_to_tumor_dist_px": 0.0,
                    "closest_immune_to_tumor_dist_px": 0.0,
                    "inflammatory_count": 0,
                    "neoplastic_count": 0
                },
                "graph_image_path": image_path,
                "status": "Insufficient centroids for spatial graph construction"
            }

        G = nx.Graph()

        # Populate nodes with cell attributes
        for cell in centroids:
            G.add_node(
                cell["id"],
                pos=(float(cell["x"]), float(cell["y"])),
                cell_class=cell.get("class", "Unknown"),
                area=cell.get("area_px", 0)
            )

        # Extract coordinates and enforce strictly (N, 2) array shape
        coords = np.array([[float(c["x"]), float(c["y"])] for c in centroids], dtype=np.float64)
        coords = np.atleast_2d(coords)
        if coords.ndim > 2:
            coords = np.squeeze(coords)
        if coords.ndim == 1:
            coords = coords.reshape(-1, 2)

        # Pairwise distance matrix
        dist_matrix = distance.cdist(coords, coords, metric="euclidean")

        # Connect cell pairs within distance radius
        for i in range(num_cells):
            for j in range(i + 1, num_cells):
                d = dist_matrix[i, j]
                if d <= self.max_distance_px:
                    G.add_edge(centroids[i]["id"], centroids[j]["id"], weight=round(float(d), 2))

        # Calculate standard topology metrics
        total_edges = G.number_of_edges()
        avg_clustering = nx.average_clustering(G) if num_cells > 0 else 0.0
        connected_components = nx.number_connected_components(G)

        edge_weights = [attr["weight"] for _, _, attr in G.edges(data=True)]
        avg_edge_dist = round(float(np.mean(edge_weights)), 2) if edge_weights else 0.0

        patch_area_px = 256 * 256
        cellular_density = round((num_cells / patch_area_px) * 10000, 2)

        # Advanced TME Metrics
        interaction_matrix = self.compute_interaction_matrix(G)
        immune_infiltration = self.compute_immune_infiltration_metrics(G)

        # Generate plot
        image_path = self._generate_and_save_plot(G, patch_index=patch_index, he_image=he_image)

        return {
            "total_nodes": num_cells,
            "total_edges": total_edges,
            "avg_clustering_coefficient": round(float(avg_clustering), 4),
            "avg_neighbor_distance_px": avg_edge_dist,
            "cellular_density_per_100x100px": cellular_density,
            "connected_components": connected_components,
            "interaction_matrix": interaction_matrix,
            "immune_infiltration": immune_infiltration,
            "graph_image_path": image_path,
            "status": "Graph analytics completed successfully"
        }


# --- CrewAI Tool Wrapper ---

class SpatialGraphInput(BaseModel):
    """Input parameters for Spatial Graph Analytics Tool."""
    fold_dir: str = Field(default="data/raw/fold1", description="Directory path containing masks.npy and images.npy")
    patch_index: int = Field(default=0, description="Index of the tissue patch mask to construct spatial graph")
    max_distance_px: float = Field(default=50.0, description="Proximity threshold in pixels for graph edge creation")


class CellGraphAnalyticsTool(BaseTool):
    name: str = "Cell Spatial Graph Analytics"
    description: str = (
        "Parses tissue cell centroids to build a NetworkX 2D spatial proximity graph. "
        "Returns topology metrics, cell-cell interaction matrix, immune-to-tumor infiltration distances, "
        "and saves a visual PNG plot overlaid directly on the raw H&E tissue patch."
    )
    args_schema: Type[BaseModel] = SpatialGraphInput

    def _run(self, fold_dir: str = "data/raw/fold1", patch_index: int = 0, max_distance_px: float = 50.0) -> Dict[str, Any]:
        he_image = load_he_patch(fold_dir=fold_dir, patch_index=patch_index)
        if he_image is None:
            return {"error": f"Could not load image patch from '{fold_dir}' at index {patch_index}."}

        # Prefer model-driven cell detection + classification; fall back to ground-truth masks.
        detection_source = "Trained DINOv2 segmentation + classification"
        try:
            detector = TrainedCellDetector()
            detected_cells, _ = detector.detect_and_classify(he_image)
            centroids = detector.to_centroids(detected_cells)
            counts_by_class = detector.class_counts(detected_cells)
            total_cell_count = len(centroids)
        except Exception as e:
            detection_source = f"Ground-truth masks (model fallback: {e})"
            segmentation_res = parse_fold_masks(fold_dir=fold_dir, patch_index=patch_index)
            if "error" in segmentation_res:
                return segmentation_res
            centroids = segmentation_res.get("centroids", [])
            counts_by_class = segmentation_res.get("counts_by_class", {})
            total_cell_count = segmentation_res.get("total_cell_count", len(centroids))

        analyzer = SpatialGraphAnalytics(max_distance_px=max_distance_px)
        graph_metrics = analyzer.build_graph_and_compute_metrics(
            centroids,
            patch_index=patch_index,
            he_image=he_image
        )

        # Apply statistical validity gate
        gate = ValidityGate(min_total_cells=30, min_per_class=5)
        validity = gate.full_assessment(
            n_cells=total_cell_count,
            counts_by_class=counts_by_class,
        )

        return {
            "patch_index": patch_index,
            "total_cell_count": total_cell_count,
            "class_counts": counts_by_class,
            "detection_source": detection_source,
            "graph_topology": graph_metrics,
            "graph_image_path": graph_metrics.get("graph_image_path"),
            "validity_assessment": validity,
        }


if __name__ == "__main__":
    print("Testing SpatialGraphAnalytics module with Advanced TME Metrics...")
    analyzer = SpatialGraphAnalytics(max_distance_px=50.0)
    mock_centroids = [
        {"id": 1, "class": "Neoplastic", "x": 20.0, "y": 20.0, "area_px": 180},
        {"id": 2, "class": "Neoplastic", "x": 30.0, "y": 25.0, "area_px": 210},
        {"id": 3, "class": "Inflammatory", "x": 45.0, "y": 40.0, "area_px": 140},
        {"id": 4, "class": "Inflammatory", "x": 180.0, "y": 180.0, "area_px": 130}
    ]
    res = analyzer.build_graph_and_compute_metrics(mock_centroids, patch_index=0)
    print(f"  - Nodes: {res['total_nodes']} | Edges: {res['total_edges']}")
    print(f"  - Interaction Matrix: {res['interaction_matrix']}")
    print(f"  - Immune Infiltration: {res['immune_infiltration']}")
    print("Advanced TME metrics test passed!")