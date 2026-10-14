from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import warnings
import numpy as np
from scipy import ndimage
import cv2

warnings.warn(
    "NucleiSegmenter (StarDist-based) is deprecated. "
    "Use CellposeSegmenter from src.models.cellpose_segmenter instead.",
    DeprecationWarning,
    stacklevel=2,
)

PANNUKE_CLASSES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]


def parse_fold_masks(fold_dir: Union[str, Path], patch_index: int) -> Dict[str, Any]:
    """Loads and processes the segmentation mask for a specific patch index from a PanNuke fold directory.

    Args:
        fold_dir: Path or string to the fold directory (e.g., 'data/raw/fold1').
        patch_index: Zero-based index of the target patch.

    Returns:
        Dict[str, Any]: Dictionary containing 'centroids' (standardized to y, x ordering),
        'total_cell_count', and 'counts_by_class'.
    """
    fold_path = Path(fold_dir)
    mask_file = fold_path / "masks.npy"
    patch_idx = int(patch_index)

    # 1. Locate and load mask array patch
    if not mask_file.exists():
        npz_file = fold_path / "masks.npz"
        if npz_file.exists():
            data = np.load(npz_file)
            first_key = list(data.keys())[0]
            mask_patch = data[first_key][patch_idx]
        else:
            return {
                "error": f"Could not locate masks file at: {mask_file.resolve()}"
            }
    else:
        masks = np.load(mask_file, mmap_mode="r")
        mask_patch = masks[patch_idx]

    centroids: List[Dict[str, Any]] = []
    counts_by_class: Dict[str, int] = {
        cls_name: 0 for cls_name in PANNUKE_CLASSES
    }
    cell_id = 1

    # 2. Extract cell instances using native PanNuke instance IDs (Shape: 256 x 256 x 6)
    if mask_patch.ndim == 3 and mask_patch.shape[-1] >= 5:
        for channel_idx, class_name in enumerate(PANNUKE_CLASSES):
            channel_mask = mask_patch[:, :, channel_idx]
            instance_ids = np.unique(channel_mask)
            instance_ids = instance_ids[instance_ids != 0]  # Exclude background

            for inst_id in instance_ids:
                ys, xs = np.where(channel_mask == inst_id)
                area = len(ys)

                if area >= 5:  # Filter noise artifacts
                    cy = float(np.mean(ys))
                    cx = float(np.mean(xs))

                    centroids.append({
                        "id": cell_id,
                        "y": cy,  # Standardized array row coordinate (y)
                        "x": cx,  # Standardized array column coordinate (x)
                        "class": class_name,
                        "area_px": int(area),
                    })
                    counts_by_class[class_name] += 1
                    cell_id += 1
    else:
        # Fallback for 2D single-channel mask
        non_bg_mask = (mask_patch > 0).astype(np.uint8)
        labeled_array, num_features = ndimage.label(non_bg_mask)

        if num_features > 0:
            objects = ndimage.find_objects(labeled_array)
            for i, slc in enumerate(objects, 1):
                if slc is not None:
                    cell_mask = labeled_array[slc] == i
                    area = int(np.sum(cell_mask))
                    if area >= 5:
                        ys, xs = np.where(labeled_array == i)
                        cy = float(np.mean(ys))
                        cx = float(np.mean(xs))

                        centroids.append({
                            "id": cell_id,
                            "y": cy,  # Standardized array row coordinate (y)
                            "x": cx,  # Standardized array column coordinate (x)
                            "class": "Unknown",
                            "area_px": area,
                        })
                        cell_id += 1

    return {
        "centroids": centroids,
        "total_cell_count": len(centroids),
        "counts_by_class": counts_by_class,
    }


class NucleiSegmenter:
    """STEP 1: Segments cells from masks/images -> extracts centroids and individual cell masks.
    
    Pipeline:
    1. segment_from_mask(mask) -> centroids + cell masks
    2. get_all_cell_crops(image, centroids) -> individual cell crops (48x48)
    3. Pass crops to DINOv2CellClassifier for classification
    """

    def __init__(self, use_stardist_fallback: bool = True):
        self.use_stardist = use_stardist_fallback
        self.stardist_model = None

    def _init_stardist(self):
        """Lazy initialization for StarDist2D pre-trained model."""
        if self.stardist_model is None:
            from stardist.models import StarDist2D

            print(
                "Loading pre-trained StarDist2D model ('2D_versatile_he')..."
            )
            self.stardist_model = StarDist2D.from_pretrained("2D_versatile_he")

    def segment_from_mask(self, mask_patch: np.ndarray) -> Dict[str, Any]:
        """STEP 1A: Segment cells from PanNuke ground-truth mask.
        
        Args:
            mask_patch: Multi-channel mask array (H x W x C) where C >= 5 for PanNuke classes
        
        Returns:
            - centroids: List[Dict] with id, y, x, class, area_px
            - cell_masks: Dict[cell_id -> binary mask]
            - count: Total cells found
        """
        centroids: List[Dict[str, Any]] = []
        cell_masks: Dict[int, np.ndarray] = {}
        cell_id = 1

        if mask_patch.ndim == 3 and mask_patch.shape[-1] >= 5:
            for c in range(min(5, mask_patch.shape[-1])):
                channel_mask = mask_patch[:, :, c]
                instance_ids = np.unique(channel_mask)
                instance_ids = instance_ids[instance_ids != 0]

                for inst_id in instance_ids:
                    ys, xs = np.where(channel_mask == inst_id)
                    area = len(ys)
                    if area >= 5:
                        cy = float(np.mean(ys))
                        cx = float(np.mean(xs))
                        
                        centroids.append({
                            "id": cell_id,
                            "y": cy,
                            "x": cx,
                            "class": PANNUKE_CLASSES[c],
                            "area_px": int(area)
                        })
                        
                        cell_binary_mask = np.zeros_like(channel_mask, dtype=np.uint8)
                        cell_binary_mask[channel_mask == inst_id] = 1
                        cell_masks[cell_id] = cell_binary_mask
                        cell_id += 1
        else:
            combined_mask = (mask_patch > 0).astype(np.uint8)
            labeled_array, num_features = ndimage.label(combined_mask)
            
            if num_features > 0:
                for i in range(1, num_features + 1):
                    ys, xs = np.where(labeled_array == i)
                    area = len(ys)
                    if area >= 5:
                        centroids.append({
                            "id": cell_id,
                            "y": float(np.mean(ys)),
                            "x": float(np.mean(xs)),
                            "class": "Unknown",
                            "area_px": int(area)
                        })
                        
                        cell_binary_mask = np.zeros_like(labeled_array, dtype=np.uint8)
                        cell_binary_mask[labeled_array == i] = 1
                        cell_masks[cell_id] = cell_binary_mask
                        cell_id += 1

        return {"centroids": centroids, "cell_masks": cell_masks, "count": len(centroids)}

    def segment_raw_image(self, rgb_image: np.ndarray) -> Dict[str, Any]:
        """STEP 1B: Segment cells from raw H&E image using StarDist.
        
        Args:
            rgb_image: RGB image array (H x W x 3)
        
        Returns:
            - centroids: List[Dict] with id, y, x, class, area_px
            - cell_masks: Dict[cell_id -> binary mask]
            - labels: Labeled segmentation map
        """
        self._init_stardist()
        from csbdeep.utils import normalize

        normalized_img = normalize(rgb_image)
        labels, details = self.stardist_model.predict_instances(normalized_img)

        centroids: List[Dict[str, Any]] = []
        cell_masks: Dict[int, np.ndarray] = {}
        
        if "points" in details:
            for cell_id, pt in enumerate(details["points"], 1):
                centroids.append({
                    "id": cell_id,
                    "y": float(pt[0]),
                    "x": float(pt[1]),
                    "class": "Unknown",
                    "area_px": -1
                })
                cell_binary_mask = (labels == cell_id).astype(np.uint8)
                cell_masks[cell_id] = cell_binary_mask

        return {"centroids": centroids, "cell_masks": cell_masks, "count": len(centroids), "labels": labels}

    def get_cell_crop(self, centroid: Dict[str, Any], image: np.ndarray, crop_size: int = 48) -> np.ndarray:
        """STEP 2: Extract a fixed-size square crop around a cell centroid.
        
        Args:
            centroid: Dict with 'y', 'x' keys (as returned by segment_from_mask)
            image: Input image (H x W x C)
            crop_size: Output crop size (default 48x48)
        
        Returns:
            Cropped patch (crop_size x crop_size x C)
        """
        h, w = image.shape[:2]
        cy, cx = int(centroid["y"]), int(centroid["x"])
        half = crop_size // 2

        y1 = max(0, cy - half)
        y2 = min(h, cy + half)
        x1 = max(0, cx - half)
        x2 = min(w, cx + half)

        pad_top = max(0, half - cy)
        pad_bottom = max(0, (cy + half) - h)
        pad_left = max(0, half - cx)
        pad_right = max(0, (cx + half) - w)

        crop = image[y1:y2, x1:x2]

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)

        return crop

    def get_all_cell_crops(self, image: np.ndarray, centroids: List[Dict[str, Any]], crop_size: int = 48) -> List[Dict[str, Any]]:
        """STEP 2: Extract crops for all cells from an image.
        
        Args:
            image: Input image (H x W x C)
            centroids: List of centroid dicts (as returned by segment_from_mask)
            crop_size: Size of the square crop
        
        Returns:
            List of dicts with 'id', 'crop', 'centroid', 'class' - ready for DINOv2CellClassifier
        """
        crops = []
        for centroid in centroids:
            crop = self.get_cell_crop(centroid, image, crop_size)
            crops.append({
                "id": centroid["id"],
                "crop": crop,
                "centroid": centroid,
                "class": centroid.get("class", "Unknown")
            })
        return crops