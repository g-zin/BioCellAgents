"""Offline unit tests for src.data.dataset."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.dataset import (
    CELL_TYPES,
    ORGAN_TYPES,
    PanNukeCellPatchDataset,
    PanNukeOrganDataset,
    PanNukeSegmentationDataset,
)


class TestPanNukeDatasets:
    """Behavior tests for PanNuke datasets using synthetic temporary data."""

    def test_constants(self):
        assert CELL_TYPES == ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
        assert len(ORGAN_TYPES) == 19
        assert "Breast" in ORGAN_TYPES
        assert "Pancreatic" in ORGAN_TYPES

    def test_pannuke_cell_patch_dataset(self, tmp_path):
        crops_dir = tmp_path / "data" / "processed" / "fold1" / "crops"
        crops_dir.mkdir(parents=True)
        crop_path = crops_dir / "cell_0.png"
        Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8)).save(crop_path)

        meta_path = tmp_path / "data" / "processed" / "metadata.csv"
        pd.DataFrame(
            [
                {
                    "crop_path": "data/processed/fold1/crops/cell_0.png",
                    "fold": "fold1",
                    "image_idx": 0,
                    "class_id": 2,
                    "class_name": "Connective",
                }
            ]
        ).to_csv(meta_path, index=False)

        ds = PanNukeCellPatchDataset(metadata_path=meta_path, fold="1", is_train=False, img_size=64)
        assert len(ds) == 1
        img, label = ds[0]
        assert img.shape == (3, 64, 64)
        assert int(label.item()) == 2

        with pytest.raises(ValueError, match="No samples found"):
            PanNukeCellPatchDataset(metadata_path=meta_path, fold="fold3", is_train=False)

    def test_pannuke_segmentation_and_organ_datasets(self, tmp_path):
        raw_fold = tmp_path / "data" / "raw" / "fold3"
        raw_fold.mkdir(parents=True)
        meta_path = tmp_path / "data" / "processed" / "metadata.csv"
        meta_path.parent.mkdir(parents=True)
        meta_path.write_text("crop_path,fold,image_idx,class_id,class_name\n", encoding="utf-8")

        imgs = np.random.rand(2, 64, 64, 3).astype(np.float32)
        masks = np.zeros((2, 64, 64, 6), dtype=np.int32)
        masks[0, 10:20, 10:20, 0] = 1
        masks[1, 20:30, 20:30, 3] = 2
        tissues = np.array(["Breast", b"Liver"], dtype=object)

        np.save(raw_fold / "images.npy", imgs)
        np.save(raw_fold / "masks.npy", masks)
        np.save(raw_fold / "tissues.npy", tissues)

        seg_ds = PanNukeSegmentationDataset(
            metadata_path=meta_path, fold="3", is_train=True, img_size=64
        )
        assert len(seg_ds) == 2
        seg_img, seg_mask = seg_ds[0]
        assert seg_img.shape == (3, 64, 64)
        assert seg_mask.shape == (64, 64)
        assert seg_mask.dtype == torch.long
        assert set(seg_mask.unique().tolist()).issubset({0, 1})

        org_ds = PanNukeOrganDataset(
            metadata_path=meta_path, fold=["fold3"], is_train=False, img_size=64
        )
        assert len(org_ds) == 2
        org_img, org_label = org_ds[1]
        assert org_img.shape == (3, 64, 64)
        assert int(org_label.item()) == ORGAN_TYPES.index("Liver")
