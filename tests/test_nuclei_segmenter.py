"""Offline unit tests for src.models.nuclei_segmenter."""

import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from src.models.nuclei_segmenter import NucleiSegmenter, parse_fold_masks


class TestNucleiSegmenter:
    """Behavior tests for mask parsing, centroid extraction, and cell cropping."""

    def test_parse_fold_masks_3d_npy_and_missing(self, tmp_path):
        missing = parse_fold_masks(tmp_path, 0)
        assert "error" in missing

        masks = np.zeros((2, 64, 64, 6), dtype=np.int32)
        masks[0, 10:20, 10:20, 0] = 1  # Neoplastic (100 px)
        masks[0, 30:40, 30:40, 1] = 2  # Inflammatory (100 px)
        masks[0, 0:2, 0:2, 2] = 3  # Too small (<5 px), filtered
        np.save(tmp_path / "masks.npy", masks)

        res = parse_fold_masks(tmp_path, 0)
        assert res["total_cell_count"] == 2
        assert res["counts_by_class"]["Neoplastic"] == 1
        assert res["counts_by_class"]["Inflammatory"] == 1
        assert res["counts_by_class"]["Connective"] == 0

    def test_parse_fold_masks_npz_2d_fallback(self, tmp_path):
        masks = np.zeros((1, 64, 64), dtype=np.int32)
        masks[0, 10:20, 10:20] = 1
        np.savez(tmp_path / "masks.npz", masks=masks)

        res = parse_fold_masks(tmp_path, 0)
        assert res["total_cell_count"] == 1
        assert res["centroids"][0]["class"] == "Unknown"

    def test_segment_from_mask_and_crops(self):
        seg = NucleiSegmenter(use_stardist_fallback=False)
        mask_patch = np.zeros((64, 64, 6), dtype=np.int32)
        mask_patch[5:15, 5:15, 0] = 1
        mask_patch[40:50, 40:50, 4] = 2

        parsed = seg.segment_from_mask(mask_patch)
        assert parsed["count"] == 2
        assert len(parsed["cell_masks"]) == 2

        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        crops = seg.get_all_cell_crops(img, parsed["centroids"], crop_size=32)
        assert len(crops) == 2
        assert all(c["crop"].shape == (32, 32, 3) for c in crops)

    def test_segment_from_mask_rejects_under_5_channel_3d_and_odd_crop_size(self):
        import pytest

        seg = NucleiSegmenter(use_stardist_fallback=False)
        with pytest.raises(ValueError, match="at least 5 channels"):
            seg.segment_from_mask(np.zeros((64, 64, 3), dtype=np.int32))

        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        odd_crop = seg.get_cell_crop({"y": 30, "x": 30}, img, crop_size=49)
        assert odd_crop.shape == (49, 49, 3)

        oob_crop = seg.get_cell_crop({"y": 200, "x": 200}, img, crop_size=49)
        assert oob_crop.shape == (49, 49, 3)
        assert np.all(oob_crop == 0)
