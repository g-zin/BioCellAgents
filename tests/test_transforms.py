"""Unit tests for BioCellAgents data transforms."""
import numpy as np
import pytest


class TestCleanOrganLabel:
    """Tests for clean_organ_label."""

    def test_normal_string(self):
        from src.data.transforms import clean_organ_label
        assert clean_organ_label("Liver") == "Liver"

    def test_bytes_input(self):
        from src.data.transforms import clean_organ_label
        assert clean_organ_label(b"Kidney") == "Kidney"

    def test_numpy_array(self):
        from src.data.transforms import clean_organ_label
        arr = np.array("Brain")
        assert clean_organ_label(arr) == "Brain"

    def test_quoted_string(self):
        from src.data.transforms import clean_organ_label
        assert clean_organ_label("'Heart'") == "Heart"
        assert clean_organ_label('"Lung"') == "Lung"

    def test_none_input(self):
        from src.data.transforms import clean_organ_label
        assert clean_organ_label(None) == "Unknown"

    def test_empty_string(self):
        from src.data.transforms import clean_organ_label
        assert clean_organ_label("") == "Unknown"

    def test_whitespace(self):
        from src.data.transforms import clean_organ_label
        assert clean_organ_label("  Pancreatic  ") == "Pancreatic"


class TestCropNucleus:
    """Tests for crop_nucleus."""

    def test_center_crop(self):
        from src.data.transforms import crop_nucleus
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        crop = crop_nucleus(image, centroid=(50, 50), crop_size=48)
        assert crop.shape == (48, 48, 3)

    def test_edge_crop_with_padding(self):
        from src.data.transforms import crop_nucleus
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Centroid near edge should still produce full crop
        crop = crop_nucleus(image, centroid=(5, 5), crop_size=48)
        assert crop.shape == (48, 48, 3)

    def test_corner_crop(self):
        from src.data.transforms import crop_nucleus
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        crop = crop_nucleus(image, centroid=(0, 0), crop_size=48)
        assert crop.shape == (48, 48, 3)

    def test_different_crop_sizes(self):
        from src.data.transforms import crop_nucleus
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        for size in [16, 32, 48, 64, 128]:
            crop = crop_nucleus(image, centroid=(100, 100), crop_size=size)
            assert crop.shape == (size, size, 3)


class TestExtractCellCrops:
    """Tests for extract_cell_crops."""

    def test_single_cell(self):
        from src.data.transforms import extract_cell_crops
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        masks = np.zeros((100, 100), dtype=np.int32)
        masks[40:60, 40:60] = 1  # One cell
        crops = extract_cell_crops(image, masks, crop_size=48, min_area=10)
        assert len(crops) == 1
        assert crops[0].shape == (48, 48, 3)

    def test_multiple_cells(self):
        from src.data.transforms import extract_cell_crops
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        masks = np.zeros((200, 200), dtype=np.int32)
        masks[20:40, 20:40] = 1
        masks[100:120, 100:120] = 2
        masks[150:170, 150:170] = 3
        crops = extract_cell_crops(image, masks, crop_size=48, min_area=10)
        assert len(crops) == 3

    def test_min_area_filtering(self):
        from src.data.transforms import extract_cell_crops
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        masks = np.zeros((100, 100), dtype=np.int32)
        masks[50:52, 50:52] = 1  # Tiny cell (4 pixels)
        crops = extract_cell_crops(image, masks, crop_size=48, min_area=10)
        assert len(crops) == 0  # Filtered out

    def test_empty_mask(self):
        from src.data.transforms import extract_cell_crops
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        masks = np.zeros((100, 100), dtype=np.int32)
        crops = extract_cell_crops(image, masks, crop_size=48)
        assert len(crops) == 0


class TestExtractOrganPatches:
    """Tests for extract_organ_patches."""

    def test_basic_extraction(self):
        from src.data.transforms import extract_organ_patches
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        patches = extract_organ_patches(image, patch_size=128, stride=128)
        assert len(patches) > 0
        assert all(p.shape == (128, 128, 3) for p in patches)

    def test_stride_covers_image(self):
        from src.data.transforms import extract_organ_patches
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        patches = extract_organ_patches(image, patch_size=128, stride=64)
        # Should have more patches with smaller stride
        patches_large = extract_organ_patches(image, patch_size=128, stride=128)
        assert len(patches) >= len(patches_large)

    def test_small_image(self):
        from src.data.transforms import extract_organ_patches
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        patches = extract_organ_patches(image, patch_size=64, stride=32)
        assert len(patches) > 0