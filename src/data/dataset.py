"""PyTorch Dataset classes and PanNuke label constants for BioCellAgents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.transforms import clean_organ_label

CELL_TYPES: List[str] = [
    "Neoplastic",
    "Inflammatory",
    "Connective",
    "Dead",
    "Epithelial",
]

ORGAN_TYPES: List[str] = [
    "Adrenal_gland",
    "Bile-duct",
    "Bladder",
    "Breast",
    "Cervix",
    "Colon",
    "Esophagus",
    "HeadNeck",
    "Kidney",
    "Liver",
    "Lung",
    "Ovarian",
    "Pancreatic",
    "Prostate",
    "Skin",
    "Stomach",
    "Testis",
    "Thyroid",
    "Uterus",
]

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _normalize_folds(fold: Union[str, int, Sequence[Union[str, int]]]) -> List[str]:
    """Normalize fold argument ('1,2', 3, ['fold1']) into ['fold1', ...]."""
    items: List[str]
    if isinstance(fold, str):
        items = [part.strip() for part in fold.split(",") if part.strip()]
    elif isinstance(fold, int):
        items = [str(fold)]
    else:
        items = [str(part).strip() for part in fold if str(part).strip()]

    normalized: List[str] = []
    for item in items:
        if item.isdigit():
            normalized.append(f"fold{item}")
        else:
            normalized.append(item)
    return normalized


def _resolve_raw_dir(metadata_path: Union[str, Path]) -> Path:
    """Resolve the raw PanNuke directory relative to metadata_path."""
    meta = Path(metadata_path)
    candidate = meta.parent.parent / "raw"
    if candidate.exists():
        return candidate
    return Path("data/raw")


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert an image array to uint8 HxWx3."""
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr


class PanNukeCellPatchDataset(Dataset):
    """Dataset of single-nucleus crops and 5-class cell type labels."""

    def __init__(
        self,
        metadata_path: Union[str, Path] = "data/processed/metadata.csv",
        fold: Union[str, int, Sequence[Union[str, int]]] = "fold1",
        is_train: bool = True,
        img_size: int = 224,
    ) -> None:
        super().__init__()
        self.is_train = is_train
        self.img_size = img_size
        self.metadata_path = Path(metadata_path)

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found at: {self.metadata_path.resolve()}")

        df = pd.read_csv(self.metadata_path)
        folds = _normalize_folds(fold)
        self.data = df[df["fold"].isin(folds)].reset_index(drop=True)
        if len(self.data) == 0:
            raise ValueError(f"No samples found in metadata for fold(s): {folds}")

        if self.is_train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
                ]
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[idx]
        image_path = Path(str(row["crop_path"]))
        if not image_path.is_absolute():
            image_path = (self.metadata_path.parent.parent.parent / image_path).resolve()

        class_id = int(row["class_id"])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(class_id, dtype=torch.long)
        return image_tensor, label_tensor


class PanNukeSegmentationDataset(Dataset):
    """Dataset of PanNuke H&E tissue patches and 6-class semantic masks."""

    def __init__(
        self,
        metadata_path: Union[str, Path] = "data/processed/metadata.csv",
        fold: Union[str, int, Sequence[Union[str, int]]] = "fold1",
        is_train: bool = True,
        img_size: int = 224,
    ) -> None:
        super().__init__()
        self.is_train = is_train
        self.img_size = img_size
        self.raw_dir = _resolve_raw_dir(metadata_path)
        self.folds = _normalize_folds(fold)

        self._images: List[Any] = []
        self._masks: List[Any] = []
        self._samples: List[Tuple[int, int]] = []

        for f_name in self.folds:
            fold_dir = self.raw_dir / f_name
            img_path = fold_dir / "images.npy"
            mask_path = fold_dir / "masks.npy"
            if not img_path.exists() or not mask_path.exists():
                raise FileNotFoundError(f"Missing images.npy or masks.npy in {fold_dir.resolve()}")
            imgs = np.load(img_path, mmap_mode="r")
            msks = np.load(mask_path, mmap_mode="r")
            fold_idx = len(self._images)
            self._images.append(imgs)
            self._masks.append(msks)
            for patch_idx in range(len(imgs)):
                self._samples.append((fold_idx, patch_idx))

        if not self._samples:
            raise ValueError(f"No segmentation samples found for fold(s): {self.folds}")

        self.img_transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fold_idx, patch_idx = self._samples[idx]
        raw_img = _to_uint8_rgb(np.array(self._images[fold_idx][patch_idx]))
        raw_mask = np.array(self._masks[fold_idx][patch_idx])

        if raw_mask.ndim == 3:
            semantic_mask = np.zeros(raw_mask.shape[:2], dtype=np.int64)
            for ch in range(min(len(CELL_TYPES), raw_mask.shape[-1])):
                semantic_mask[raw_mask[:, :, ch] > 0] = ch + 1
        else:
            semantic_mask = raw_mask.astype(np.int64)

        resized_mask = cv2.resize(
            semantic_mask.astype(np.uint8),
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int64)

        img_pil = Image.fromarray(raw_img).convert("RGB")
        if self.is_train:
            if np.random.rand() < 0.5:
                img_pil = img_pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                resized_mask = np.fliplr(resized_mask).copy()
            if np.random.rand() < 0.5:
                img_pil = img_pil.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                resized_mask = np.flipud(resized_mask).copy()

        image_tensor = self.img_transform(img_pil)
        mask_tensor = torch.from_numpy(resized_mask).long()
        return image_tensor, mask_tensor


class PanNukeOrganDataset(Dataset):
    """Dataset of PanNuke H&E tissue patches and 19-class organ labels."""

    def __init__(
        self,
        metadata_path: Union[str, Path] = "data/processed/metadata.csv",
        fold: Union[str, int, Sequence[Union[str, int]]] = "fold1",
        is_train: bool = True,
        img_size: int = 224,
    ) -> None:
        super().__init__()
        self.is_train = is_train
        self.img_size = img_size
        self.raw_dir = _resolve_raw_dir(metadata_path)
        self.folds = _normalize_folds(fold)
        self._organ_to_idx = {name: i for i, name in enumerate(ORGAN_TYPES)}

        self._images: List[Any] = []
        self._samples: List[Tuple[int, int, int]] = []

        for f_name in self.folds:
            fold_dir = self.raw_dir / f_name
            img_path = fold_dir / "images.npy"
            tissue_path = fold_dir / "tissues.npy"
            if not img_path.exists() or not tissue_path.exists():
                raise FileNotFoundError(
                    f"Missing images.npy or tissues.npy in {fold_dir.resolve()}"
                )
            imgs = np.load(img_path, mmap_mode="r")
            tissues = np.load(tissue_path, allow_pickle=True)
            fold_idx = len(self._images)
            self._images.append(imgs)
            for patch_idx in range(len(imgs)):
                label_str = clean_organ_label(tissues[patch_idx])
                if label_str in self._organ_to_idx:
                    self._samples.append((fold_idx, patch_idx, self._organ_to_idx[label_str]))

        if not self._samples:
            raise ValueError(f"No organ samples found for fold(s): {self.folds}")

        if self.is_train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
                ]
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fold_idx, patch_idx, label_idx = self._samples[idx]
        raw_img = _to_uint8_rgb(np.array(self._images[fold_idx][patch_idx]))
        img_pil = Image.fromarray(raw_img).convert("RGB")
        image_tensor = self.transform(img_pil)
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        return image_tensor, label_tensor
