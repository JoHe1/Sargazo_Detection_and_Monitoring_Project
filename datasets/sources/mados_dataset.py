"""
datasets/sources/mados_dataset.py
-----------------------------------
Dataset de PyTorch para el dataset MADOS ya preprocesado.

Cambios v4:
    - VSCP corregido a nivel de batch como en MariNeXt (Kikaki et al., 2024).
      Implementado via collate_fn: tile i recibe sargazo del tile i+N//2
      del mismo batch. Genera N//2 imagenes sinteticas adicionales por batch.
    - sargassum_weight=10.0 mantenido.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from core.config.paths import SARGASSUM_READY
from datasets.base.base_dataset import SargassoBaseDataset
import albumentations as A

CLASES_SARGAZO = {2, 3}


class MADOSDataset(SargassoBaseDataset):

    def __init__(
        self,
        root_path: str | Path = SARGASSUM_READY,
        split: str = "train",
        image_size: int = 224,
        num_classes: int = 16,
    ) -> None:
        super().__init__(root_path, split, image_size, num_classes)
        self.load()

        self._sarg_indices = [
            i for i, (_, mp) in enumerate(self.samples)
            if self._has_sargassum(mp)
        ]

        info = self.get_split_info()
        n_sarg = len(self._sarg_indices)
        print(f"[MADOSDataset] {info['split'].upper()} — "
              f"{info['num_samples']} muestras "
              f"({n_sarg} con sargazo, {info['num_samples'] - n_sarg} sin sargazo)")

    def load(self) -> None:
        img_dir  = self.root_path / self.split / "images"
        mask_dir = self.root_path / self.split / "masks"
        if not img_dir.exists():
            raise FileNotFoundError(f"No se encuentra: {img_dir}\n¿Has ejecutado MADOSPreprocessor?")
        self.samples = []
        for img_path in sorted(img_dir.glob("*.npy")):
            mask_path = mask_dir / img_path.name
            if mask_path.exists():
                self.samples.append((img_path, mask_path))
            else:
                print(f"  [AVISO] Sin mascara: {img_path.name}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image = self._reorder_channels(image)
        image = self._normalize(image)
        return image

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.split != "train":
            return image, mask
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        ])
        augmented = transform(image=image, mask=mask)
        return augmented["image"], augmented["mask"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]
        img  = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.int64)
        mask = np.clip(mask, 0, self.num_classes - 1)
        img       = self.preprocess(img)
        img, mask = self._crop(img, mask)
        img, mask = self._augment(img, mask)
        img       = np.transpose(img, (2, 0, 1))
        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)

    # ------------------------------------------------------------------
    # VSCP a nivel de batch (MariNeXt, Kikaki et al. 2024)
    # ------------------------------------------------------------------

    def vscp_collate_fn(self, batch: list) -> tuple[torch.Tensor, torch.Tensor]:
        """
        VSCP correcto a nivel de batch.

        Dado un batch de N muestras:
          N_vscp = N // 2 imagenes sinteticas nuevas.
          Imagen sintetica i = imagen i + pixeles sargazo de imagen i+N_vscp.
          Batch final = N originales + N_vscp sinteticas.

        Solo activo en split=='train'.
        """
        images = torch.stack([b[0] for b in batch])
        masks  = torch.stack([b[1] for b in batch])

        if self.split != "train":
            return images, masks

        N      = images.shape[0]
        N_vscp = N // 2
        if N_vscp == 0:
            return images, masks

        clases_tensor = torch.tensor(list(CLASES_SARGAZO), device=masks.device)
        synth_images, synth_masks = [], []

        for i in range(N_vscp):
            j = i + N_vscp
            img_base   = images[i].clone()
            mask_base  = masks[i].clone()
            img_donor  = images[j]
            mask_donor = masks[j]

            sarg_px = torch.isin(mask_donor, clases_tensor)  # (H, W)

            if sarg_px.sum() > 0:
                sarg_px_img = sarg_px.unsqueeze(0).expand_as(img_base)
                img_base[sarg_px_img]  = img_donor[sarg_px_img]
                mask_base[sarg_px]     = mask_donor[sarg_px]

            synth_images.append(img_base)
            synth_masks.append(mask_base)

        final_images = torch.cat([images, torch.stack(synth_images)], dim=0)
        final_masks  = torch.cat([masks,  torch.stack(synth_masks)],  dim=0)
        return final_images, final_masks

    # ------------------------------------------------------------------
    # get_loader
    # ------------------------------------------------------------------

    def get_loader(
        self,
        batch_size:           int   = 8,
        shuffle:              bool | None = None,
        num_workers:          int   = 2,
        pin_memory:           bool  = True,
        use_weighted_sampler: bool | None = None,
        sargassum_weight:     float = 10.0,
    ) -> DataLoader:

        if use_weighted_sampler is None:
            use_weighted_sampler = (self.split == "train")

        collate = self.vscp_collate_fn if self.split == "train" else None

        if use_weighted_sampler:
            weights = self._compute_sample_weights(sargassum_weight)
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
            return DataLoader(self, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate)
        else:
            if shuffle is None:
                shuffle = (self.split == "train")
            return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate)

    def _has_sargassum(self, mask_path: Path) -> bool:
        try:
            mask = np.load(mask_path)
            return bool(np.isin(mask, list(CLASES_SARGAZO)).any())
        except Exception:
            return False

    def _compute_sample_weights(self, sargassum_weight: float) -> list[float]:
        print(f"[MADOSDataset] Calculando pesos (sargazo_weight={sargassum_weight})...")
        weights = []
        for _, mask_path in self.samples:
            w = sargassum_weight if self._has_sargassum(mask_path) else 1.0
            weights.append(w)
        n_sarg = sum(1 for w in weights if w > 1.0)
        print(f"[MADOSDataset] {n_sarg} tiles sargazo (w={sargassum_weight}), "
              f"{len(weights)-n_sarg} sin sargazo (w=1.0)")
        return weights