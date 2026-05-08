"""
datasets/sources/mados_dataset.py
-----------------------------------
Dataset de PyTorch para el dataset MADOS ya preprocesado.

Cambios respecto a v2:
    - VSCP (Very Simple Copy-Paste) añadido en _augment().
      En cada batch, con p=0.3, copia los píxeles anotados de sargazo
      de otro tile aleatorio encima del tile actual. Opera en memoria,
      no toca las máscaras en disco.
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
    """
    Dataset de PyTorch para el dataset MADOS ya preprocesado.

    Novedades:
        - WeightedRandomSampler opcional en get_loader().
        - VSCP (Very Simple Copy-Paste) en _augment() para train.
    """

    def __init__(
        self,
        root_path: str | Path = SARGASSUM_READY,
        split: str = "train",
        image_size: int = 224,
        num_classes: int = 16,
    ) -> None:
        super().__init__(root_path, split, image_size, num_classes)
        self.load()

        # Precomputar índices de tiles con sargazo para VSCP
        self._sarg_indices = [
            i for i, (_, mp) in enumerate(self.samples)
            if self._has_sargassum(mp)
        ]

        info = self.get_split_info()
        n_sarg = len(self._sarg_indices)
        print(f"[MADOSDataset] {info['split'].upper()} — "
              f"{info['num_samples']} muestras "
              f"({n_sarg} con sargazo, {info['num_samples'] - n_sarg} sin sargazo)")

    # ------------------------------------------------------------------
    # Implementación de métodos abstractos
    # ------------------------------------------------------------------

    def load(self) -> None:
        img_dir  = self.root_path / self.split / "images"
        mask_dir = self.root_path / self.split / "masks"

        if not img_dir.exists():
            raise FileNotFoundError(
                f"No se encuentra: {img_dir}\n"
                f"¿Has ejecutado MADOSPreprocessor?"
            )

        self.samples = []
        for img_path in sorted(img_dir.glob("*.npy")):
            mask_path = mask_dir / img_path.name
            if mask_path.exists():
                self.samples.append((img_path, mask_path))
            else:
                print(f"  [AVISO] Sin máscara: {img_path.name}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        image = self._reorder_channels(image)
        image = self._normalize(image)
        return image

    def _apply_vscp(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Very Simple Copy-Paste (VSCP) — inspirado en MariNeXt (Kikaki et al., 2024).

        Selecciona un tile aleatorio con sargazo del dataset, copia sus píxeles
        anotados (clases 2 y 3) encima de la imagen actual. Opera en memoria,
        no modifica las máscaras en disco.

        Args:
            image : (H, W, C) float32 normalizada
            mask  : (H, W)    int64

        Returns:
            image y mask con sargazo sintético añadido
        """
        if not self._sarg_indices:
            return image, mask

        # Seleccionar tile donante aleatorio con sargazo
        donor_idx = np.random.choice(self._sarg_indices)
        donor_img_path, donor_mask_path = self.samples[donor_idx]

        donor_img  = np.load(donor_img_path).astype(np.float32)
        donor_mask = np.load(donor_mask_path).astype(np.int64)
        donor_mask = np.clip(donor_mask, 0, self.num_classes - 1)

        donor_img  = self.preprocess(donor_img)
        donor_img, donor_mask = self._crop(donor_img, donor_mask)

        # Máscara de píxeles anotados como sargazo en el donante
        sarg_pixels = np.isin(donor_mask, list(CLASES_SARGAZO))

        if sarg_pixels.sum() == 0:
            return image, mask

        # Copiar píxeles de sargazo del donante sobre la imagen actual
        image_out = image.copy()
        mask_out  = mask.copy()
        image_out[sarg_pixels] = donor_img[sarg_pixels]
        mask_out[sarg_pixels]  = donor_mask[sarg_pixels]

        return image_out, mask_out

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Aplica Data Augmentation al vuelo usando Albumentations + VSCP.
        Solo se aplica si el split es 'train'.
        """
        if self.split != "train":
            return image, mask

        # 1. VSCP con probabilidad 0.3
        if np.random.random() < 0.3:
            image, mask = self._apply_vscp(image, mask)

        # 2. Albumentations (igual que antes)
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
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

        img        = self.preprocess(img)
        img, mask  = self._crop(img, mask)
        img, mask  = self._augment(img, mask)
        img        = np.transpose(img, (2, 0, 1))

        return (
            torch.tensor(img,  dtype=torch.float32),
            torch.tensor(mask, dtype=torch.long),
        )

    # ------------------------------------------------------------------
    # get_loader con WeightedRandomSampler
    # ------------------------------------------------------------------

    def get_loader(
        self,
        batch_size:            int   = 8,
        shuffle:               bool | None = None,
        num_workers:           int   = 2,
        pin_memory:            bool  = True,
        use_weighted_sampler:  bool | None = None,
        sargassum_weight:      float = 10.0,
    ) -> DataLoader:
        if use_weighted_sampler is None:
            use_weighted_sampler = (self.split == "train")

        if use_weighted_sampler:
            weights = self._compute_sample_weights(sargassum_weight)
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
            )
            return DataLoader(
                self,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        else:
            if shuffle is None:
                shuffle = (self.split == "train")
            return DataLoader(
                self,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _has_sargassum(self, mask_path: Path) -> bool:
        try:
            mask = np.load(mask_path)
            return bool(np.isin(mask, list(CLASES_SARGAZO)).any())
        except Exception:
            return False

    def _compute_sample_weights(self, sargassum_weight: float) -> list[float]:
        print(f"[MADOSDataset] Calculando pesos de muestreo "
              f"(sargazo_weight={sargassum_weight})...")
        weights = []
        for _, mask_path in self.samples:
            w = sargassum_weight if self._has_sargassum(mask_path) else 1.0
            weights.append(w)
        n_sarg = sum(1 for w in weights if w > 1.0)
        print(f"[MADOSDataset] Pesos calculados: "
              f"{n_sarg} tiles con sargazo (w={sargassum_weight}), "
              f"{len(weights)-n_sarg} sin sargazo (w=1.0)")
        return weights