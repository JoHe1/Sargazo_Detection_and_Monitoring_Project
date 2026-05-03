"""
datasets/sources/mados_dataset.py
-----------------------------------
Dataset de PyTorch para el dataset MADOS ya preprocesado.

Cambios respecto a v1:
    - get_loader() acepta parámetro use_weighted_sampler (default True en train)
      que activa WeightedRandomSampler para sobremuestrear tiles con sargazo.
    - _compute_sample_weights() calcula el peso de cada tile según si contiene
      sargazo (clases 2 o 3) o no.
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

    Novedad principal: WeightedRandomSampler opcional en get_loader().
    Cuando está activo, los tiles con sargazo aparecen ~5x más por época
    que los tiles sin sargazo, sin duplicar datos en disco.

    Uso:
        dataset = MADOSDataset(split="train")
        loader  = dataset.get_loader(batch_size=4)  # oversampling activado por defecto
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

        info = self.get_split_info()
        n_sarg = sum(1 for _, mp in self.samples if self._has_sargassum(mp))
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

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Aplica Data Augmentation severo "al vuelo" usando Albumentations.
        Solo se aplica si el split es 'train'.
        """
        if self.split != "train":
            return image, mask

        # Inicializar el pipeline solo una vez (podrías pasarlo al __init__, 
        # pero aquí está bien para no romper tu SargassoBaseDataset)
        import albumentations as A
        
        # 1. Flip Horizontal (50% de probabilidad)
        # 2. Flip Vertical (50% de probabilidad)
        # 3. Rotación Aleatoria de 90 grados (50% de probabilidad)
        # 4. Transposición (intercambiar filas y columnas)
        # 5. Ligeros cambios de brillo/contraste para independizar al modelo de la iluminación
        
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,  # Alterar brillo un +/- 10%
                contrast_limit=0.1,    # Alterar contraste un +/- 10%
                p=0.3
            ),
        ])

        # Albumentations requiere que la imagen sea (H, W, C)
        # y devuelve un diccionario.
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
        sargassum_weight:      float = 5.0,
    ) -> DataLoader:
        """
        Construye un DataLoader con oversampling opcional de tiles con sargazo.

        Args:
            batch_size:           muestras por batch
            shuffle:              si None, True para train y False para val/test
            num_workers:          hilos de carga
            pin_memory:           acelera transferencia a GPU
            use_weighted_sampler: si None, se activa solo en split=="train"
                                  True  → activa oversampling siempre
                                  False → desactiva siempre
            sargassum_weight:     cuántas veces más probable es samplear un tile
                                  con sargazo respecto a uno sin sargazo.
                                  5.0 significa que los 101 tiles con sargazo
                                  aparecen ~5x más que los 1332 sin sargazo.

        Returns:
            DataLoader listo para el bucle de entrenamiento
        """
        # Decidir si usar sampler
        if use_weighted_sampler is None:
            use_weighted_sampler = (self.split == "train")

        if use_weighted_sampler:
            weights = self._compute_sample_weights(sargassum_weight)
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
            )
            # WeightedRandomSampler y shuffle son mutuamente excluyentes
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
        """Devuelve True si la máscara contiene píxeles de sargazo (cl. 2 o 3)."""
        try:
            mask = np.load(mask_path)
            return bool(np.isin(mask, list(CLASES_SARGAZO)).any())
        except Exception:
            return False

    def _compute_sample_weights(self, sargassum_weight: float) -> list[float]:
        """
        Calcula el peso de muestreo de cada tile.

        Tiles con sargazo  → peso = sargassum_weight
        Tiles sin sargazo  → peso = 1.0

        Efecto práctico con sargassum_weight=5 y los datos actuales:
            101 tiles × 5.0  = 505
            1332 tiles × 1.0 = 1332
            Total = 1837 pesos

            P(tile con sargazo)  = 505/1837 ≈ 27%  (antes era 7%)
            P(tile sin sargazo)  = 1332/1837 ≈ 73%

        Cada tile con sargazo aparece ~3.8x más veces por época que antes.
        """
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