"""
datasets/base/base_dataset.py
------------------------------
Implementación base compartida entre MADOSDataset y SentinelDataset.

IMPORTANTE — distinción de archivos:
    core/interfaces/base_dataset.py  → el ABC con las firmas (el contrato)
    datasets/base/base_dataset.py    → este archivo, lógica compartida concreta

Solo contiene lo que es literalmente idéntico en ambos datasets:
    - _normalize()         normalización de reflectancias Sentinel-2
    - _reorder_channels()  (B, G, R, NIR) → (R, G, B, NIR)
    - _random_crop()       recorte aleatorio para train
    - _center_crop()       recorte central determinista para val/test

Cada subclase implementa load(), __len__() y __getitem__() por su cuenta
porque la estructura de archivos de MADOS y Sentinel son completamente distintas.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from core.interfaces.base_dataset import BaseDataset


class SargassoBaseDataset(BaseDataset):
    """
    Clase base concreta para todos los datasets del proyecto.

    Hereda de BaseDataset (ABC) e implementa los métodos de preprocesamiento
    que son compartidos. Los métodos load(), __len__() y __getitem__()
    siguen siendo abstractos y los implementa cada subclase.

    Uso:
        No instanciar directamente. Usar MADOSDataset o SentinelDataset.
    """

    def __init__(
        self,
        root_path: str | Path,
        split: str = "train",
        image_size: int = 224,
        num_classes: int = 16,
    ) -> None:
        """
        Args:
            root_path:   ruta raíz del dataset procesado (con subcarpetas train/val/test)
            split:       "train", "val" o "test"
            image_size:  tamaño del crop cuadrado que verá el modelo (224 por defecto)
            num_classes: número de clases semánticas (16 para MADOS)
        """
        super().__init__(root_path, split)
        self.image_size  = image_size
        self.num_classes = num_classes

    # ------------------------------------------------------------------
    # MÉTODOS COMPARTIDOS — idénticos en MADOS y Sentinel
    # ------------------------------------------------------------------

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Normaliza los valores de reflectancia de una imagen Sentinel-2.

        Pasos:
            1. Elimina NaN, +inf y -inf que puedan venir del sensor o ACOLITE
            2. Si los valores están en escala de reflectancia entera (> 10.0),
               divide entre 10000 para pasar a [0, 1]
            3. Amplifica x5 para mejorar el contraste visual y la convergencia
               del modelo (igual que en train_swin.py original)
            4. Recorta al rango [0, 1]

        Args:
            img: array (H, W, C) con valores de reflectancia sin normalizar

        Returns:
            array (H, W, C) normalizado en [0, 1]
        """
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
        if img.max() > 10.0:
            img = img / 10000.0
        return np.clip(img * 5.0, 0.0, 1.0)

    def _reorder_channels(self, img: np.ndarray) -> np.ndarray:
        """
        Reordena los canales de (B, G, R, NIR) a (R, G, B, NIR).

        MADOS guarda las bandas en orden ACOLITE: Azul(492), Verde(560),
        Rojo(665), NIR(833). PyTorch y los pesos pre-entrenados de ImageNet
        esperan (R, G, B, ...), así que hay que reordenar antes de entrenar.

        Args:
            img: array (H, W, 4) en orden (B, G, R, NIR)

        Returns:
            array (H, W, 4) en orden (R, G, B, NIR)
        """
        return img[:, :, [2, 1, 0, 3]]

    def _random_crop(
        self, img: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Recorte aleatorio para el split de entrenamiento.

        Genera posiciones de inicio aleatorias dentro del tile completo,
        lo que efectivamente multiplica la variedad de datos por época.
        La misma posición se aplica a imagen y máscara para mantener
        la correspondencia píxel a píxel.

        Args:
            img:  array (H, W, C) normalizado
            mask: array (H, W) con IDs de clase

        Returns:
            tuple (img_crop, mask_crop) de tamaño (image_size, image_size)
        """
        h, w = mask.shape
        y0 = random.randint(0, h - self.image_size)
        x0 = random.randint(0, w - self.image_size)
        return (
            img [y0:y0 + self.image_size, x0:x0 + self.image_size, :],
            mask[y0:y0 + self.image_size, x0:x0 + self.image_size],
        )

    def _center_crop(
        self, img: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Recorte central determinista para val y test.

        Siempre produce el mismo recorte dado el mismo tile, lo que hace
        la evaluación reproducible entre distintos experimentos.

        Args:
            img:  array (H, W, C) normalizado
            mask: array (H, W) con IDs de clase

        Returns:
            tuple (img_crop, mask_crop) de tamaño (image_size, image_size)
        """
        h, w = mask.shape
        y0 = (h - self.image_size) // 2
        x0 = (w - self.image_size) // 2
        return (
            img [y0:y0 + self.image_size, x0:x0 + self.image_size, :],
            mask[y0:y0 + self.image_size, x0:x0 + self.image_size],
        )

    def _crop(
        self, img: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Selecciona el tipo de crop según el split actual.

        Centraliza la lógica de decisión para que __getitem__() de cada
        subclase solo tenga que llamar a self._crop(img, mask).

        Returns:
            tuple (img_crop, mask_crop) con el crop apropiado para el split
        """
        if self.split == "train":
            return self._random_crop(img, mask)
        return self._center_crop(img, mask)

    def _augment(
        self, img: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Data augmentation básico para el split de entrenamiento.

        Solo flips horizontales y verticales, que son geométricamente válidos
        para imágenes satelitales (no hay orientación canónica desde el espacio).
        No se aplica rotación ni color jitter porque podría alterar los índices
        espectrales (NDVI, FAI) que el modelo aprende a detectar.

        Args:
            img:  array (H, W, C)
            mask: array (H, W)

        Returns:
            tuple (img, mask) con augmentation aplicado (o sin cambios si split != train)
        """
        if self.split != "train":
            return img, mask

        if random.random() > 0.5:
            img  = np.flip(img,  axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        if random.random() > 0.5:
            img  = np.flip(img,  axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        return img, mask

    # ------------------------------------------------------------------
    # MÉTODOS ABSTRACTOS — cada subclase los implementa
    # ------------------------------------------------------------------

    def load(self) -> None:
        raise NotImplementedError("Implementa load() en la subclase.")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Implementa preprocess() en la subclase.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        raise NotImplementedError("Implementa __getitem__() en la subclase.")