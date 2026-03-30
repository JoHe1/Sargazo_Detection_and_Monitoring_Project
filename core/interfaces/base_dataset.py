"""
core/interfaces/base_dataset.py
---------------------------------
Interfaz base para todos los datasets del proyecto.

Uso:
    Crea una subclase e implementa los métodos abstractos:
        - load()
        - preprocess()

    get_loader() y get_split_info() están implementados aquí
    y se heredan en MADOSDataset, SentinelDataset, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
from torch.utils.data import DataLoader, Dataset


class BaseDataset(ABC, Dataset):
    """
    Clase base para todos los datasets del proyecto.

    Hereda de torch.utils.data.Dataset, por lo que es directamente
    compatible con DataLoader de PyTorch sin ningún cambio.

    Métodos ABSTRACTOS (obligatorio implementar en cada subclase):
        load()        — carga los archivos del disco y llena self.samples
        preprocess()  — aplica normalización y transformaciones
        __len__()     — número de muestras
        __getitem__() — devuelve (imagen, máscara) en el índice idx

    Métodos IMPLEMENTADOS (heredados sin cambios):
        get_loader()      — construye y devuelve un DataLoader
        get_split_info()  — estadísticas básicas del split
    """

    def __init__(self, root_path: str | Path, split: str = "train") -> None:
        """
        Args:
            root_path: ruta raíz del dataset (carpeta con train/val/test)
            split:     "train", "val" o "test"
        """
        super().__init__()
        self.root_path = Path(root_path)
        self.split = split
        self.samples: list = []  # lista de (img_path, mask_path) — rellenar en load()

    # ------------------------------------------------------------------
    # MÉTODOS ABSTRACTOS
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """
        Carga la lista de muestras disponibles desde disco.
        Debe rellenar self.samples con pares (img_path, mask_path).
        Se llama automáticamente en el __init__ de cada subclase.
        """
        ...

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica el preprocesamiento específico de este dataset a una imagen.

        Cada dataset puede tener normalización distinta:
            - MADOS:    valores de reflectancia / 10000 * 5
            - Sentinel: valores float en [0, 1] desde API

        Args:
            image: array (H, W, C) sin preprocesar

        Returns:
            array (H, W, C) preprocesado y listo para el modelo
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Número total de muestras en el split."""
        ...

    @abstractmethod
    def __getitem__(self, idx: int):
        """
        Devuelve la muestra en el índice idx.

        Returns:
            tuple (image_tensor, mask_tensor)
            image_tensor: torch.FloatTensor (C, H, W)
            mask_tensor:  torch.LongTensor  (H, W)
        """
        ...

    # ------------------------------------------------------------------
    # MÉTODOS IMPLEMENTADOS
    # ------------------------------------------------------------------

    def get_loader(
        self,
        batch_size: int = 8,
        shuffle: Optional[bool] = None,
        num_workers: int = 2,
        pin_memory: bool = True,
    ) -> DataLoader:
        """
        Construye y devuelve un DataLoader para este dataset.

        Por defecto shuffle=True para train, False para val/test.

        Args:
            batch_size:  número de muestras por batch
            shuffle:     si None, se decide automáticamente según el split
            num_workers: hilos de carga en paralelo
            pin_memory:  acelera la transferencia a GPU

        Returns:
            DataLoader listo para usar en el bucle de entrenamiento
        """
        if shuffle is None:
            shuffle = self.split == "train"

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def get_split_info(self) -> dict:
        """
        Devuelve estadísticas básicas del split actual.
        Útil para logging antes de entrenar.

        Returns:
            dict con num_samples, split, root_path, dataset_class
        """
        return {
            "dataset_class": self.__class__.__name__,
            "split": self.split,
            "root_path": str(self.root_path),
            "num_samples": len(self),
        }