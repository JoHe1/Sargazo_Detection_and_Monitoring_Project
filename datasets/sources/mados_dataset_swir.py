"""
datasets/sources/mados_dataset_swir.py
----------------------------------------
Versión extendida de MADOSDataset para tiles de 6 canales
(B, G, R, NIR, SWIR1, SWIR2) generados por MADOSPreprocessorSWIR.

Solo sobreescribe _reorder_channels() para manejar los 6 canales.
Todo lo demás (WeightedRandomSampler, augmentation, crop) es idéntico
al MADOSDataset original.

Uso:
    dataset = MADOSDatasetSWIR(
        root_path="datasets/data/Sargassum_Ready_Dataset_SWIR",
        split="train",
    )
    loader = dataset.get_loader(batch_size=8)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from core.config.paths import SARGASSUM_READY
from datasets.sources.mados_dataset import MADOSDataset

# Ruta al dataset SWIR — carpeta paralela al dataset original
SARGASSUM_READY_SWIR = Path(str(SARGASSUM_READY) + "_SWIR")


class MADOSDatasetSWIR(MADOSDataset):
    """
    Dataset MADOS con 6 canales espectrales (RGB + NIR + SWIR1 + SWIR2).

    Hereda todo de MADOSDataset y solo sobreescribe _reorder_channels()
    para reordenar correctamente los 6 canales:
        Orden ACOLITE:   (B, G, R, NIR, SWIR1, SWIR2)  → índices [0,1,2,3,4,5]
        Orden modelo:    (R, G, B, NIR, SWIR1, SWIR2)  → índices [2,1,0,3,4,5]

    Los canales SWIR no se reordenan porque no tienen equivalente en
    los pesos pre-entrenados de ImageNet — solo se mantienen en su
    posición original tras el intercambio RGB.
    """

    def __init__(
        self,
        root_path: str | Path = SARGASSUM_READY_SWIR,
        split: str = "train",
        image_size: int = 224,
        num_classes: int = 16,
    ) -> None:
        # Llamar al padre con la ruta del dataset SWIR
        super().__init__(
            root_path=root_path,
            split=split,
            image_size=image_size,
            num_classes=num_classes,
        )

    def _reorder_channels(self, img: np.ndarray) -> np.ndarray:
        """
        Reordena los canales de (B, G, R, NIR, SWIR1, SWIR2)
                              a (R, G, B, NIR, SWIR1, SWIR2).

        Los canales SWIR permanecen en posiciones 4 y 5 sin cambio.
        Solo se intercambia el orden RGB → se cambia B(0) con R(2).

        Args:
            img: array (H, W, 6) en orden (B, G, R, NIR, SWIR1, SWIR2)

        Returns:
            array (H, W, 6) en orden (R, G, B, NIR, SWIR1, SWIR2)
        """
        if img.shape[2] == 6:
            return img[:, :, [2, 1, 0, 3, 4, 5]]
        # Fallback por si algún tile no tiene SWIR (rellenado con ceros)
        return img[:, :, [2, 1, 0, 3]]
