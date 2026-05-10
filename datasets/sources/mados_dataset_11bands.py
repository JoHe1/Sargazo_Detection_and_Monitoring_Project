"""
datasets/sources/mados_dataset_11bands.py
------------------------------------------
Dataset PyTorch para tiles de 11 canales Sentinel-2 generados por
MADOSPreprocessor11Bands.

Canales en el .npy (orden ACOLITE/preprocesador):
    0  B1  443nm   Coastal Aerosol
    1  B2  492nm   Azul
    2  B3  560nm   Verde
    3  B4  665nm   Rojo
    4  B5  704nm   Red-Edge 1
    5  B6  740nm   Red-Edge 2
    6  B7  783nm   Red-Edge 3
    7  B8  833nm   NIR
    8  B8A 865nm   NIR narrow
    9  B11 1610nm  SWIR1
    10 B12 2190nm  SWIR2

Reordenación en carga para compatibilidad con pesos ImageNet:
    Posiciones 0-2 → R(B4), G(B3), B(B2)  [canales 3, 2, 1 del .npy]
    Posiciones 3-10 → resto en orden: B1, B5, B6, B7, B8, B8A, B11, B12

Todo lo demás (VSCP, WeightedSampler, augmentation, crop) es
idéntico al MADOSDataset original.

Uso:
    dataset = MADOSDataset11Bands(
        root_path="datasets/data/Sargassum_Ready_Dataset_11bands",
        split="train",
    )
    loader = dataset.get_loader(batch_size=8)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from core.config.paths import SARGASSUM_READY
from datasets.sources.mados_dataset import MADOSDataset

SARGASSUM_READY_11BANDS = Path(str(SARGASSUM_READY) + "_11bands")


class MADOSDataset11Bands(MADOSDataset):
    """
    Dataset MADOS con 11 canales espectrales Sentinel-2 completos.

    Hereda todo de MADOSDataset y solo sobreescribe _reorder_channels()
    para reordenar los 11 canales correctamente:

        Orden .npy:   [B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12]
                       [0,  1,  2,  3,  4,  5,  6,  7,  8,   9,  10]

        Orden modelo: [R,  G,  B,  B1, B5, B6, B7, B8, B8A, B11, B12]
                       [3,  2,  1,  0,  4,  5,  6,  7,  8,   9,  10]

    Los primeros 3 canales (R, G, B) se mapean a los pesos RGB
    pre-entrenados de Swin-Base en ImageNet.
    El resto se inicializa con la media de los pesos RGB (estrategia
    estándar para canales sin equivalente en ImageNet).
    """

    def __init__(
        self,
        root_path: str | Path = SARGASSUM_READY_11BANDS,
        split: str = "train",
        image_size: int = 224,
        num_classes: int = 16,
    ) -> None:
        super().__init__(
            root_path=root_path,
            split=split,
            image_size=image_size,
            num_classes=num_classes,
        )

    def _reorder_channels(self, img: np.ndarray) -> np.ndarray:
        """
        Reordena de orden .npy [B1,B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12]
                    a orden modelo [R, G, B, B1,B5,B6,B7,B8,B8A,B11,B12].

        Índices .npy:  B1=0, B2=1, B3=2, B4=3, B5=4, B6=5,
                       B7=6, B8=7, B8A=8, B11=9, B12=10

        Reordenación:  [3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10]
                        R  G  B  B1 B5 B6 B7 B8 B8A B11 B12
        """
        if img.shape[2] == 11:
            return img[:, :, [3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10]]
        # Fallback a 4 canales si algo falla
        return img[:, :, [3, 2, 1, 7]]
