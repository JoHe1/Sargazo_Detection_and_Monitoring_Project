"""
datasets/sources/sentinel_dataset.py
--------------------------------------
Dataset de PyTorch para patches de Sentinel-2 descargados de Copernicus.

Este archivo tiene UNA SOLA responsabilidad:
    Leer los patches .tiff y servirlos como tensores para inferencia.

Para descargar y trocear imágenes de Copernicus:
    datasets/preprocessors/sentinel/sentinel_downloader.py

Para aplicar la land mask:
    datasets/preprocessors/sentinel/land_mask_processor.py

Flujo completo:
    [SentinelDownloader]  → patches .tiff en sentinel_downloads/
    [LandMaskProcessor]   → patches .tiff enmascarados (opcional)
    [SentinelDataset]     → tensores para el modelo
    [Modelo]              → predicciones de segmentación
    [App web]             → visualización en mapa

Diferencias clave respecto a MADOSDataset:
    - Sin máscaras ground truth (imágenes nuevas, sin etiquetar)
    - Lee .tiff con rasterio en lugar de .npy
    - NO reordena canales: el evalscript ya devuelve (R, G, B, NIR)
    - NO hace augmentation: siempre es inferencia, no entrenamiento
    - __getitem__ devuelve solo el tensor de imagen, sin máscara
    - get_patch_metadata() permite georreferenciar las predicciones
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch

from datasets.base.base_dataset import SargassoBaseDataset


class SentinelDataset(SargassoBaseDataset):
    """
    Dataset de PyTorch para patches de Sentinel-2 listos para inferencia.

    Hereda de SargassoBaseDataset para reutilizar _normalize().
    No usa _reorder_channels() ni _augment() porque no son necesarios.
    """

    def __init__(
        self,
        patches_dir: str | Path,
        image_size:  int = 224,
    ) -> None:
        """
        Args:
            patches_dir: carpeta con los patches .tiff generados por
                         SentinelDownloader (con o sin land mask aplicada)
            image_size:  tamaño del crop cuadrado que verá el modelo (224)
        """
        super().__init__(
            root_path=patches_dir,
            split="test",
            image_size=image_size,
            num_classes=0,
        )
        self.patches_dir = Path(patches_dir)

        if not self.patches_dir.exists():
            raise FileNotFoundError(
                f"No se encuentra la carpeta de patches: {self.patches_dir}\n"
                f"¿Has ejecutado SentinelDownloader antes?"
            )

        self.load()
        print(f"[SentinelDataset] {len(self.samples)} patches en: {self.patches_dir.name}")

    # ------------------------------------------------------------------
    # Implementación de métodos abstractos
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Escanea la carpeta de patches y rellena self.samples.
        self.samples es lista de (patch_path, None) — None = sin máscara GT.
        """
        patches = sorted(self.patches_dir.glob("*.tiff"))
        self.samples = [(p, None) for p in patches]

        if not self.samples:
            print(f"  [AVISO] No se encontraron .tiff en: {self.patches_dir}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Normaliza un patch de Sentinel-2.

        El evalscript ya devuelve (R, G, B, NIR) — no hay que reordenar.
        Solo normalizamos las reflectancias al rango [0, 1].
        """
        return self._normalize(image)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Devuelve el patch como tensor PyTorch.

        Returns:
            FloatTensor (4, H, W) — canales R, G, B, NIR
        """
        patch_path, _ = self.samples[idx]

        with rasterio.open(patch_path) as src:
            img = src.read().astype(np.float32)  # (C, H, W)
            img = np.transpose(img, (1, 2, 0))   # → (H, W, C)

        img = self.preprocess(img)

        # Center crop si el patch es más grande que image_size
        h, w = img.shape[:2]
        if h >= self.image_size and w >= self.image_size:
            y0  = (h - self.image_size) // 2
            x0  = (w - self.image_size) // 2
            img = img[y0:y0 + self.image_size, x0:x0 + self.image_size, :]

        img = np.transpose(img, (2, 0, 1))  # (H, W, C) → (C, H, W)

        return torch.tensor(img, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Utilidades específicas de inferencia
    # ------------------------------------------------------------------

    def get_patch_path(self, idx: int) -> Path:
        """
        Devuelve la ruta del patch en el índice idx.
        Útil para leer los metadatos geoespaciales tras la inferencia.
        """
        return self.samples[idx][0]

    def get_patch_metadata(self, idx: int) -> dict:
        """
        Devuelve los metadatos geoespaciales del patch.

        Necesario para georreferenciar las predicciones en el mapa web.

        Returns:
            dict con crs, transform, bounds, width, height y nombre
        """
        patch_path = self.get_patch_path(idx)
        with rasterio.open(patch_path) as src:
            return {
                "name":      patch_path.name,
                "crs":       str(src.crs),
                "transform": list(src.transform),
                "bounds":    list(src.bounds),
                "width":     src.width,
                "height":    src.height,
                "bands":     src.count,
            }