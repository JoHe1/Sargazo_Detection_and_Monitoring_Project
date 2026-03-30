"""
datasets/preprocessors/sentinel/land_mask_processor.py
--------------------------------------------------------
Aplica una máscara de tierra a patches de Sentinel-2.

Enmascara los píxeles de tierra (y opcionalmente islas pequeñas)
poniendo sus valores a 0, dejando solo los píxeles de océano visibles.
Esto evita que el modelo confunda vegetación terrestre con sargazo.

Los shapefiles de Natural Earth (ne_10m_land y ne_10m_minor_islands)
están en datasets/preprocessors/land_mask/ y son datos estáticos
que no cambian entre experimentos.

Flujo de uso:
    # Después de SentinelDownloader.download_and_tile():
    processor = LandMaskProcessor()
    n = processor.apply_to_directory(
        input_dir  = "datasets/raw_data/sentinel_downloads/patches_2025-06-13",
        output_dir = "datasets/raw_data/sentinel_downloads/patches_2025-06-13_masked",
    )

    # O sobre un único patch:
    processor.apply_to_file(input_path, output_path)

Shapefiles necesarios en LAND_MASK_DIR:
    ne_10m_land.shp           — masas de tierra principales
    ne_10m_minor_islands.shp  — islas pequeñas no incluidas en el anterior
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask

from core.config.paths import LAND_MASK_DIR


class LandMaskProcessor:
    """
    Enmascara los píxeles de tierra en imágenes Sentinel-2.

    Carga los shapefiles de Natural Earth una sola vez en el constructor
    y los reutiliza para todos los patches, evitando leerlos en disco
    en cada iteración (que era el problema del script original).

    Uso:
        processor = LandMaskProcessor()

        # Procesar una carpeta entera:
        n = processor.apply_to_directory(input_dir, output_dir)

        # Procesar un único archivo:
        processor.apply_to_file(input_path, output_path)
    """

    def __init__(
        self,
        land_mask_dir: str | Path = LAND_MASK_DIR,
        include_minor_islands: bool = True,
    ) -> None:
        """
        Args:
            land_mask_dir:         carpeta con los shapefiles de Natural Earth.
                                   Por defecto usa LAND_MASK_DIR de paths.py
            include_minor_islands: si True, también enmascara islas pequeñas.
                                   Recomendado True para el Caribe.
        """
        land_mask_dir = Path(land_mask_dir)

        shp_land   = land_mask_dir / "ne_10m_land.shp"
        shp_islands = land_mask_dir / "ne_10m_minor_islands.shp"

        if not shp_land.exists():
            raise FileNotFoundError(
                f"No se encuentra ne_10m_land.shp en: {land_mask_dir}\n"
                f"Descárgalo de https://www.naturalearthdata.com/downloads/10m-physical-vectors/"
            )

        print("[LandMaskProcessor] Cargando shapefiles de Natural Earth...")
        geo_land = gpd.read_file(shp_land)

        if include_minor_islands and shp_islands.exists():
            geo_islands = gpd.read_file(shp_islands)
            self.world = pd.concat([geo_land, geo_islands], ignore_index=True)
            print("[LandMaskProcessor] Cargados: tierra + islas pequeñas")
        else:
            self.world = geo_land
            print("[LandMaskProcessor] Cargados: solo tierra principal")

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def apply_to_directory(
        self,
        input_dir:  str | Path,
        output_dir: str | Path,
        pattern:    str = "*.tiff",
    ) -> int:
        """
        Aplica la land mask a todos los patches de una carpeta.

        Args:
            input_dir:  carpeta con los patches .tiff de entrada
            output_dir: carpeta donde guardar los patches enmascarados
            pattern:    patrón glob para filtrar archivos (por defecto *.tiff)

        Returns:
            número de patches procesados correctamente
        """
        input_dir  = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        patches = sorted(input_dir.glob(pattern))
        if not patches:
            print(f"[LandMaskProcessor] No se encontraron archivos en: {input_dir}")
            return 0

        print(f"[LandMaskProcessor] Procesando {len(patches)} patches...")
        procesados = 0
        errores    = 0

        for patch_path in patches:
            out_path = output_dir / patch_path.name
            try:
                self.apply_to_file(patch_path, out_path)
                procesados += 1
            except Exception as e:
                print(f"  [ERROR] {patch_path.name}: {e}")
                errores += 1

        print(f"[LandMaskProcessor] Completado: {procesados} OK, {errores} errores")
        return procesados

    def apply_to_file(
        self,
        input_path:  str | Path,
        output_path: str | Path,
    ) -> None:
        """
        Aplica la land mask a un único patch .tiff.

        Si el patch no intersecta con ninguna masa de tierra (patch
        completamente oceánico), copia los datos sin modificar.

        Args:
            input_path:  ruta al patch .tiff de entrada
            output_path: ruta donde guardar el patch enmascarado
        """
        input_path  = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(input_path) as src:
            # Reproyectar el shapefile al CRS del patch
            world_crs = self.world.to_crs(src.crs)

            try:
                # invert=True → enmascarar lo que está DENTRO del shapefile (tierra)
                # dejando visible lo que está FUERA (océano)
                img_masked, out_transform = mask(
                    src, world_crs.geometry, invert=True
                )
            except ValueError:
                # El patch no intersecta con ninguna masa de tierra
                # (patch completamente oceánico) — copiar sin modificar
                img_masked    = src.read()
                out_transform = src.transform

            out_meta = src.meta.copy()
            out_meta.update({
                "driver":    "GTiff",
                "height":    img_masked.shape[1],
                "width":     img_masked.shape[2],
                "transform": out_transform,
            })

            with rasterio.open(output_path, "w", **out_meta) as dst:
                dst.write(img_masked)

    # ------------------------------------------------------------------
    # Utilidad
    # ------------------------------------------------------------------

    def has_land(self, patch_path: str | Path) -> bool:
        """
        Comprueba si un patch contiene píxeles de tierra.

        Útil para filtrar patches completamente oceánicos antes
        de aplicar la máscara, ahorrando tiempo de procesamiento.

        Args:
            patch_path: ruta al patch .tiff

        Returns:
            True si el patch intersecta con alguna masa de tierra
        """
        with rasterio.open(patch_path) as src:
            world_crs = self.world.to_crs(src.crs)
            try:
                mask(src, world_crs.geometry, invert=True)
                return True
            except ValueError:
                return False


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT — uso manual desde terminal
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Uso: python land_mask_processor.py <input_dir> <output_dir>")
        print("Ejemplo: python land_mask_processor.py "
              "datasets/raw_data/sentinel_downloads/patches_2025-06-13 "
              "datasets/raw_data/sentinel_downloads/patches_2025-06-13_masked")
        sys.exit(1)

    processor = LandMaskProcessor()
    n = processor.apply_to_directory(
        input_dir  = sys.argv[1],
        output_dir = sys.argv[2],
    )
    print(f"\n{n} patches enmascarados correctamente.")