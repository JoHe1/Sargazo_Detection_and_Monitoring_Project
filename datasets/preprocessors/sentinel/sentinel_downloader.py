"""
datasets/preprocessors/sentinel/sentinel_downloader.py
--------------------------------------------------------
Descarga una región de Sentinel-2 de la API de Copernicus
y la trocea en patches .tiff listos para inferencia.

Se llama desde la app web o manualmente para obtener imágenes nuevas.
NO se usa durante el entrenamiento (eso es MADOS).

Flujo:
    SentinelDownloader.download_and_tile()
        → descarga mega_region.tiff de Copernicus API
        → trocea en patches_{fecha}/patch_001.tiff, patch_002.tiff...
        → devuelve lista de rutas a los patches

    Después, opcionalmente:
    LandMaskProcessor.apply_to_directory()
        → enmascara la tierra en cada patch

Credenciales:
    Las credenciales de Copernicus NO van en el código.
    Defínelas en un archivo .env en la raíz del proyecto:

        COPERNICUS_CLIENT_ID=sh-fee1f5d8-...
        COPERNICUS_CLIENT_SECRET=8Gfz02k...

    Y cárgalas antes de instanciar SentinelDownloader:

        from dotenv import load_dotenv
        load_dotenv()

Uso:
    downloader = SentinelDownloader(
        output_dir="datasets/raw_data/sentinel_downloads"
    )
    patches = downloader.download_and_tile(
        bbox=[-68.13, 19.13, -68.01, 19.25],
        date_from="2025-06-13",
        date_to="2025-06-13",
        patch_size=224,
    )
"""

from __future__ import annotations

import os
from pathlib import Path

import rasterio
import requests
from rasterio.windows import Window

from core.config.paths import SENTINEL_DIR


# ══════════════════════════════════════════════════════════════════════
# EVALSCRIPT — 4 bandas (R, G, B, NIR)
# ══════════════════════════════════════════════════════════════════════
# Devuelve exactamente las mismas 4 bandas con las que está entrenado
# el modelo (MADOS también tiene R, G, B, NIR en ese orden).
# SCL filtra píxeles malos: nubes, sombras, nieve, sin datos.

EVALSCRIPT_4B = """
function setup() {
    return {
        input:  ["B02", "B03", "B04", "B08", "SCL", "dataMask"],
        output: { bands: 4, sampleType: "FLOAT32" }
    };
}
function evaluatePixel(sample) {
    let badPixels = [0, 3, 8, 9, 10];
    if (badPixels.includes(sample.SCL) || sample.dataMask === 0) {
        return [0, 0, 0, 0];
    }
    // Orden: R, G, B, NIR — igual que lo que espera el modelo
    return [
        sample.B04 * 2.5,
        sample.B03 * 2.5,
        sample.B02 * 2.5,
        sample.B08 * 2.5
    ];
}
"""


class SentinelDownloader:
    """
    Descarga una región de Sentinel-2 y la trocea en patches .tiff.

    Cada patch preserva los metadatos geoespaciales del TIFF original
    (CRS, transform) para poder reconstruir la posición en el mapa
    tras la inferencia.
    """

    TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    API_URL   = "https://sh.dataspace.copernicus.eu/api/v1/process"

    def __init__(
        self,
        output_dir: str | Path = SENTINEL_DIR,
    ) -> None:
        """
        Args:
            output_dir: carpeta donde guardar los patches descargados.
                        Por defecto usa SENTINEL_DIR de core/config/paths.py
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Credenciales desde variables de entorno — nunca hardcodeadas
        self.client_id     = os.environ.get("COPERNICUS_CLIENT_ID", "")
        self.client_secret = os.environ.get("COPERNICUS_CLIENT_SECRET", "")

        if not self.client_id or not self.client_secret:
            raise EnvironmentError(
                "Credenciales de Copernicus no encontradas.\n"
                "Añade en tu .env:\n"
                "  COPERNICUS_CLIENT_ID=sh-...\n"
                "  COPERNICUS_CLIENT_SECRET=..."
            )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def download_and_tile(
        self,
        bbox: list[float],
        date_from: str,
        date_to: str,
        patch_size: int = 224,
        image_size: int = 1280,
        max_cloud_coverage: int = 40,
    ) -> list[Path]:
        """
        Descarga una región de Sentinel-2 y la trocea en patches.

        Args:
            bbox:               [lon_min, lat_min, lon_max, lat_max]
                                Ejemplo: [-68.13, 19.13, -68.01, 19.25]
            date_from:          fecha inicio en formato "YYYY-MM-DD"
            date_to:            fecha fin    en formato "YYYY-MM-DD"
            patch_size:         tamaño de cada patch en píxeles.
                                Usar 224 para que coincida con el modelo.
            image_size:         resolución de la imagen completa a descargar.
                                1280 genera 25 patches de 224px aprox.
            max_cloud_coverage: porcentaje máximo de nubosidad aceptado (0-100)

        Returns:
            lista de Paths a los patches .tiff generados, ordenados
        """
        print(f"[SentinelDownloader] Obteniendo token de Copernicus...")
        token = self._get_token()

        mega_tiff = self.output_dir / f"mega_{date_from}.tiff"
        print(
            f"[SentinelDownloader] Descargando región bbox={bbox} "
            f"({image_size}x{image_size}px, nubosidad≤{max_cloud_coverage}%)..."
        )
        self._download_region(
            token, bbox, date_from, date_to,
            image_size, max_cloud_coverage, mega_tiff
        )

        patches_dir = self.output_dir / f"patches_{date_from}"
        patches_dir.mkdir(exist_ok=True)

        print(f"[SentinelDownloader] Trocando en patches de {patch_size}x{patch_size}px...")
        patches = self._tile_image(mega_tiff, patches_dir, patch_size)
        print(f"[SentinelDownloader] {len(patches)} patches generados en: {patches_dir}")

        return patches

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _get_token(self) -> str:
        """Obtiene el token OAuth2 de la API de Copernicus."""
        response = requests.post(
            self.TOKEN_URL,
            data={
                "grant_type":    "client_credentials",
                "client_id":     self.client_id,
                "client_secret": self.client_secret,
            },
        )
        response.raise_for_status()
        return response.json()["access_token"]

    def _download_region(
        self,
        token: str,
        bbox: list[float],
        date_from: str,
        date_to: str,
        image_size: int,
        max_cloud: int,
        output_path: Path,
    ) -> None:
        """
        Hace la petición a Sentinel Hub Process API y guarda el TIFF.

        Lanza RuntimeError si la API devuelve un error.
        """
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
            "Accept":        "image/tiff",
        }
        payload = {
            "input": {
                "bounds": {
                    "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
                    "bbox": bbox,
                },
                "data": [{
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{date_from}T00:00:00Z",
                            "to":   f"{date_to}T23:59:59Z",
                        },
                        "maxCloudCoverage": max_cloud,
                    },
                }],
            },
            "evalscript": EVALSCRIPT_4B,
            "output": {
                "width":  image_size,
                "height": image_size,
                "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
            },
        }

        response = requests.post(self.API_URL, headers=headers, json=payload)

        if response.status_code != 200:
            raise RuntimeError(
                f"Error al descargar de Copernicus ({response.status_code}):\n"
                f"{response.text}\n"
                f"Revisa bbox, fechas y nubosidad."
            )

        output_path.write_bytes(response.content)
        print(f"[SentinelDownloader] Imagen descargada: {output_path.name}")

    def _tile_image(
        self,
        tiff_path: Path,
        output_dir: Path,
        patch_size: int,
    ) -> list[Path]:
        """
        Trocea el TIFF grande en patches más pequeños.

        Preserva los metadatos geoespaciales (CRS, transform) en cada patch
        para poder reconstruir su posición en el mapa tras la inferencia.

        Returns:
            lista de Paths a los patches generados, en orden fila-columna
        """
        patches = []

        with rasterio.open(tiff_path) as src:
            contador = 1

            for j in range(0, src.height, patch_size):
                for i in range(0, src.width, patch_size):

                    window    = Window(i, j, patch_size, patch_size)
                    transform = src.window_transform(window)
                    recorte   = src.read(window=window)

                    meta = src.meta.copy()
                    meta.update({
                        "height":    patch_size,
                        "width":     patch_size,
                        "transform": transform,
                    })

                    patch_path = output_dir / f"patch_{contador:03d}.tiff"
                    with rasterio.open(patch_path, "w", **meta) as dst:
                        dst.write(recorte)

                    patches.append(patch_path)
                    contador += 1

        return patches


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT — uso manual desde terminal
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    downloader = SentinelDownloader()
    patches = downloader.download_and_tile(
        bbox=[-68.13, 19.13, -68.01, 19.25],
        date_from="2025-06-13",
        date_to="2025-06-13",
        patch_size=224,
        image_size=1280,
        max_cloud_coverage=40,
    )
    print(f"\nPatches generados: {len(patches)}")
    for p in patches:
        print(f"  {p}")