"""
datasets/preprocessors/mados/mados_preprocessor_11bands.py
------------------------------------------------------------
Preprocesador MADOS con las 11 bandas completas de Sentinel-2.

Bandas incluidas (orden final en el array .npy):
    Canal 0  — B1  443nm  (60m → upsample 6x a 10m)  Coastal/Aerosol
    Canal 1  — B2  492nm  (10m)                        Azul
    Canal 2  — B3  560nm  (10m)                        Verde
    Canal 3  — B4  665nm  (10m)                        Rojo
    Canal 4  — B5  704nm  (20m → upsample 2x a 10m)   Red-Edge 1
    Canal 5  — B6  740nm  (20m → upsample 2x a 10m)   Red-Edge 2
    Canal 6  — B7  783nm  (20m → upsample 2x a 10m)   Red-Edge 3
    Canal 7  — B8  833nm  (10m)                        NIR
    Canal 8  — B8A 865nm  (20m → upsample 2x a 10m)   NIR narrow
    Canal 9  — B11 1610nm (20m → upsample 2x a 10m)   SWIR1
    Canal 10 — B12 2190nm (20m → upsample 2x a 10m)   SWIR2

Bandas faltantes en una escena concreta se rellenan con ceros —
el tile NO se descarta por bandas ausentes, para no perder tiles
de sargazo que son escasos.

Las máscaras (masks/) NO se modifican — el GT refinado manualmente
permanece intacto.

Estructura de entrada esperada:
    datasets/data/MADOS/MADOS/
    └── Scene_XXX/
        ├── 10/   ← B2(492), B3(560/559), B4(665), B8(833)
        ├── 20/   ← B5(704), B6(740), B7(783), B8A(865), B11(1610), B12(2190)
        └── 60/   ← B1(443)

Estructura de salida:
    datasets/data/Sargassum_Ready_Dataset_11bands/
    ├── train/images/  ← .npy (H, W, 11) float32
    ├── train/masks/   ← .npy (H, W)     uint8
    ├── val/...
    └── test/...

Uso:
    python datasets/preprocessors/mados/mados_preprocessor_11bands.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile as tiff
from scipy.ndimage import zoom
from tqdm import tqdm

from core.config.paths import MADOS_RAW_DIR, SARGASSUM_READY

SARGASSUM_READY_11BANDS = Path(str(SARGASSUM_READY) + "_11bands")


class MADOSPreprocessor11Bands:
    """
    Convierte MADOS crudo a arrays .npy de 11 canales Sentinel-2.

    Decisiones de diseño:
      - Bandas a 20m: upsample 2x bilineal → 10m.
      - Bandas a 60m: upsample 6x bilineal → 10m.
      - Si una banda no existe en una escena → canal relleno con ceros.
        El tile NO se descarta — no podemos perder tiles de sargazo.
      - Orden canales: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
        (ascendente por longitud de onda, igual que el paper MADOS).
      - MADOSDataset11Bands reordena en carga para compatibilidad ImageNet.
    """

    NUM_CLASSES   = 16
    NUM_BANDS     = 11
    MIN_TILE_SIZE = 224

    # (nombre, nm, carpeta_raw, zoom_factor_a_10m)
    BAND_DEFS = [
        ("b1_443",   "443",  "60", 6),
        ("b2_492",   "492",  "10", 1),
        ("b3_560",   "560",  "10", 1),   # fallback a 559
        ("b4_665",   "665",  "10", 1),
        ("b5_704",   "704",  "20", 2),
        ("b6_740",   "740",  "20", 2),
        ("b7_783",   "783",  "20", 2),
        ("b8_833",   "833",  "10", 1),
        ("b8a_865",  "865",  "20", 2),
        ("b11_1610", "1610", "20", 2),
        ("b12_2190", "2190", "20", 2),
    ]

    def __init__(
        self,
        mados_root: str | Path = MADOS_RAW_DIR,
        output_dir: str | Path = SARGASSUM_READY_11BANDS,
    ) -> None:
        self.mados_root = Path(mados_root)
        self.output_dir = Path(output_dir)
        if not self.mados_root.exists():
            raise FileNotFoundError(
                f"No se encuentra MADOS en: {self.mados_root}\n"
                f"Revisa MADOS_RAW_DIR en core/config/paths.py"
            )

    def run(self) -> dict:
        split_files = {
            "train": self.mados_root / "splits" / "train_X.txt",
            "val":   self.mados_root / "splits" / "val_X.txt",
            "test":  self.mados_root / "splits" / "test_X.txt",
        }
        print("=" * 60)
        print("  MADOS Preprocessor — 11 bandas Sentinel-2")
        print(f"  Origen : {self.mados_root}")
        print(f"  Destino: {self.output_dir}")
        print("=" * 60)
        self._setup_directories()
        resumen = {"ok": 0, "skip_parse": 0, "skip_missing": 0,
                   "skip_size": 0, "error": 0}
        for split_name, split_path in split_files.items():
            if not split_path.exists():
                print(f"\n  AVISO: {split_path.name} no encontrado. Saltando {split_name}.")
                continue
            print(f"\nProcesando split: {split_name.upper()}")
            conteo = self._process_split(split_name, split_path)
            for k, v in conteo.items():
                resumen[k] += v
            self._print_split_summary(conteo)
        self._print_global_summary(resumen)
        return resumen

    def _setup_directories(self) -> None:
        for split in ("train", "val", "test"):
            for subdir in ("images", "masks"):
                (self.output_dir / split / subdir).mkdir(parents=True, exist_ok=True)

    def _process_split(self, split_name: str, split_path: Path) -> dict:
        conteo = {"ok": 0, "skip_parse": 0, "skip_missing": 0,
                  "skip_size": 0, "error": 0}
        with open(split_path, "r") as f:
            crop_lines = [line.strip() for line in f if line.strip()]
        for crop_line in tqdm(crop_lines, desc=f"  {split_name}"):
            conteo[self._process_crop(crop_line, split_name)] += 1
        return conteo

    def _parse_crop_line(self, crop_line: str):
        partes = crop_line.split("_")
        if len(partes) < 3:
            return None, None
        return "_".join(partes[:-1]), partes[-1]

    def _resolve_band_path(
        self, scene_root: Path, carpeta_escena: str,
        id_recorte: str, nm: str, carpeta: str
    ) -> Path | None:
        base = scene_root / carpeta
        path = base / f"{carpeta_escena}_L2R_rhorc_{nm}_{id_recorte}.tif"
        if path.exists():
            return path
        # Fallback banda verde 559/560
        if nm == "560":
            alt = base / f"{carpeta_escena}_L2R_rhorc_559_{id_recorte}.tif"
            if alt.exists():
                return alt
        return None  # banda ausente → ceros

    def _read_band(
        self, path: Path | None, ref_shape: tuple, zoom_factor: int
    ) -> np.ndarray:
        """Lee una banda y la lleva a ref_shape (10m). Ceros si ausente."""
        if path is None:
            return np.zeros(ref_shape, dtype=np.float32)
        try:
            band = tiff.imread(str(path)).astype(np.float32)
            if zoom_factor == 1:
                if band.shape != ref_shape:
                    band = self._match_shape(band, ref_shape)
                return band
            upsampled = zoom(band, zoom_factor, order=1).astype(np.float32)
            if upsampled.shape != ref_shape:
                upsampled = self._match_shape(upsampled, ref_shape)
            return upsampled
        except Exception:
            return np.zeros(ref_shape, dtype=np.float32)

    @staticmethod
    def _match_shape(band: np.ndarray, target: tuple) -> np.ndarray:
        """Recorta o padea para que coincida exactamente con target (H, W)."""
        h, w = target
        band = band[:h, :w]
        ph = h - band.shape[0]
        pw = w - band.shape[1]
        if ph > 0 or pw > 0:
            band = np.pad(band, ((0, ph), (0, pw)), mode='edge')
        return band

    def _process_crop(self, crop_line: str, split_name: str) -> str:
        carpeta_escena, id_recorte = self._parse_crop_line(crop_line)
        if carpeta_escena is None:
            return "skip_parse"

        scene_root = self.mados_root / carpeta_escena
        path_10m   = scene_root / "10"

        # Máscara y B2 son obligatorias
        mask_file = path_10m / f"{carpeta_escena}_L2R_cl_{id_recorte}.tif"
        b2_file   = path_10m / f"{carpeta_escena}_L2R_rhorc_492_{id_recorte}.tif"
        if not mask_file.exists() or not b2_file.exists():
            return "skip_missing"

        try:
            b2_ref    = tiff.imread(str(b2_file)).astype(np.float32)
            h, w      = b2_ref.shape
            ref_shape = (h, w)

            if h < self.MIN_TILE_SIZE or w < self.MIN_TILE_SIZE:
                return "skip_size"

            bands = []
            for _, nm, carpeta, zf in self.BAND_DEFS:
                if nm == "492":
                    bands.append(b2_ref)  # ya leída
                else:
                    bpath = self._resolve_band_path(
                        scene_root, carpeta_escena, id_recorte, nm, carpeta
                    )
                    bands.append(self._read_band(bpath, ref_shape, zf))

            img  = np.stack(bands, axis=-1).astype(np.float32)  # (H, W, 11)
            mask = tiff.imread(str(mask_file)).astype(np.uint8)
            mask = np.clip(mask, 0, self.NUM_CLASSES - 1)

            out_img  = self.output_dir / split_name / "images" / f"{crop_line}.npy"
            out_mask = self.output_dir / split_name / "masks"  / f"{crop_line}.npy"
            np.save(str(out_img),  img)
            np.save(str(out_mask), mask)
            return "ok"

        except Exception as e:
            print(f"  [ERROR] {crop_line}: {e}")
            return "error"

    @staticmethod
    def _print_split_summary(conteo: dict) -> None:
        print(f"  ✔ Guardados      : {conteo['ok']}")
        if conteo["skip_missing"] > 0:
            print(f"  ✗ Sin archivos   : {conteo['skip_missing']}")
        if conteo["skip_size"]    > 0:
            print(f"  ✗ Tile pequeño   : {conteo['skip_size']}")
        if conteo["skip_parse"]   > 0:
            print(f"  ✗ Nombre inválido: {conteo['skip_parse']}")
        if conteo["error"]        > 0:
            print(f"  ✗ Errores I/O    : {conteo['error']}")

    @staticmethod
    def _print_global_summary(resumen: dict) -> None:
        print("\n" + "═" * 60)
        print("  RESUMEN GLOBAL — 11 bandas")
        print(f"  Tiles guardados     : {resumen['ok']}")
        print(f"  Archivos faltantes  : {resumen['skip_missing']}")
        print(f"  Tiles demasiado peq.: {resumen['skip_size']}")
        print(f"  Nombres inválidos   : {resumen['skip_parse']}")
        print(f"  Errores de lectura  : {resumen['error']}")
        print("═" * 60)
        print("¡Preprocesamiento 11 bandas completado!")


if __name__ == "__main__":
    preprocessor = MADOSPreprocessor11Bands()
    preprocessor.run()
