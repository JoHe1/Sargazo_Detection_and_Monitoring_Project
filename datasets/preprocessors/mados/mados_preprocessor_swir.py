"""
datasets/preprocessors/mados/mados_preprocessor_swir.py
---------------------------------------------------------
Versión extendida del preprocesador MADOS que añade las bandas
SWIR de Sentinel-2 (B11=1610nm, B12=2190nm) a las 4 bandas
existentes (B, G, R, NIR), generando tiles de 6 canales.

Las bandas SWIR están en la carpeta 20/ de cada escena a 20m
de resolución. Este preprocesador las lee y aplica un upsample
bilineal 2x para llevarlas a la resolución de 10m, de forma que
coincidan espacialmente con las bandas RGB+NIR.

IMPORTANTE: Este preprocesador SOLO sobreescribe los archivos
de images/. Las máscaras (masks/) NO se tocan, por lo que el
Ground Truth corregido manualmente permanece intacto.

Estructura esperada de MADOS en disco:
    datasets/data/MADOS/MADOS/
    └── Scene_XXX/
        ├── 10/
        │   ├── Scene_XXX_L2R_cl_YY.tif         ← máscara de clases
        │   ├── Scene_XXX_L2R_rhorc_492_YY.tif  ← banda azul (10m)
        │   ├── Scene_XXX_L2R_rhorc_560_YY.tif  ← banda verde (10m)
        │   ├── Scene_XXX_L2R_rhorc_665_YY.tif  ← banda roja (10m)
        │   └── Scene_XXX_L2R_rhorc_833_YY.tif  ← NIR (10m)
        └── 20/
            ├── Scene_XXX_L2R_rhorc_1610_YY.tif ← SWIR1 B11 (20m)
            └── Scene_XXX_L2R_rhorc_2190_YY.tif ← SWIR2 B12 (20m)

Estructura generada en disco:
    datasets/data/Sargassum_Ready_Dataset_SWIR/
    ├── train/
    │   ├── images/  ← arrays .npy (H, W, 6): (B, G, R, NIR, SWIR1, SWIR2)
    │   └── masks/   ← SYMLINK o copia de las máscaras originales
    ├── val/  ...
    └── test/ ...

Uso:
    python datasets/preprocessors/mados/mados_preprocessor_swir.py

    O apuntando a un output diferente:
        preprocessor = MADOSPreprocessorSWIR(output_dir="datasets/data/Sargassum_Ready_Dataset_SWIR")
        preprocessor.run()
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile as tiff
from tqdm import tqdm

from core.config.paths import DATA_ROOT, MADOS_RAW_DIR, SARGASSUM_READY


class MADOSPreprocessorSWIR:
    """
    Convierte el dataset MADOS crudo (TIFs de ACOLITE) a arrays .npy.

    Decisiones de diseño importantes:
      - Los tiles se guardan en su tamaño ORIGINAL (≥224px).
        El recorte a 224×224 lo hace MADOSDataset en __getitem__,
        lo que permite random crop en train y center crop en val/test
        sin necesidad de preprocesar dos veces.

      - Las bandas se guardan en orden ACOLITE (B, G, R, NIR).
        MADOSDataset las reordena a (R, G, B, NIR) en tiempo de carga
        para que coincidan con los pesos pre-entrenados de ImageNet.

      - La banda verde puede ser 559 nm o 560 nm según la versión
        de ACOLITE con la que se procesó cada escena. El preprocessor
        detecta automáticamente cuál existe.

      - Los valores de máscara se clamean a [0, NUM_CLASSES-1] para
        evitar crashes en la CrossEntropyLoss con valores NoData (255).
    """

    NUM_CLASSES   = 16
    NUM_BANDS     = 6   # B, G, R, NIR, SWIR1, SWIR2
    MIN_TILE_SIZE = 224  # tiles más pequeños se descartan

    def __init__(
        self,
        mados_root: str | Path = MADOS_RAW_DIR,
        output_dir: str | Path = None,
    ) -> None:
        """
        Args:
            mados_root: ruta a la carpeta raíz de MADOS que contiene las escenas
                        y la carpeta splits/. Por defecto usa MADOS_RAW_DIR de paths.py
            output_dir: carpeta de salida con estructura train/val/test/images|masks.
                        Por defecto usa SARGASSUM_READY de paths.py
        """
        self.mados_root = Path(mados_root)
        # Si no se especifica output_dir, usamos un dataset nuevo con sufijo _SWIR
        if output_dir is None:
            self.output_dir = Path(str(SARGASSUM_READY) + "_SWIR")
        else:
            self.output_dir = Path(output_dir)

        if not self.mados_root.exists():
            raise FileNotFoundError(
                f"No se encuentra la carpeta de MADOS en: {self.mados_root}\n"
                f"Revisa la ruta MADOS_RAW_DIR en core/config/paths.py"
            )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Ejecuta el preprocesamiento completo para los tres splits.

        Returns:
            dict con el resumen global de tiles procesados y saltados
        """
        split_files = {
            "train": self.mados_root / "splits" / "train_X.txt",
            "val":   self.mados_root / "splits" / "val_X.txt",
            "test":  self.mados_root / "splits" / "test_X.txt",
        }

        print("=" * 50)
        print("  MADOS Preprocessor — 16 clases")
        print(f"  Origen : {self.mados_root}")
        print(f"  Destino: {self.output_dir}")
        print("=" * 50)

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

    # ------------------------------------------------------------------
    # Internos — estructura de directorios
    # ------------------------------------------------------------------

    def _setup_directories(self) -> None:
        """Crea la estructura train/val/test + images/masks si no existe."""
        for split in ("train", "val", "test"):
            for subdir in ("images", "masks"):
                (self.output_dir / split / subdir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internos — procesamiento por split
    # ------------------------------------------------------------------

    def _process_split(self, split_name: str, split_path: Path) -> dict:
        """
        Procesa todos los recortes de un split.

        Returns:
            dict con conteo de resultados por categoría
        """
        conteo = {"ok": 0, "skip_parse": 0, "skip_missing": 0,
                  "skip_size": 0, "error": 0}

        with open(split_path, "r") as f:
            crop_lines = [line.strip() for line in f if line.strip()]

        for crop_line in tqdm(crop_lines, desc=f"  {split_name}"):
            resultado = self._process_crop(crop_line, split_name)
            conteo[resultado] += 1

        return conteo

    # ------------------------------------------------------------------
    # Internos — procesamiento por tile
    # ------------------------------------------------------------------

    def _parse_crop_line(self, crop_line: str) -> tuple[str | None, str | None]:
        """
        Extrae (carpeta_escena, id_recorte) del nombre de un recorte.

        Convención de nombres en MADOS:
            'Scene_127_48'  →  carpeta='Scene_127',  id='48'

        El ID es siempre el último segmento separado por '_'.
        La carpeta de escena son todos los segmentos anteriores.

        Returns:
            (carpeta_escena, id_recorte) o (None, None) si el nombre es inválido
        """
        partes = crop_line.split("_")
        if len(partes) < 3:
            return None, None

        id_recorte     = partes[-1]
        carpeta_escena = "_".join(partes[:-1])
        return carpeta_escena, id_recorte

    def _resolve_green_band(
        self, path_10m: Path, carpeta_escena: str, id_recorte: str
    ) -> Path:
        """
        Resuelve la ruta de la banda verde (559 nm o 560 nm).

        Algunas escenas de MADOS fueron procesadas con una versión de ACOLITE
        que nombra la banda como 560 nm y otras como 559 nm.
        Intentamos el 560 primero (más común) y caemos al 559 si no existe.

        Returns:
            Path a la banda verde que existe (o al 560 si ninguna existe,
            para que falle con skip_missing de forma controlada)
        """
        band_560 = path_10m / f"{carpeta_escena}_L2R_rhorc_560_{id_recorte}.tif"
        band_559 = path_10m / f"{carpeta_escena}_L2R_rhorc_559_{id_recorte}.tif"
        return band_560 if band_560.exists() else band_559

    def _process_crop(self, crop_line: str, split_name: str) -> str:
        """
        Procesa un único recorte de MADOS: lee los TIFs, apila las bandas
        y guarda los arrays .npy en la carpeta de salida.

        Returns:
            "ok"           → tile guardado correctamente
            "skip_parse"   → nombre de recorte inválido (< 3 segmentos)
            "skip_missing" → algún archivo TIF requerido no existe en disco
            "skip_size"    → tile más pequeño que MIN_TILE_SIZE (descartado)
            "error"        → excepción durante lectura o escritura
        """
        carpeta_escena, id_recorte = self._parse_crop_line(crop_line)
        if carpeta_escena is None:
            return "skip_parse"

        path_10m = self.mados_root / carpeta_escena / "10"

        # Rutas a los archivos requeridos
        mask_file  = path_10m / f"{carpeta_escena}_L2R_cl_{id_recorte}.tif"
        band_492   = path_10m / f"{carpeta_escena}_L2R_rhorc_492_{id_recorte}.tif"
        band_verde = self._resolve_green_band(path_10m, carpeta_escena, id_recorte)
        band_665   = path_10m / f"{carpeta_escena}_L2R_rhorc_665_{id_recorte}.tif"
        band_833   = path_10m / f"{carpeta_escena}_L2R_rhorc_833_{id_recorte}.tif"

        archivos = [mask_file, band_492, band_verde, band_665, band_833]
        if not all(f.exists() for f in archivos):
            return "skip_missing"

        try:
            # Leer las cuatro bandas espectrales a 10m
            b_azul  = tiff.imread(band_492)    # Azul  (492 nm)
            b_verde = tiff.imread(band_verde)  # Verde (559 ó 560 nm)
            b_rojo  = tiff.imread(band_665)    # Rojo  (665 nm)
            b_nir   = tiff.imread(band_833)    # NIR   (833 nm)

            # Leer bandas SWIR desde carpeta 20/ y hacer upsample 2x → 10m
            path_20m = self.mados_root / carpeta_escena / "20"
            swir1_path = path_20m / f"{carpeta_escena}_L2R_rhorc_1610_{id_recorte}.tif"
            swir2_path = path_20m / f"{carpeta_escena}_L2R_rhorc_2190_{id_recorte}.tif"

            if swir1_path.exists() and swir2_path.exists():
                b_swir1 = tiff.imread(swir1_path).astype(np.float32)
                b_swir2 = tiff.imread(swir2_path).astype(np.float32)
                # Upsample 2x bilineal para llevar de 20m a 10m
                b_swir1 = self._upsample_2x(b_swir1, b_azul.shape)
                b_swir2 = self._upsample_2x(b_swir2, b_azul.shape)
                tiene_swir = True
            else:
                # Si no hay SWIR, rellenar con ceros para mantener consistencia
                b_swir1 = np.zeros_like(b_azul, dtype=np.float32)
                b_swir2 = np.zeros_like(b_azul, dtype=np.float32)
                tiene_swir = False

            # Apilar en orden: (B, G, R, NIR, SWIR1, SWIR2)
            img = np.stack(
                (b_azul, b_verde, b_rojo, b_nir, b_swir1, b_swir2), axis=-1
            ).astype(np.float32)  # → (H, W, 6)

            # Leer máscara y sanear valores NoData
            mask = tiff.imread(mask_file).astype(np.uint8)
            mask = np.clip(mask, 0, self.NUM_CLASSES - 1)

            # Descartar tiles demasiado pequeños para el crop
            h, w = mask.shape
            if h < self.MIN_TILE_SIZE or w < self.MIN_TILE_SIZE:
                return "skip_size"

            # Guardar tile completo — el crop se hace en MADOSDataset.__getitem__
            out_img  = self.output_dir / split_name / "images" / f"{crop_line}.npy"
            out_mask = self.output_dir / split_name / "masks"  / f"{crop_line}.npy"
            np.save(out_img,  img)
            np.save(out_mask, mask)
            return "ok"

        except Exception as e:
            print(f"  [ERROR] {crop_line}: {e}")
            return "error"

    # ------------------------------------------------------------------
    # Internos — upsample SWIR
    # ------------------------------------------------------------------

    @staticmethod
    def _upsample_2x(band: np.ndarray, target_shape: tuple) -> np.ndarray:
        """
        Upsample bilineal de una banda de 20m a 10m.

        Usa interpolación bilineal mediante numpy para evitar
        dependencias adicionales. El target_shape es el shape
        de las bandas de 10m con las que debe coincidir.

        Args:
            band:         array (H20, W20) a 20m
            target_shape: shape (H10, W10) de destino a 10m

        Returns:
            array (H10, W10) interpolado
        """
        from scipy.ndimage import zoom
        h_target, w_target = target_shape
        h_src,    w_src    = band.shape
        zoom_h = h_target / h_src
        zoom_w = w_target / w_src
        return zoom(band, (zoom_h, zoom_w), order=1).astype(np.float32)

    # ------------------------------------------------------------------
    # Internos — logging
    # ------------------------------------------------------------------

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
        print("\n" + "═" * 50)
        print("  RESUMEN GLOBAL")
        print(f"  Tiles guardados     : {resumen['ok']}")
        print(f"  Archivos faltantes  : {resumen['skip_missing']}")
        print(f"  Tiles demasiado peq.: {resumen['skip_size']}")
        print(f"  Nombres inválidos   : {resumen['skip_parse']}")
        print(f"  Errores de lectura  : {resumen['error']}")
        print("═" * 50)
        print("¡Preprocesamiento completado!")


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    preprocessor = MADOSPreprocessorSWIR()
    preprocessor.run()