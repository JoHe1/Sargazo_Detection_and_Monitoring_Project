"""
core/interfaces/base_preprocessor.py
--------------------------------------
Interfaz base para los preprocesadores de imagen del proyecto.

Los preprocesadores son transformaciones que se aplican a imágenes
ANTES de pasarlas al dataset o al modelo. Ejemplos:
    - Corrección atmosférica de imágenes Sentinel-2
    - Aplicación de land mask (enmascarar tierra)
    - Generación de tiles a partir de una imagen grande

Uso:
    Crea una subclase e implementa apply().
    El método validate() tiene una implementación por defecto
    que puedes sobreescribir si necesitas validaciones específicas.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BasePreprocessor(ABC):
    """
    Clase base para todos los preprocesadores del proyecto.

    Es la interfaz más sencilla del proyecto: una transformación
    que entra con datos y sale con datos procesados.

    Métodos ABSTRACTOS (obligatorio implementar):
        apply() — aplica la transformación

    Métodos IMPLEMENTADOS (se pueden sobreescribir si hace falta):
        validate() — comprueba que los datos de entrada son válidos
    """

    # ------------------------------------------------------------------
    # MÉTODOS ABSTRACTOS
    # ------------------------------------------------------------------

    @abstractmethod
    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Aplica el preprocesamiento a una imagen.

        Args:
            image:   array de entrada (H, W, C) o (H, W)
            **kwargs: parámetros adicionales específicos del preprocesador

        Returns:
            array procesado con la misma forma que la entrada
            salvo que el preprocesador cambie dimensiones (ej: tiling)
        """
        ...

    # ------------------------------------------------------------------
    # MÉTODOS IMPLEMENTADOS (sobreescribibles)
    # ------------------------------------------------------------------

    def validate(self, image: np.ndarray) -> bool:
        """
        Comprueba que la imagen de entrada tiene un formato válido.

        Implementación por defecto: verifica que no esté vacía y que
        sea un array numpy. Sobreescribe este método en subclases que
        necesiten validaciones más específicas (ej: número de canales).

        Args:
            image: array a validar

        Returns:
            True si es válido, False si no lo es
        """
        if not isinstance(image, np.ndarray):
            return False
        if image.size == 0:
            return False
        return True

    def apply_to_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        pattern: str = "*.tiff",
    ) -> int:
        """
        Aplica el preprocesamiento a todos los archivos de un directorio.

        Implementación por defecto usando apply() en cada archivo.
        Sobreescribe si necesitas lógica de batch más compleja.

        Args:
            input_dir:  carpeta con los archivos de entrada
            output_dir: carpeta donde guardar los resultados
            pattern:    patrón glob para filtrar archivos

        Returns:
            número de archivos procesados correctamente
        """
        import rasterio

        input_dir  = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(input_dir.glob(pattern))
        processed = 0

        for file_path in files:
            try:
                with rasterio.open(file_path) as src:
                    image   = src.read()                  # (C, H, W)
                    image   = np.transpose(image, (1, 2, 0))  # → (H, W, C)
                    meta    = src.meta.copy()

                result = self.apply(image)
                result_bands = np.transpose(result, (2, 0, 1))  # → (C, H, W)

                out_path = output_dir / file_path.name
                meta.update({
                    "height": result.shape[0],
                    "width":  result.shape[1],
                })
                with rasterio.open(out_path, "w", **meta) as dst:
                    dst.write(result_bands)

                processed += 1
            except Exception as e:
                print(f"[{self.__class__.__name__}] Error procesando {file_path.name}: {e}")

        print(f"[{self.__class__.__name__}] Procesados {processed}/{len(files)} archivos.")
        return processed