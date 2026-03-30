"""
core/config/experiment_config.py
----------------------------------
Configuración central de un experimento.

Un ExperimentConfig viaja a través de todo el sistema:
    train.py → Trainer → CheckpointManager → metadata.json

Soporta dos formas de uso:

    1. Desde Python (más cómodo para pruebas rápidas):
        config = ExperimentConfig(
            model_name="swin_transformer",
            dataset_name="mados",
            epochs=50,
            lr=5e-5,
        )

    2. Desde YAML (más cómodo para lanzar experimentos reproducibles):
        config = ExperimentConfig.from_yaml("experiments/configs/swin_mados.yaml")

    Para guardar la config actual como YAML:
        config.to_yaml("experiments/configs/mi_experimento.yaml")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# PyYAML es una dependencia muy ligera; si no está instalada, solo
# se desactiva from_yaml() / to_yaml() y el resto funciona igual.
try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


@dataclass
class ExperimentConfig:
    """
    Configuración completa de un experimento de entrenamiento.

    Todos los campos tienen valores por defecto razonables para poder
    hacer pruebas rápidas sin rellenar todo. En producción conviene
    ser explícito con todos los parámetros importantes.

    Campos principales:
        model_name    — nombre de la arquitectura (debe coincidir con ModelRegistry)
        dataset_name  — nombre del dataset (debe coincidir con DatasetRegistry)
        run_name      — nombre único para este experimento (se usa como carpeta)

    Hiperparámetros de entrenamiento:
        epochs, batch_size, lr, weight_decay, patience, optimizer_name

    Configuración del modelo y datos:
        num_classes, input_channels, image_size, num_workers

    Rutas (se generan automáticamente si no se especifican):
        checkpoint_dir — donde guardar pesos y metadata
        dataset_dir    — ruta raíz del dataset
    """

    # — Identificación del experimento —
    model_name:   str = "swin_transformer"
    dataset_name: str = "mados"
    run_name:     str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    # — Hiperparámetros de entrenamiento —
    epochs:         int   = 50
    batch_size:     int   = 8
    lr:             float = 5e-5
    weight_decay:   float = 1e-4
    patience:       int   = 8          # épocas sin mejora antes de early stopping
    optimizer_name: str   = "adamw"    # "adamw", "adam", "sgd"
    scheduler_name: str   = "plateau"  # "plateau", "cosine", "none"

    # — Configuración del modelo —
    num_classes:    int = 16
    input_channels: int = 4            # RGB + NIR para Sentinel-2
    image_size:     int = 224

    # — Configuración de datos —
    num_workers:    int = 2
    dataset_dir:    str = "datasets/data/Sargassum_Ready_Dataset"

    # — Rutas de salida —
    # Si se deja vacío, se genera automáticamente como:
    # experiments/runs/{run_name}_{model_name}_{dataset_name}/
    checkpoint_dir: str = ""

    # — Flags opcionales —
    use_amp:        bool = False       # Automatic Mixed Precision (requiere GPU)
    seed:           int  = 42

    def __post_init__(self) -> None:
        """Se ejecuta al crear el objeto. Genera checkpoint_dir si está vacío."""
        if not self.checkpoint_dir:
            self.checkpoint_dir = str(
                Path("experiments") / "runs" /
                f"{self.run_name}_{self.model_name}_{self.dataset_name}"
            )

    # ------------------------------------------------------------------
    # SERIALIZACIÓN
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convierte la config a un diccionario plano."""
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        """Guarda la config como JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[ExperimentConfig] Guardada en: {path}")

    def to_yaml(self, path: str | Path) -> None:
        """Guarda la config como YAML (requiere PyYAML)."""
        if not _YAML_AVAILABLE:
            raise ImportError("PyYAML no está instalado. Ejecuta: pip install pyyaml")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
        print(f"[ExperimentConfig] Guardada en: {path}")

    # ------------------------------------------------------------------
    # CARGA
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """Crea una ExperimentConfig desde un diccionario."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        """Carga una config desde un archivo JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Carga una config desde un archivo YAML (requiere PyYAML)."""
        if not _YAML_AVAILABLE:
            raise ImportError("PyYAML no está instalado. Ejecuta: pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # UTILIDADES
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Imprime un resumen legible de la configuración."""
        print("\n" + "═" * 50)
        print("  EXPERIMENT CONFIG")
        print("═" * 50)
        for key, value in self.to_dict().items():
            print(f"  {key:<20}: {value}")
        print("═" * 50 + "\n")