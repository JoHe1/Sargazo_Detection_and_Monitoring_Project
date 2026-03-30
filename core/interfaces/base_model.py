"""
core/interfaces/base_model.py
------------------------------
Interfaz base para todos los modelos de segmentación del proyecto.

Uso:
    Crea una subclase e implementa los métodos abstractos:
        - forward(x)
        - configure_optimizers(config)

    Los métodos save(), load(), count_parameters() y get_info()
    están implementados aquí y se heredan sin cambios en todas las
    arquitecturas (Swin, UNet, SegFormer, etc.).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """
    Clase base para todos los modelos de segmentación.

    Hereda de nn.Module (PyTorch puro), por lo que es compatible tanto
    con modelos HuggingFace como con arquitecturas PyTorch clásicas.
    HuggingFace internamente también usa nn.Module, así que no hay conflicto.

    Métodos ABSTRACTOS (obligatorio implementar en cada subclase):
        forward()               — paso hacia adelante de la red
        configure_optimizers()  — devuelve el optimizador configurado

    Métodos IMPLEMENTADOS (heredados sin cambios):
        save()             — guarda pesos + metadata JSON en un checkpoint
        load()             — carga pesos desde un checkpoint
        count_parameters() — cuenta parámetros entrenables
        get_info()         — devuelve dict con info del modelo
    """

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # MÉTODOS ABSTRACTOS — cada subclase DEBE implementarlos
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante.

        Args:
            x: tensor de entrada (B, C, H, W)

        Returns:
            logits de salida (B, num_classes, H, W)
        """
        ...

    @abstractmethod
    def configure_optimizers(self, config: Any) -> torch.optim.Optimizer:
        """
        Construye y devuelve el optimizador para este modelo.

        Se define en cada subclase porque distintas arquitecturas pueden
        necesitar distintos grupos de parámetros o learning rates.

        Args:
            config: ExperimentConfig con lr, optimizer_name, weight_decay, etc.

        Returns:
            Optimizador de PyTorch listo para usar.
        """
        ...

    # ------------------------------------------------------------------
    # MÉTODOS IMPLEMENTADOS — se heredan tal cual en todas las subclases
    # ------------------------------------------------------------------

    def save(self, checkpoint_dir: str | Path, metadata: dict) -> Path:
        """
        Guarda el modelo en un directorio de checkpoint.

        Crea dos archivos:
            weights.pth   — state_dict del modelo
            metadata.json — config del experimento + fecha + métricas

        Args:
            checkpoint_dir: carpeta donde guardar (se crea si no existe)
            metadata: dict con información del experimento
                      Ejemplo: {"epochs": 50, "lr": 5e-5, "mIoU": 0.72, ...}

        Returns:
            Path al archivo weights.pth guardado
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        weights_path = checkpoint_dir / "weights.pth"
        metadata_path = checkpoint_dir / "metadata.json"

        # Guardar pesos
        torch.save(self.state_dict(), weights_path)

        # Enriquecer metadata con info automática
        full_metadata = {
            "model_class": self.__class__.__name__,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "num_parameters": self.count_parameters(),
            **metadata,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(full_metadata, f, indent=2, ensure_ascii=False)

        print(f"[{self.__class__.__name__}] Checkpoint guardado en: {checkpoint_dir}")
        return weights_path

    def load(self, checkpoint_dir: str | Path, device: str = "cpu") -> dict:
        """
        Carga los pesos desde un checkpoint y devuelve su metadata.

        Args:
            checkpoint_dir: carpeta del checkpoint (debe contener weights.pth)
            device: dispositivo donde cargar ("cpu" o "cuda")

        Returns:
            dict con la metadata del checkpoint (vacío si no existe metadata.json)
        """
        checkpoint_dir = Path(checkpoint_dir)
        weights_path = checkpoint_dir / "weights.pth"
        metadata_path = checkpoint_dir / "metadata.json"

        if not weights_path.exists():
            raise FileNotFoundError(f"No se encontraron pesos en: {weights_path}")

        state = torch.load(weights_path, map_location=device)
        self.load_state_dict(state)
        print(f"[{self.__class__.__name__}] Pesos cargados desde: {weights_path}")

        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        return metadata

    def count_parameters(self) -> int:
        """Devuelve el número de parámetros entrenables del modelo."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_info(self) -> dict:
        """
        Devuelve un dict con información básica del modelo.
        Útil para logging y para comparar modelos entre sí.
        """
        return {
            "model_class": self.__class__.__name__,
            "num_parameters": self.count_parameters(),
            "num_parameters_M": round(self.count_parameters() / 1e6, 2),
        }