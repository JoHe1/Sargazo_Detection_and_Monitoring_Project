"""
core/interfaces/base_trainer.py
---------------------------------
Interfaz base para el entrenamiento de modelos.

Uso:
    Crea una subclase e implementa los métodos abstractos:
        - train_epoch()
        - validate_epoch()

    El bucle completo train(), el early stopping, el guardado
    del mejor modelo y el logging al CSV están implementados aquí.
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


class BaseTrainer(ABC):
    """
    Clase base para todos los trainers del proyecto.

    Gestiona el bucle de entrenamiento completo:
        - Iteración por épocas
        - Early stopping
        - Guardado del mejor modelo (por val_loss)
        - Logging de métricas a CSV

    Métodos ABSTRACTOS (obligatorio implementar en cada subclase):
        train_epoch()    — una época de entrenamiento, devuelve métricas
        validate_epoch() — una época de validación, devuelve métricas

    Métodos IMPLEMENTADOS (heredados sin cambios):
        train()     — bucle completo con early stopping y logging
        _log_csv()  — escribe una fila de métricas en el CSV
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            model:        modelo a entrenar (subclase de BaseModel)
            optimizer:    optimizador ya configurado
            criterion:    función de pérdida
            train_loader: DataLoader del split train
            val_loader:   DataLoader del split val
            config:       ExperimentConfig con epochs, patience, checkpoint_dir, etc.
            device:       "cpu" o "cuda"
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.model.to(self.device)

    # ------------------------------------------------------------------
    # MÉTODOS ABSTRACTOS
    # ------------------------------------------------------------------

    @abstractmethod
    def train_epoch(self) -> dict:
        """
        Ejecuta una época completa de entrenamiento.

        Returns:
            dict con métricas del epoch. Debe incluir al menos:
            {"train_loss": float}
            Puede incluir métricas adicionales: {"train_loss": ..., "train_iou": ...}
        """
        ...

    @abstractmethod
    def validate_epoch(self) -> dict:
        """
        Ejecuta una época completa de validación.

        Returns:
            dict con métricas del epoch. Debe incluir al menos:
            {"val_loss": float}
            Puede incluir métricas adicionales: {"val_loss": ..., "val_iou": ..., "mIoU": ...}
        """
        ...

    # ------------------------------------------------------------------
    # MÉTODOS IMPLEMENTADOS
    # ------------------------------------------------------------------

    def train(self) -> dict:
        """
        Bucle de entrenamiento completo.

        Itera por épocas llamando a train_epoch() y validate_epoch(),
        aplica early stopping si val_loss no mejora en config.patience épocas,
        guarda el mejor modelo automáticamente y loguea métricas a CSV.

        Returns:
            dict con el resumen del entrenamiento:
            {"best_val_loss": float, "best_epoch": int, "total_epochs": int}
        """
        epochs    = self.config.epochs
        patience  = self.config.patience
        ckpt_dir  = Path(self.config.checkpoint_dir)
        csv_path  = ckpt_dir / "metrics.csv"

        ckpt_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss      = float("inf")
        epochs_no_improve  = 0
        best_epoch         = 0
        csv_header_written = False

        print(f"\n[Trainer] Iniciando entrenamiento: {epochs} épocas máx. | "
              f"patience={patience} | device={self.device}")
        print(f"[Trainer] Checkpoints en: {ckpt_dir}\n")

        for epoch in range(1, epochs + 1):

            # — Entrenamiento —
            self.model.train()
            train_metrics = self.train_epoch()

            # — Validación —
            self.model.eval()
            with torch.no_grad():
                val_metrics = self.validate_epoch()

            # — Combinar métricas —
            all_metrics = {"epoch": epoch, **train_metrics, **val_metrics}

            # — Escribir cabecera CSV en la primera época —
            if not csv_header_written:
                self._log_csv(csv_path, all_metrics, write_header=True)
                csv_header_written = True
            else:
                self._log_csv(csv_path, all_metrics, write_header=False)

            # — Imprimir resumen de época —
            metrics_str = "  ".join(
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in all_metrics.items()
            )
            print(f"[Epoch {epoch:>3}/{epochs}]  {metrics_str}")

            # — Early stopping y guardado del mejor modelo —
            val_loss = val_metrics.get("val_loss", float("inf"))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch    = epoch
                epochs_no_improve = 0

                # Guardar checkpoint del mejor modelo
                metadata = {
                    **vars(self.config),
                    "best_epoch": epoch,
                    "best_val_loss": round(best_val_loss, 5),
                    **{k: round(v, 5) if isinstance(v, float) else v
                       for k, v in val_metrics.items()},
                }
                self.model.save(ckpt_dir, metadata)
                print(f"  ✔ Mejor modelo guardado (val_loss={best_val_loss:.4f})")
            else:
                epochs_no_improve += 1
                print(f"  · Sin mejora ({epochs_no_improve}/{patience})")
                if epochs_no_improve >= patience:
                    print(f"\n[Trainer] Early stopping tras {epoch} épocas.")
                    break

        print(f"\n[Trainer] Entrenamiento finalizado.")
        print(f"  Mejor val_loss : {best_val_loss:.4f} (época {best_epoch})")
        print(f"  Métricas CSV   : {csv_path}")

        return {
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "total_epochs": epoch,
        }

    def _log_csv(self, csv_path: Path, metrics: dict, write_header: bool) -> None:
        """
        Escribe una fila de métricas en el CSV de entrenamiento.

        Args:
            csv_path:     ruta al archivo CSV
            metrics:      dict con los valores a escribir
            write_header: si True, escribe la cabecera antes de la fila
        """
        mode = "w" if write_header else "a"
        with open(csv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)