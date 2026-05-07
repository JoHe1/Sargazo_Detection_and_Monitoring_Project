"""
train.py — Entry point de entrenamiento
-----------------------------------------
Cambios respecto a v2:
    - EMA (Exponential Moving Average) añadido con alpha=0.999.
      Mantiene una copia promediada de los pesos durante entrenamiento.
      El checkpoint guarda los pesos EMA, no los pesos normales.
      Esto produce modelos más estables y robustos (usado en MariNeXt).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from core.config.experiment_config import ExperimentConfig
from core.config.paths import check_paths, SARGASSUM_READY
from core.utils.metrics import compute_metrics, iou_per_class
from datasets.sources.mados_dataset import MADOSDataset
from datasets.sources.mados_dataset_swir import MADOSDatasetSWIR, SARGASSUM_READY_SWIR
from models.losses.cross_entropy_dice import CrossEntropyDiceLoss
from models.losses.focal_dice import FocalDiceLoss
from models.losses.cross_entropy_dice_tversky import CrossEntropyDiceTverskyLoss
from models.registry import ModelRegistry


# ══════════════════════════════════════════════════════════════════════
# EMA — Exponential Moving Average de pesos
# ══════════════════════════════════════════════════════════════════════

class EMA:
    """
    Mantiene una copia con media exponencial de los parámetros del modelo.

    Uso en MariNeXt (Kikaki et al., 2024): alpha=0.999
    θ_ema ← alpha * θ_ema + (1 - alpha) * θ

    El modelo EMA se usa para validación y para guardar el checkpoint.
    El modelo normal se sigue usando para el forward/backward de train.
    """

    def __init__(self, model: torch.nn.Module, alpha: float = 0.999) -> None:
        self.alpha  = alpha
        # Copia profunda de los pesos iniciales
        self.shadow = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def update(self, model: torch.nn.Module) -> None:
        """Actualiza los pesos shadow tras cada paso de optimización."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.alpha * self.shadow[name]
                    + (1.0 - self.alpha) * param.data
                )

    def apply(self, model: torch.nn.Module) -> None:
        """Copia los pesos EMA al modelo (para validación/inferencia)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module, backup: dict) -> None:
        """Restaura los pesos originales al modelo tras validación."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in backup:
                param.data.copy_(backup[name])

    def backup_params(self, model: torch.nn.Module) -> dict:
        """Hace backup de los pesos actuales antes de aplicar EMA."""
        return {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }


def train(config: ExperimentConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.print_summary()
    print(f"[train] Dispositivo: {device.upper()}")

    # ── Datasets ──────────────────────────────────────────────────────
    print("\n[train] Cargando datasets...")
    usa_swir = "swir" in config.model_name.lower()
    DatasetClass = MADOSDatasetSWIR if usa_swir else MADOSDataset
    dataset_root = SARGASSUM_READY_SWIR if usa_swir else SARGASSUM_READY
    if usa_swir:
        print("[train] Modo SWIR: usando MADOSDatasetSWIR (6 canales)")

    train_dataset = DatasetClass(
        root_path=dataset_root,
        split="train",
        image_size=config.image_size,
        num_classes=config.num_classes,
    )
    val_dataset = DatasetClass(
        root_path=dataset_root,
        split="val",
        image_size=config.image_size,
        num_classes=config.num_classes,
    )

    if len(train_dataset) == 0:
        print("[ERROR] No se encontraron datos de entrenamiento.")
        print("        ¿Has ejecutado MADOSPreprocessor?")
        return

    train_loader = train_dataset.get_loader(
        batch_size=config.batch_size, num_workers=config.num_workers
    )
    val_loader = val_dataset.get_loader(
        batch_size=config.batch_size, num_workers=config.num_workers
    )

    # ── Modelo ────────────────────────────────────────────────────────
    print(f"\n[train] Construyendo modelo: {config.model_name}")
    model = ModelRegistry.build(
        config.model_name,
        num_classes=config.num_classes,
    ).to(device)
    info = model.get_info()
    print(f"[train] Parámetros entrenables: {info['num_parameters_M']}M")

    # ── EMA ───────────────────────────────────────────────────────────
    ema = EMA(model, alpha=0.99)
    print("[train] EMA activado (alpha=0.99)")

    # ── Optimizer y scheduler ────────────────────────────────────────
    optimizer = model.configure_optimizers(config)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=12
    )

    # ── Loss ──────────────────────────────────────────────────────────
    # criterion = CrossEntropyDiceLoss(num_classes=config.num_classes, device=device).to(device)
    criterion = FocalDiceLoss(num_classes=config.num_classes, gamma=2.0, device=device).to(device)
    # criterion = CrossEntropyDiceTverskyLoss(num_classes=config.num_classes, device=device).to(device)

    loss_name = criterion.__class__.__name__

    # ── Preparar directorio de checkpoint ────────────────────────────
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ckpt_dir / "metrics.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss",
            "iou_sargassum_denso", "iou_algas_sparse",
            "iou_sargassum_combinado", "mIoU", "lr"
        ])

    # ── Bucle de entrenamiento ────────────────────────────────────────
    mejor_iou_sarg     = -1.0   # criterio: IoU sargazo combinado (clases 2+3)
    epocas_sin_mejorar = 0

    print(f"\n[train] Iniciando entrenamiento "
          f"({config.num_classes} clases | {config.image_size}×{config.image_size})")
    print(f"{'Época':>6}  {'Train Loss':>11}  {'Val Loss':>10}  "
          f"{'IoU Sarg.':>10}  {'IoU Algas':>10}  {'mIoU':>8}")
    print("─" * 65)

    for epoch in range(1, config.epochs + 1):

        # — Entrenamiento —
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks  = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # Actualizar EMA tras cada paso
            ema.update(model)
            train_loss += loss.item()

        # — Validación con pesos EMA —
        backup = ema.backup_params(model)  # guardar pesos actuales
        ema.apply(model)                   # aplicar pesos EMA al modelo
        model.eval()

        val_loss  = 0.0
        iou_acum  = np.zeros(config.num_classes)
        iou_count = np.zeros(config.num_classes)

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks  = masks.to(device)
                outputs = model(images)
                loss    = criterion(outputs, masks)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                ious  = iou_per_class(preds, masks, config.num_classes)

                for c, iou in enumerate(ious):
                    if not np.isnan(iou):
                        iou_acum[c]  += iou
                        iou_count[c] += 1

        # Restaurar pesos normales para continuar entrenando
        ema.restore(model, backup)

        t_loss = train_loss / len(train_loader)
        v_loss = val_loss   / len(val_loader)

        iou_medias    = np.where(iou_count > 0, iou_acum / iou_count, np.nan)
        mean_iou      = float(np.nanmean(iou_medias))
        iou_sarg      = float(iou_medias[2]) if not np.isnan(iou_medias[2]) else float("nan")
        iou_algas     = float(iou_medias[3]) if not np.isnan(iou_medias[3]) else float("nan")
        sarg_vals     = [v for v in [iou_sarg, iou_algas] if not np.isnan(v)]
        iou_sargassum = float(np.mean(sarg_vals)) if sarg_vals else float("nan")
        iou_comb_str  = f"{iou_sargassum:.4f}" if not np.isnan(iou_sargassum) else "  n/a  "
        lr_actual     = optimizer.param_groups[0]["lr"]

        scheduler.step(v_loss)
        lr_nuevo = optimizer.param_groups[0]["lr"]
        lr_str   = f"  [LR: {lr_actual:.2e}" + (
            f" → {lr_nuevo:.2e}]" if lr_nuevo != lr_actual else "]"
        )

        iou_s_str = f"{iou_sarg:.4f}"  if not np.isnan(iou_sarg)  else "  n/a  "
        iou_a_str = f"{iou_algas:.4f}" if not np.isnan(iou_algas) else "  n/a  "
        print(f"{epoch:>6}  {t_loss:>11.4f}  {v_loss:>10.4f}  "
              f"{iou_s_str:>10}  {iou_a_str:>10}  {iou_comb_str:>12}  {mean_iou:>8.4f}{lr_str}")

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                round(t_loss, 5), round(v_loss, 5),
                round(iou_sarg,       5) if not np.isnan(iou_sarg)      else "",
                round(iou_algas,      5) if not np.isnan(iou_algas)     else "",
                round(iou_sargassum,  5) if not np.isnan(iou_sargassum) else "",
                round(mean_iou, 5),
                round(lr_actual, 8),
            ])

        # — Early stopping por IoU sargazo combinado —
        # Si iou_sargassum es nan (no hay sargazo en val), usamos 0.0
        iou_sarg_criterio = iou_sargassum if not np.isnan(iou_sargassum) else 0.0

        if iou_sarg_criterio > mejor_iou_sarg:
            mejor_iou_sarg     = iou_sarg_criterio
            epocas_sin_mejorar = 0
            # Aplicar EMA antes de guardar
            backup_save = ema.backup_params(model)
            ema.apply(model)
            model.save(
                checkpoint_dir=ckpt_dir,
                metadata={
                    **config.to_dict(),
                    "loss_function": loss_name,
                    "best_epoch":    epoch,
                    "best_val_loss": round(v_loss, 5),
                    "iou_sargassum": round(iou_sarg,  5) if not np.isnan(iou_sarg)  else None,
                    "iou_algas":     round(iou_algas, 5) if not np.isnan(iou_algas) else None,
                    "iou_sargassum_combinado": round(iou_sarg_criterio, 5),
                    "mIoU":          round(mean_iou, 5),
                    "ema_alpha":     0.99,
                }
            )
            ema.restore(model, backup_save)
            print(f"  ✔ Mejora IoU sargazo={mejor_iou_sarg:.4f} → checkpoint EMA guardado")
        else:
            epocas_sin_mejorar += 1
            print(f"  · Sin mejora ({epocas_sin_mejorar}/{config.patience})")
            if epocas_sin_mejorar >= config.patience:
                print(f"\n[train] Early stopping tras {epoch} épocas.")
                break

    print(f"\n[train] Entrenamiento finalizado.")
    print(f"  Mejor IoU sargazo : {mejor_iou_sarg:.4f}")
    print(f"  Checkpoint        : {ckpt_dir}")
    print(f"  Métricas CSV      : {csv_path}")


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos de segmentación")
    parser.add_argument("--config",   type=str,   default=None)
    parser.add_argument("--model",    type=str,   default="swin_transformer")
    parser.add_argument("--dataset",  type=str,   default="mados")
    parser.add_argument("--epochs",   type=int,   default=50)
    parser.add_argument("--lr",       type=float, default=5e-5)
    parser.add_argument("--batch",    type=int,   default=8)
    parser.add_argument("--patience", type=int,   default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig(
            model_name   = args.model,
            dataset_name = args.dataset,
            epochs       = args.epochs,
            lr           = args.lr,
            batch_size   = args.batch,
            patience     = args.patience,
        )

    train(config)