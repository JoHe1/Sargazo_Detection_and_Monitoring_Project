"""
train.py — Entry point de entrenamiento
-----------------------------------------
Entrena el modelo configurado en ExperimentConfig sobre el dataset MADOS.

Uso:
    # Con configuración por defecto (Swin Transformer, 50 épocas)
    python train.py

    # Con parámetros personalizados
    python train.py --model swin_transformer --epochs 30 --lr 1e-4 --batch 4

    # Desde un archivo YAML
    python train.py --config experiments/configs/swin_mados.yaml

Qué hace este archivo:
    - Carga la configuración (por defecto o desde YAML/args)
    - Instancia el modelo desde ModelRegistry
    - Instancia los datasets MADOSDataset (train y val)
    - Ejecuta el bucle de entrenamiento con early stopping
    - Guarda el mejor checkpoint en experiments/runs/{run_name}/

Lo que NO hace este archivo:
    - No define la arquitectura del modelo (→ models/architectures/)
    - No define la loss (→ models/losses/)
    - No define el dataset (→ datasets/sources/)
    - No define las métricas (→ core/utils/metrics.py)
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
#from models.losses.cross_entropy_dice import CrossEntropyDiceLoss
#from models.losses.focal_dice import FocalDiceLoss
from models.losses.cross_entropy_dice_tversky import CrossEntropyDiceTverskyLoss
from models.registry import ModelRegistry


def train(config: ExperimentConfig) -> None:
    """
    Bucle de entrenamiento completo.

    Args:
        config: configuración del experimento con todos los hiperparámetros
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.print_summary()
    print(f"[train] Dispositivo: {device.upper()}")

    # ── Datasets ──────────────────────────────────────────────────────
    print("\n[train] Cargando datasets...")
    train_dataset = MADOSDataset(
        root_path=SARGASSUM_READY,
        split="train",
        image_size=config.image_size,
        num_classes=config.num_classes,
    )
    val_dataset = MADOSDataset(
        root_path=SARGASSUM_READY,
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

    # ── Optimizer y scheduler ────────────────────────────────────────
    optimizer = model.configure_optimizers(config)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )

    # ── Loss ──────────────────────────────────────────────────────────
    #criterion = FocalDiceLoss(
    #    num_classes=config.num_classes,
    #    gamma=2.0,
    #    device=device,
    #).to(device)

    criterion = CrossEntropyDiceTverskyLoss(num_classes=config.num_classes, device=device).to(device)


    # ── Preparar directorio de checkpoint ────────────────────────────
    ckpt_dir  = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = ckpt_dir / "metrics.csv"

    # Cabecera del CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss",
            "iou_sargassum_denso", "iou_algas_sparse", "iou_sargassum_combinado", "mIoU", "lr"
        ])
    # ── Bucle de entrenamiento ────────────────────────────────────────
    mejor_val_loss     = float("inf")
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
            optimizer.step()
            train_loss += loss.item()

        # — Validación —
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

        t_loss = train_loss / len(train_loader)
        v_loss = val_loss   / len(val_loader)

        iou_medias  = np.where(iou_count > 0, iou_acum / iou_count, np.nan)
        mean_iou    = float(np.nanmean(iou_medias))
        iou_sarg    = float(iou_medias[2]) if not np.isnan(iou_medias[2]) else float("nan")
        iou_algas   = float(iou_medias[3]) if not np.isnan(iou_medias[3]) else float("nan")
        sarg_vals     = [v for v in [iou_sarg, iou_algas] if not np.isnan(v)]
        iou_sargassum = float(np.mean(sarg_vals)) if sarg_vals else float("nan")
        iou_comb_str  = f"{iou_sargassum:.4f}" if not np.isnan(iou_sargassum) else "  n/a  "
        lr_actual   = optimizer.param_groups[0]["lr"]

        scheduler.step(v_loss)
        lr_nuevo = optimizer.param_groups[0]["lr"]
        lr_str   = f"  [LR: {lr_actual:.2e}" + (
            f" → {lr_nuevo:.2e}]" if lr_nuevo != lr_actual else "]"
        )

        iou_s_str = f"{iou_sarg:.4f}"  if not np.isnan(iou_sarg)  else "  n/a  "
        iou_a_str = f"{iou_algas:.4f}" if not np.isnan(iou_algas) else "  n/a  "
        print(f"{epoch:>6}  {t_loss:>11.4f}  {v_loss:>10.4f}  "
            f"{iou_s_str:>10}  {iou_a_str:>10}  {iou_comb_str:>12}  {mean_iou:>8.4f}{lr_str}")


        # — Guardar métricas en CSV —
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                round(t_loss, 5), round(v_loss, 5),
                round(iou_sarg,       5) if not np.isnan(iou_sarg)       else "",
                round(iou_algas,      5) if not np.isnan(iou_algas)      else "",
                round(iou_sargassum,  5) if not np.isnan(iou_sargassum)  else "",
                round(mean_iou, 5),
                round(lr_actual, 8),
            ])

        # — Early stopping y guardado del mejor checkpoint —
        if v_loss < mejor_val_loss:
            mejor_val_loss     = v_loss
            epocas_sin_mejorar = 0
            model.save(
                checkpoint_dir=ckpt_dir,
                metadata={
                    **config.to_dict(),
                    "best_epoch":    epoch,
                    "best_val_loss": round(mejor_val_loss, 5),
                    "iou_sargassum": round(iou_sarg,  5) if not np.isnan(iou_sarg)  else None,
                    "iou_algas":     round(iou_algas, 5) if not np.isnan(iou_algas) else None,
                    "mIoU":          round(mean_iou, 5),
                }
            )
            print(f"  ✔ Mejora → checkpoint guardado (val_loss={mejor_val_loss:.4f})")
        else:
            epocas_sin_mejorar += 1
            print(f"  · Sin mejora ({epocas_sin_mejorar}/{config.patience})")
            if epocas_sin_mejorar >= config.patience:
                print(f"\n[train] Early stopping tras {epoch} épocas.")
                break

    print(f"\n[train] Entrenamiento finalizado.")
    print(f"  Mejor val_loss : {mejor_val_loss:.4f}")
    print(f"  Checkpoint     : {ckpt_dir}")
    print(f"  Métricas CSV   : {csv_path}")


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos de segmentación")
    parser.add_argument("--config",  type=str, default=None,
                        help="Ruta a un archivo YAML de configuración")
    parser.add_argument("--model",   type=str, default="swin_transformer",
                        help="Nombre del modelo (default: swin_transformer)")
    parser.add_argument("--dataset", type=str, default="mados",
                        help="Nombre del dataset (default: mados)")
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--lr",      type=float, default=5e-5)
    parser.add_argument("--batch",   type=int, default=8)
    parser.add_argument("--patience",type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    # check_paths()
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