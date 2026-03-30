"""
evaluate.py — Evaluación rápida de un checkpoint
---------------------------------------------------
Carga un modelo entrenado y evalúa su rendimiento sobre
imágenes del dataset MADOS, mostrando imagen + GT + predicción.

Es el antiguo test_rapido.py, refactorizado para usar la arquitectura.

Uso:
    # Evaluar sobre val con configuración por defecto
    python evaluate.py

    # Especificar checkpoint y split
    python evaluate.py --modelo experiments/runs/mi_experimento --split test

    # Evaluar N imágenes aleatorias
    python evaluate.py --n 5 --split val

    # Solo imágenes con sargazo
    python evaluate.py --solo-sargazo
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from core.config.paths import SARGASSUM_READY, RESULTS_DIR
from core.utils.metrics import compute_metrics
from core.utils.visualization import MADOS_CLASSES, show_prediction
from models.architectures.swin_transformer import SwinSegmenter

DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES      = 16
CLASES_SARGASSUM = {2, 3}


def evaluar_muestra(
    model:       torch.nn.Module,
    img_path:    Path,
    mask_path:   Path | None,
    umbral:      float = 0.30,
    save_dir:    Path | None = None,
) -> dict:
    """
    Evalúa el modelo sobre una única imagen y muestra los resultados.

    Args:
        model:     modelo cargado en eval()
        img_path:  ruta al .npy de imagen
        mask_path: ruta al .npy de máscara (None si no existe)
        umbral:    umbral de probabilidad para binarizar sargazo
        save_dir:  carpeta donde guardar la figura (None = no guardar)

    Returns:
        dict con métricas de la imagen (vacío si no hay máscara)
    """
    # Cargar y preprocesar imagen
    img  = np.load(img_path).astype(np.float32)
    img  = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    img  = img[:, :, [2, 1, 0, 3]]          # (B,G,R,NIR) → (R,G,B,NIR)
    if img.max() > 10.0:
        img = img / 10000.0
    img = np.clip(img * 5.0, 0, 1)

    # Center crop
    TARGET = 224
    h, w   = img.shape[:2]
    y0 = (h - TARGET) // 2
    x0 = (w - TARGET) // 2
    img = img[y0:y0 + TARGET, x0:x0 + TARGET, :]

    # Cargar máscara si existe
    mask_real = None
    if mask_path and mask_path.exists():
        mask_raw  = np.load(mask_path).astype(np.int64)
        mask_real = mask_raw[y0:y0 + TARGET, x0:x0 + TARGET]

    # Tensor para el modelo
    tensor = torch.tensor(
        np.transpose(img, (2, 0, 1)), dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    # Inferencia
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits).squeeze().cpu().numpy()
        pred   = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.int32)

    # Calcular métricas si hay GT
    metricas = {}
    if mask_real is not None:
        mask_tensor = torch.tensor(mask_real, dtype=torch.long).unsqueeze(0)
        pred_tensor = torch.tensor(pred,      dtype=torch.long).unsqueeze(0)
        metricas    = compute_metrics(pred_tensor, mask_tensor, NUM_CLASSES)

        iou_s = metricas.get("iou_sargassum")
        miou  = metricas.get("mIoU", float("nan"))
        print(f"  mIoU: {miou:.4f}  |  IoU sargazo: "
              f"{iou_s:.4f if iou_s else 'n/a'}  |  "
              f"Px sargazo pred: {int((pred == 2).sum() + (pred == 3).sum())}")

    # Visualizar
    show_prediction(
        image=img,
        ground_truth=mask_real if mask_real is not None
                     else np.zeros((TARGET, TARGET), dtype=np.int32),
        prediction=pred,
        class_map=MADOS_CLASSES,
        highlight_classes=CLASES_SARGASSUM,
        title=f"{img_path.name}  —  umbral={umbral:.0%}",
        save_path=(save_dir / f"eval_{img_path.stem}.png") if save_dir else None,
    )

    return metricas


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación rápida de un checkpoint")
    parser.add_argument("--modelo",        type=str, default=None,
                        help="Carpeta del checkpoint (contiene weights.pth)")
    parser.add_argument("--split",         type=str, default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--n",             type=int, default=4,
                        help="Número de imágenes a evaluar (default: 4)")
    parser.add_argument("--umbral",        type=float, default=0.30)
    parser.add_argument("--solo-sargazo",  action="store_true",
                        help="Solo evaluar imágenes que contienen sargazo")
    parser.add_argument("--guardar",       action="store_true",
                        help="Guardar figuras en experiments/results/")
    args = parser.parse_args()

    # ── Cargar modelo ─────────────────────────────────────────────────
    if args.modelo is None:
        runs = sorted(Path("experiments/runs").glob("*/weights.pth"))
        if not runs:
            print("[ERROR] No hay checkpoints en experiments/runs/")
            return
        weights_path = runs[-1]
        print(f"[evaluate] Checkpoint más reciente: {weights_path.parent.name}")
    else:
        weights_path = Path(args.modelo) / "weights.pth"

    model = SwinSegmenter(num_classes=NUM_CLASSES).to(DEVICE)
    model.load(checkpoint_dir=weights_path.parent, device=DEVICE)
    model.eval()
    print(f"[evaluate] Modelo cargado | {DEVICE.upper()}")

    # ── Seleccionar imágenes ──────────────────────────────────────────
    img_dir  = SARGASSUM_READY / args.split / "images"
    mask_dir = SARGASSUM_READY / args.split / "masks"

    if not img_dir.exists():
        print(f"[ERROR] No existe: {img_dir}")
        print("        ¿Has ejecutado MADOSPreprocessor?")
        return

    img_paths = sorted(img_dir.glob("*.npy"))

    if args.solo_sargazo:
        img_paths = [
            ip for ip in img_paths
            if np.isin(np.load(mask_dir / ip.name), list(CLASES_SARGASSUM)).any()
        ]
        print(f"[evaluate] Imágenes con sargazo: {len(img_paths)}")

    if not img_paths:
        print("[AVISO] No se encontraron imágenes.")
        return

    seleccion = random.sample(img_paths, min(args.n, len(img_paths)))
    save_dir  = RESULTS_DIR if args.guardar else None

    print(f"[evaluate] Evaluando {len(seleccion)} imágenes del split {args.split}...\n")

    miou_global = []
    for img_path in seleccion:
        mask_path = mask_dir / img_path.name
        print(f"  {img_path.name}")
        m = evaluar_muestra(model, img_path, mask_path,
                            umbral=args.umbral, save_dir=save_dir)
        if "mIoU" in m:
            miou_global.append(m["mIoU"])

    if miou_global:
        print(f"\n[evaluate] mIoU medio sobre {len(miou_global)} imágenes: "
              f"{np.mean(miou_global):.4f}")


if __name__ == "__main__":
    main()