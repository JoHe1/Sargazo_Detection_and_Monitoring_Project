"""
core/utils/metrics.py
-----------------------
Funciones de métricas de segmentación reutilizables en todo el proyecto.

Todas las funciones trabajan con tensores PyTorch y son agnósticas
al modelo: se pueden usar con Swin, UNet, SegFormer, o cualquier otro.

Funciones disponibles:
    iou_per_class()    — IoU para cada clase individualmente
    mean_iou()         — mIoU promedio (excluyendo clases sin presencia)
    pixel_accuracy()   — precisión a nivel de píxel
    compute_metrics()  — calcula todas las métricas de una vez (recomendado)
"""

from __future__ import annotations

import numpy as np
import torch


def iou_per_class(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 0,
) -> list[float]:
    """
    Calcula el IoU (Intersection over Union) para cada clase.

    Args:
        preds:        predicciones del modelo (B, H, W) con valores 0..num_classes-1
                      Si se pasan logits (B, C, H, W), se aplica argmax automáticamente.
        targets:      máscaras ground truth (B, H, W) con valores 0..num_classes-1
        num_classes:  número total de clases
        ignore_index: clase a ignorar en el cálculo (por defecto 0 = Non-annotated en MADOS)

    Returns:
        lista de floats de longitud num_classes.
        float("nan") si la clase no está presente en este batch (ni en pred ni en GT).
        El índice ignore_index siempre devuelve float("nan").
    """
    # Si vienen logits (B, C, H, W), convertir a predicciones de clase
    if preds.dim() == 4:
        preds = preds.argmax(dim=1)

    preds_flat   = preds.contiguous().view(-1)
    targets_flat = targets.contiguous().view(-1)

    iou_list = []
    for c in range(num_classes):
        if c == ignore_index:
            iou_list.append(float("nan"))
            continue

        pred_c   = preds_flat == c
        target_c = targets_flat == c

        intersection = (pred_c & target_c).sum().item()
        union        = (pred_c | target_c).sum().item()

        if union == 0:
            iou_list.append(float("nan"))  # clase no presente en este batch
        else:
            iou_list.append(intersection / union)

    return iou_list


def mean_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 0,
) -> float:
    """
    Calcula el mIoU promediando solo las clases presentes en el batch.

    Args:
        preds, targets, num_classes, ignore_index: igual que iou_per_class()

    Returns:
        float con el mIoU. Si no hay ninguna clase válida, devuelve 0.0.
    """
    ious = iou_per_class(preds, targets, num_classes, ignore_index)
    valid = [v for v in ious if not np.isnan(v)]
    return float(np.mean(valid)) if valid else 0.0


def pixel_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 0,
) -> float:
    """
    Calcula la precisión a nivel de píxel, ignorando ignore_index.

    Args:
        preds:        predicciones (B, H, W) o logits (B, C, H, W)
        targets:      ground truth (B, H, W)
        ignore_index: clase a ignorar

    Returns:
        float en [0, 1] con la proporción de píxeles correctamente clasificados
    """
    if preds.dim() == 4:
        preds = preds.argmax(dim=1)

    mask    = targets != ignore_index
    correct = (preds[mask] == targets[mask]).sum().item()
    total   = mask.sum().item()

    return correct / total if total > 0 else 0.0


def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 0,
    class_names: dict | None = None,
) -> dict:
    """
    Calcula todas las métricas de una vez.

    Función recomendada para usar en el bucle de validación porque
    evita hacer múltiples pasadas sobre los datos.

    Args:
        preds:        predicciones (B, H, W) o logits (B, C, H, W)
        targets:      ground truth (B, H, W)
        num_classes:  número total de clases
        ignore_index: clase a ignorar (0 = Non-annotated en MADOS)
        class_names:  dict opcional {class_id: "nombre"} para el resumen por clase
                      Ejemplo: {2: "Dense Sargassum", 3: "Sparse Floating Algae"}

    Returns:
        dict con:
            "mIoU"         — IoU medio sobre clases presentes
            "pixel_acc"    — precisión a nivel de píxel
            "iou_per_class"— dict {class_id: iou_value} para clases presentes
            "iou_sargassum"— IoU medio de las clases de sargazo (2 y 3) si presentes
    """
    ious = iou_per_class(preds, targets, num_classes, ignore_index)
    miou = mean_iou(preds, targets, num_classes, ignore_index)
    pacc = pixel_accuracy(preds, targets, ignore_index)

    # IoU por clase (solo las que tienen valor)
    iou_dict = {}
    for c, v in enumerate(ious):
        if not np.isnan(v):
            name = class_names[c] if class_names and c in class_names else str(c)
            iou_dict[name] = round(v, 4)

    # IoU específico de sargazo (clases 2 y 3 en MADOS)
    sarg_ious = [ious[c] for c in [2, 3] if c < len(ious) and not np.isnan(ious[c])]
    iou_sargassum = float(np.mean(sarg_ious)) if sarg_ious else float("nan")

    return {
        "mIoU":          round(miou, 4),
        "pixel_acc":     round(pacc, 4),
        "iou_per_class": iou_dict,
        "iou_sargassum": round(iou_sargassum, 4) if not np.isnan(iou_sargassum) else None,
    }