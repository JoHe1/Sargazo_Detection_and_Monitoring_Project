"""
inference.py — Entry point de inferencia
------------------------------------------
Ejecuta inferencia sobre imágenes del dataset MADOS con el modelo entrenado.

Toda la lógica de TTA, FAI, postprocesado y visualización se mantiene igual.
La única diferencia respecto al original es que SwinSegmenter se importa
desde models/architectures/ en lugar de estar definida aquí.

Uso:
    python inference.py                          # val, sargazo, umbral 0.35, TTA activado
    python inference.py --split test             # evaluación final sobre test
    python inference.py --n 5                    # 5 imágenes aleatorias
    python inference.py --umbral 0.25            # umbral más permisivo
    python inference.py --sin-tta                # desactivar TTA (más rápido)
    python inference.py --sigma 0                # desactivar suavizado gaussiano
    python inference.py --min-pixels 20          # componentes mínimos para limpieza
    python inference.py --modelo path/a/ckpt/    # checkpoint personalizado
"""

from __future__ import annotations

import argparse
import glob
import random
from pathlib import Path
from xml.parsers.expat import model

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import binary_fill_holes, gaussian_filter, label

from core.config.paths import SARGASSUM_READY, CHECKPOINTS_DIR, RESULTS_DIR
from core.utils.visualization import MADOS_CLASSES
from models.architectures.swin_transformer import SwinSegmenter

# ══════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES  = 16
CLASES       = MADOS_CLASSES   # importado desde visualization.py — una sola definición

CLASES_MOSTRAR   = {1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15}
CLASES_SARGASSUM = {2, 3}


# ══════════════════════════════════════════════════════════════════════
# PREPROCESAMIENTO
# ══════════════════════════════════════════════════════════════════════

def preprocesar(img_path: str) -> tuple[np.ndarray, torch.Tensor]:
    """
    Carga y preprocesa una imagen .npy para inferencia.
    Aplica exactamente el mismo pipeline que MADOSDataset en val/test.
    """
    img = np.load(img_path).astype(np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    img = img[:, :, [2, 1, 0, 3]]           # (B,G,R,NIR) → (R,G,B,NIR)
    if img.max() > 10.0:
        img = img / 10000.0
    img = np.clip(img * 5.0, 0.0, 1.0)

    TARGET = 224
    h, w   = img.shape[:2]
    y0 = (h - TARGET) // 2
    x0 = (w - TARGET) // 2
    img = img[y0:y0 + TARGET, x0:x0 + TARGET, :]

    tensor = torch.tensor(
        np.transpose(img, (2, 0, 1)), dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    return img, tensor   # img normalizada (H,W,4), tensor (1,4,H,W)


# ══════════════════════════════════════════════════════════════════════
# INFERENCIA
# ══════════════════════════════════════════════════════════════════════

def inferir_single(model: nn.Module, tensor: torch.Tensor) -> np.ndarray:
    """Inferencia simple — devuelve (NUM_CLASSES, H, W) de probabilidades."""
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze(0)
    return probs.cpu().numpy()


def inferir(
    model: nn.Module,
    tensor: torch.Tensor,
    usar_tta: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inferencia con TTA opcional (Test-Time Augmentation).

    TTA promedia las probabilidades sobre 4 variantes:
        original, flip-H, flip-V, flip-H+V.
    Reduce el efecto granular en bordes de patch y mejora la coherencia espacial.

    Returns:
        clase_predicha : (H, W) int
        prob_sargassum : (H, W) float  — P(cl.2) + P(cl.3)
        prob_todas     : (NUM_CLASSES, H, W) float
    """
    if not usar_tta:
        probs_np = inferir_single(model, tensor)
    else:
        acum  = np.zeros((NUM_CLASSES, 224, 224), dtype=np.float32)
        flips = [
            (False, False),
            (True,  False),
            (False, True),
            (True,  True),
        ]
        for flip_h, flip_v in flips:
            t = tensor.clone()
            if flip_h: t = torch.flip(t, dims=[3])
            if flip_v: t = torch.flip(t, dims=[2])
            probs = inferir_single(model, t)
            if flip_h: probs = probs[:, :, ::-1].copy()
            if flip_v: probs = probs[:, ::-1, :].copy()
            acum += probs
        probs_np = acum / len(flips)

    clase_predicha = probs_np.argmax(axis=0).astype(np.int32)
    prob_sargassum = probs_np[2] + probs_np[3]
    return clase_predicha, prob_sargassum, probs_np


def postprocesar(
    prob_sarg:  np.ndarray,
    umbral:     float = 0.35,
    sigma:      float = 2.0,
    min_pixels: int   = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pipeline de post-procesado para reducir ruido en el mapa de sargazo.

    1. Suavizado gaussiano — elimina ruido granular de alta frecuencia
    2. Umbral — binariza el mapa suavizado
    3. Eliminar componentes pequeños — elimina FP aislados
    4. Rellenar huecos internos — da continuidad a las manchas
    """
    prob_suave = gaussian_filter(prob_sarg.astype(np.float32), sigma=sigma) \
                 if sigma > 0 else prob_sarg.copy()

    binario = (prob_suave >= umbral).astype(np.uint8)

    if min_pixels > 0:
        etiquetas, n_comp = label(binario)
        for i in range(1, n_comp + 1):
            if (etiquetas == i).sum() < min_pixels:
                binario[etiquetas == i] = 0

    mascara_limpia = binary_fill_holes(binario).astype(np.uint8)
    return prob_suave, mascara_limpia


def calcular_fai_mask(
    img_npy_path: str,
    umbral_fai:   float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Máscara FAI (Floating Algae Index) = NIR - Rojo.

    Filtro físico: el sargazo siempre tiene FAI positivo.
    Agua limpia y sedimento tienen FAI negativo o cercano a 0.

    Returns:
        mascara_fai: (224, 224) bool
        mapa_fai:    (224, 224) float
    """
    img_raw = np.load(img_npy_path).astype(np.float32)
    if img_raw.max() > 10.0:
        img_raw = img_raw / 10000.0

    TARGET = 224
    h, w   = img_raw.shape[:2]
    y0 = (h - TARGET) // 2
    x0 = (w - TARGET) // 2
    img_crop = img_raw[y0:y0 + TARGET, x0:x0 + TARGET, :]

    nir  = img_crop[:, :, 3]   # índice 3 en orden (B,G,R,NIR)
    rojo = img_crop[:, :, 2]   # índice 2 en orden (B,G,R,NIR)
    fai  = nir - rojo

    return (fai >= umbral_fai), fai


# ══════════════════════════════════════════════════════════════════════
# MÉTRICAS POR IMAGEN
# ══════════════════════════════════════════════════════════════════════

def calcular_iou(pred: np.ndarray, gt: np.ndarray, clase: int) -> float:
    inter = ((pred == clase) & (gt == clase)).sum()
    union = ((pred == clase) | (gt == clase)).sum()
    return inter / union if union > 0 else float("nan")


def metricas_imagen(clase_pred: np.ndarray, mascara_gt: np.ndarray) -> dict:
    """IoU por clase para las clases presentes en GT (excluye clase 0)."""
    return {
        c: calcular_iou(clase_pred, mascara_gt, c)
        for c in set(np.unique(mascara_gt).tolist()) - {0}
    }


# ══════════════════════════════════════════════════════════════════════
# VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════

def hex_to_rgba(hex_color: str, alpha: float = 0.65) -> list:
    h = hex_color.lstrip("#")
    r, g, b = [int(h[i:i+2], 16) / 255 for i in (0, 2, 4)]
    return [r, g, b, alpha]


def crear_overlay(
    mask:            np.ndarray,
    shape:           tuple,
    clases_visibles: set,
    alpha_sarg:      float = 0.75,
) -> np.ndarray:
    overlay = np.zeros((*shape, 4), dtype=np.float32)
    for cid in clases_visibles:
        if not (mask == cid).any():
            continue
        alpha = alpha_sarg if cid in CLASES_SARGASSUM else 0.50
        overlay[mask == cid] = hex_to_rgba(CLASES[cid][1], alpha)
    return overlay


def parches_leyenda(mask: np.ndarray) -> list:
    patches = []
    for cid in sorted(set(np.unique(mask).tolist()) - {0}):
        nombre, color = CLASES[cid]
        marca = "★ " if cid in CLASES_SARGASSUM else ""
        patches.append(mpatches.Patch(
            facecolor=hex_to_rgba(color, 1.0)[:3],
            label=f"{marca}{cid}: {nombre}"
        ))
    return patches


def visualizar(
    img_norm:          np.ndarray,
    mascara_gt:        np.ndarray,
    clase_pred:        np.ndarray,
    prob_sarg_raw:     np.ndarray,
    prob_sarg_filtrada:np.ndarray,
    prob_sarg_suave:   np.ndarray,
    mascara_limpia:    np.ndarray,
    mascara_swin_pura: np.ndarray,   # <--- AÑADIR ESTA LÍNEA
    mapa_fai:          np.ndarray,
    nombre:            str,
    umbral:            float,
    umbral_fai:        float,
    sigma:             float,
    metricas:          dict,
    save_dir:          Path | None = None,
) -> None:
    """6 paneles: RGB | GT | FAI | P(sargazo) | Máscara final | TP/FP/FN"""
    rgb   = img_norm[:, :, :3]
    shape = rgb.shape[:2]

    TARGET = 224
    h, w   = mascara_gt.shape
    y0 = (h - TARGET) // 2
    x0 = (w - TARGET) // 2
    mascara_gt_crop = mascara_gt[y0:y0 + TARGET, x0:x0 + TARGET]

    overlay_gt = crear_overlay(mascara_gt_crop, shape, CLASES_MOSTRAR)
    gt_sarg    = np.isin(mascara_gt_crop, list(CLASES_SARGASSUM))
    pred_sarg  = mascara_limpia.astype(bool)

    comp = np.zeros((*shape, 4), dtype=np.float32)
    comp[gt_sarg  & pred_sarg]  = [0.0, 1.0, 0.0, 0.75]
    comp[~gt_sarg & pred_sarg]  = [1.0, 0.0, 0.0, 0.60]
    comp[gt_sarg  & ~pred_sarg] = [1.0, 1.0, 0.0, 0.75]

    iou2 = metricas.get(2, float("nan"))
    iou3 = metricas.get(3, float("nan"))
    
    # Pre-calculamos los textos 
    iou2_str = f"{iou2:.4f}" if not np.isnan(iou2) else "n/a"
    iou3_str = f"{iou3:.4f}" if not np.isnan(iou3) else "n/a"

    # EL ÚNICO PRINT QUE DEBE HABER (Asegúrate de que no haya otro debajo)
    print(f"  P_max: {prob_sarg_raw.max()*100:.1f}%  |  "
          f"IoU Denso: {iou2_str}  |  "
          f"IoU Escaso: {iou3_str}  |  "
          f"Px: {mascara_limpia.sum()}")

    fig, axes = plt.subplots(1, 6, figsize=(32, 5.5))
    fig.suptitle(
        f"{nombre}  |  IoU Denso: {iou2_str}  IoU Escaso: {iou3_str}  |  "
        f"Umbral: {umbral:.0%}  Sigma: {sigma}  P_max: {prob_sarg_raw.max()*100:.1f}%",
        fontsize=9, fontweight="bold"
    )

    axes[0].imshow(rgb)
    axes[0].set_title("RGB", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(rgb)
    axes[1].imshow(overlay_gt, interpolation="nearest")
    axes[1].set_title("Ground Truth (MADOS)", fontsize=9)
    axes[1].axis("off")
    ley_gt = parches_leyenda(mascara_gt_crop)
    if ley_gt:
        axes[1].legend(handles=ley_gt, fontsize=5.5, loc="lower right",
                       framealpha=0.85, title="Clases GT", title_fontsize=5.5)

    vmax_fai = max(abs(mapa_fai.min()), abs(mapa_fai.max()), 0.001)
    im3 = axes[2].imshow(mapa_fai, cmap="RdYlGn", vmin=-vmax_fai, vmax=vmax_fai)
    if umbral_fai > 0:
        axes[2].contour((mapa_fai >= umbral_fai).astype(float),
                        levels=[0.5], colors="white", linewidths=0.8)
    axes[2].set_title(f"FAI = NIR - Rojo\nBlanco: FAI>={umbral_fai:.3f}", fontsize=8)
    axes[2].axis("off")
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

# Cambiamos el panel de calor por la Visión Pura de la IA
    mascara_pura_rgba = np.zeros((*shape, 4), dtype=np.float32)
    mascara_pura_rgba[mascara_swin_pura == 1] = [1.0, 0.0, 0.0, 0.7] # Rojo translúcido
    axes[3].imshow(rgb)
    axes[3].imshow(mascara_pura_rgba, interpolation="nearest")
    axes[3].set_title(f"Swin Puro (>{umbral:.0%})\n¡Ignorando el FAI!", fontsize=8)
    axes[3].axis("off")

    mascara_rgba = np.zeros((*shape, 4), dtype=np.float32)
    mascara_rgba[mascara_limpia == 1] = [0.12, 0.52, 0.29, 0.82]
    axes[4].imshow(rgb)
    axes[4].imshow(mascara_rgba, interpolation="nearest")
    axes[4].set_title(f"Detección final\n{int(mascara_limpia.sum())} px sargazo", fontsize=8)
    axes[4].axis("off")

    axes[5].imshow(rgb)
    axes[5].imshow(comp, interpolation="nearest")
    tp = int((gt_sarg  &  pred_sarg).sum())
    fp = int((~gt_sarg &  pred_sarg).sum())
    fn = int((gt_sarg  & ~pred_sarg).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    prec_str = f"{prec:.2f}" if not np.isnan(prec) else "n/a"
    rec_str  = f"{rec:.2f}"  if not np.isnan(rec)  else "n/a"
    axes[5].set_title(f"TP/FP/FN  Prec:{prec_str} Rec:{rec_str}", fontsize=8)
    axes[5].axis("off")
    axes[5].legend(handles=[
        mpatches.Patch(color=[0, 1, 0], label=f"TP={tp}"),
        mpatches.Patch(color=[1, 0, 0], label=f"FP={fp}"),
        mpatches.Patch(color=[1, 1, 0], label=f"FN={fn}"),
    ], fontsize=7, loc="lower right", framealpha=0.85,
       title="Píxeles", title_fontsize=6)

    plt.tight_layout()
    out_dir  = save_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"inferencia_{nombre.replace('.npy', '')}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  [OK] Guardado: {out_path}")
    plt.show()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia Swin MADOS v2")
    parser.add_argument("--split",      default="val",
                        choices=["val", "test", "train"])
    parser.add_argument("--n",          type=int,   default=None)
    parser.add_argument("--todas",      action="store_true")
    parser.add_argument("--umbral",     type=float, default=0.35)
    parser.add_argument("--sigma",      type=float, default=2.0)
    parser.add_argument("--min-pixels", type=int,   default=50)
    parser.add_argument("--sin-tta",    action="store_true")
    parser.add_argument("--umbral-fai", type=float, default=0.005)
    parser.add_argument("--dataset",    type=str,   default=str(SARGASSUM_READY))
    parser.add_argument("--modelo",     type=str,   default=None,
                        help="Carpeta del checkpoint (contiene weights.pth)")
    args = parser.parse_args()

    usar_tta = not args.sin_tta

    # ── Cargar modelo ─────────────────────────────────────────────────
    if args.modelo is None:
        # Buscar el checkpoint más reciente si no se especifica uno
        runs = sorted(Path("experiments/runs").glob("*/weights.pth"))
        if not runs:
            print("[ERROR] No se encontró ningún checkpoint en experiments/runs/")
            print("        Especifica uno con --modelo ruta/al/checkpoint/")
            return
        weights_path = runs[-1]
        print(f"[inference] Usando checkpoint más reciente: {weights_path.parent.name}")
    else:
        weights_path = Path(args.modelo) / "weights.pth"

    print(f"[inference] Cargando modelo desde: {weights_path}")
    model = SwinSegmenter(num_classes=NUM_CLASSES).to(DEVICE)
    model.load(checkpoint_dir=weights_path.parent, device=DEVICE)
    model.eval()
    print(f"[inference] {DEVICE.upper()} | TTA: {'ON' if usar_tta else 'OFF'} | "
          f"Umbral: {args.umbral:.0%} | Sigma: {args.sigma}")

    # ── Seleccionar imágenes ──────────────────────────────────────────
    img_dir   = f"{args.dataset}/{args.split}/images"
    mask_dir  = f"{args.dataset}/{args.split}/masks"
    img_paths = sorted(glob.glob(f"{img_dir}/*.npy"))

    if not img_paths:
        print(f"[ERROR] No hay imágenes en: {img_dir}")
        return

    if not args.todas:
        img_paths = [
            ip for ip in img_paths
            if np.isin(np.load(f"{mask_dir}/{Path(ip).name}"),
                       list(CLASES_SARGASSUM)).any()
        ]
        print(f"[inference] Imágenes con sargazo en {args.split}: {len(img_paths)}")
    else:
        print(f"[inference] Total imágenes en {args.split}: {len(img_paths)}")

    if not img_paths:
        print("[AVISO] Sin imágenes. Usa --todas para ver todas.")
        return

    if args.n:
        img_paths = random.sample(img_paths, min(args.n, len(img_paths)))

    print(f"\n[inference] Procesando {len(img_paths)} imágenes...\n")

    iou2_global, iou3_global = [], []

    for i, ip in enumerate(img_paths):
        nombre = Path(ip).name
        print(f"[{i+1}/{len(img_paths)}] {nombre}")

        img_norm, tensor = preprocesar(ip)
        mp = f"{mask_dir}/{nombre}"
        mascara_gt = np.load(mp) if Path(mp).exists() else np.zeros((224, 224), dtype=np.int32)

        clase_pred, prob_sarg_raw, _ = inferir(model, tensor, usar_tta=usar_tta)

        # --- AÑADIR ESTA LÍNEA AQUÍ ---
        mascara_swin_pura = (prob_sarg_raw >= args.umbral).astype(np.float32)
        # ------------------------------

        if args.umbral_fai > 0:
            mascara_fai, mapa_fai = calcular_fai_mask(ip, umbral_fai=args.umbral_fai)
            prob_sarg_filtrada    = prob_sarg_raw * mascara_fai.astype(np.float32)
        else:
            mascara_fai        = np.ones_like(prob_sarg_raw, dtype=bool)
            prob_sarg_filtrada = prob_sarg_raw
            mapa_fai           = np.zeros_like(prob_sarg_raw)

        prob_sarg_suave, mascara_limpia = postprocesar(
            prob_sarg_filtrada,
            umbral=args.umbral, sigma=args.sigma, min_pixels=args.min_pixels,
        )

        TARGET = 224
        h, w   = mascara_gt.shape
        y0 = (h - TARGET) // 2
        x0 = (w - TARGET) // 2
        gt_crop = mascara_gt[y0:y0 + TARGET, x0:x0 + TARGET]

        metricas = metricas_imagen(clase_pred, gt_crop)
        iou2 = metricas.get(2, float("nan"))
        iou3 = metricas.get(3, float("nan"))

        # --- ESTAS SON LAS DOS LÍNEAS QUE FALTABAN ---
        iou2_str = f"{iou2:.4f}" if not np.isnan(iou2) else "n/a"
        iou3_str = f"{iou3:.4f}" if not np.isnan(iou3) else "n/a"
        # ---------------------------------------------

        print(f"  P_max: {prob_sarg_raw.max()*100:.1f}%  |  "
              f"IoU Denso: {iou2_str}  |  "
              f"IoU Escaso: {iou3_str}  |  "
              f"Px: {mascara_limpia.sum()}")

        if not np.isnan(iou2): iou2_global.append(iou2)
        if not np.isnan(iou3): iou3_global.append(iou3)

        visualizar(
                    img_norm, mascara_gt, clase_pred,
                    prob_sarg_raw, prob_sarg_filtrada, prob_sarg_suave,
                    mascara_limpia, mascara_swin_pura, mapa_fai,  # <--- AÑADIDA AQUÍ
                    nombre, args.umbral, args.umbral_fai, args.sigma, metricas,
                )

    print("\n" + "=" * 55)
    print("  RESUMEN")
    print(f"  Imágenes procesadas: {len(img_paths)}")
    if iou2_global:
        print(f"  IoU medio Dense Sarg.  : {np.mean(iou2_global):.4f}")
    if iou3_global:
        print(f"  IoU medio Sparse Algae : {np.mean(iou3_global):.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()