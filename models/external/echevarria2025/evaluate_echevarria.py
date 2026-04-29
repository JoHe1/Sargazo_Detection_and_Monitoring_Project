"""
models/external/echevarria2025/evaluate_echevarria.py
------------------------------------------------------
Evalúa los modelos de Echevarría et al. (2025) sobre el split de test
de MADOS para comparativa justa con el Swin Transformer.

Los modelos de Echevarría son clasificadores de píxeles — reciben los
valores de 4 bandas (Blue, Green, Red, NIR) de un píxel y predicen si
es sargazo o no. Este script extrae píxel a píxel cada tile .npy del
test, aplica cada modelo y calcula métricas contra el GT corregido.

Métricas calculadas (iguales a las del Swin para comparativa):
    - Precision, Recall, F1 para la clase sargazo
    - IoU sargazo (clases 2 y 3 combinadas)
    - mIoU (media sobre todas las clases presentes)

Uso:
    python -m models.external.echevarria2025.evaluate_echevarria
    python -m models.external.echevarria2025.evaluate_echevarria --split val
    python -m models.external.echevarria2025.evaluate_echevarria --n 10
    python -m models.external.echevarria2025.evaluate_echevarria --split test
    python -m models.external.echevarria2025.evaluate_echevarria --split test --modelos-dir models/external/echevarria2025/models_4bands_mados

Salida:
    Tabla de métricas por modelo en consola
    models/external/echevarria2025/evaluacion_test.json
"""

from __future__ import annotations

import argparse
import json
import glob
import random
from pathlib import Path

import numpy as np
from joblib import load
from sklearn.metrics import precision_score, recall_score, f1_score

from core.config.paths import SARGASSUM_READY

# ── Rutas ─────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
MODELS_DIR   = SCRIPT_DIR / "models_4bands"   # sobreescrito desde main() via --modelos-dir

# Clases de sargazo en MADOS
CLASES_SARGASSUM = {2, 3}   # Dense Sargassum + Sparse Floating Algae
NUM_CLASSES      = 16

# Orden de bandas en los .npy de MADOS: (B, G, R, NIR)
# Los modelos de Echevarría se entrenaron con: Blue, Green, Red, NIR
# → índices: 0=Blue, 1=Green, 2=Red, 3=NIR  (coinciden exactamente)
BAND_INDICES = [0, 1, 2, 3]


# ══════════════════════════════════════════════════════════════════════
# CARGA DE MODELOS
# ══════════════════════════════════════════════════════════════════════

def cargar_modelos() -> dict:
    """
    Carga todos los modelos disponibles en MODELS_DIR.

    Detecta automáticamente el sufijo de los archivos:
        - models_4bands/      → sufijo "_4b"        (modelos originales Echevarría)
        - models_4bands_mados/ → sufijo "_4b_mados"  (modelos reentrenados con MADOS)
    """
    modelos = {}

    # Detectar sufijo según la carpeta
    sufijo = "_4b_mados" if "mados" in MODELS_DIR.name else "_4b"

    archivos = {
        "RandomForest": MODELS_DIR / f"randomforest{sufijo}.joblib",
        "XGBoost":      MODELS_DIR / f"xgboost{sufijo}.joblib",
        "KNN":          MODELS_DIR / f"knn{sufijo}.joblib",
        "MLP":          MODELS_DIR / f"mlp{sufijo}.joblib",
    }

    scaler_path = MODELS_DIR / f"scaler{sufijo}.joblib"
    le_path     = MODELS_DIR / f"label_encoder{sufijo}.joblib"

    if not scaler_path.exists():
        print(f"[ERROR] Scaler no encontrado: {scaler_path}")
        print(f"        Ejecuta primero el script de reentrenamiento correspondiente")
        return {}

    scaler = load(scaler_path)
    le     = load(le_path) if le_path.exists() else None

    # Determinar índice de la clase sargazo en el LabelEncoder
    sarg_idx = 1  # default
    if le is not None:
        clases = list(le.classes_)
        if "sargassum" in clases:
            sarg_idx = clases.index("sargassum")
        print(f"[info] Clases LabelEncoder: {clases}  → sargassum_idx={sarg_idx}")

    for nombre, path in archivos.items():
        if path.exists():
            modelos[nombre] = {"model": load(path), "scaler": scaler,
                               "sarg_idx": sarg_idx, "tipo": "sklearn"}
            print(f"  ✔ {nombre} cargado")
        else:
            print(f"  · {nombre} no encontrado ({path.name})")

    # CNN (TensorFlow)
    cnn_path = MODELS_DIR / f"cnn{sufijo}.keras"
    if cnn_path.exists():
        try:
            import tensorflow as tf
            cnn = tf.keras.models.load_model(str(cnn_path))
            modelos["CNN-1D"] = {"model": cnn, "scaler": scaler,
                                 "sarg_idx": sarg_idx, "tipo": "keras"}
            print(f"  ✔ CNN-1D cargado")
        except Exception as e:
            print(f"  · CNN-1D no disponible: {e}")

    return modelos


# ══════════════════════════════════════════════════════════════════════
# EXTRACCIÓN DE PÍXELES DE UN TILE
# ══════════════════════════════════════════════════════════════════════

def extraer_pixeles(img_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrae píxeles de un tile .npy para clasificación.

    Returns:
        X        : (N, 4) float32 — valores de banda por píxel
        y_gt     : (N,)   int    — clase GT por píxel (0-15)
        coords   : (N, 2) int    — coordenadas (row, col) en el tile 224x224
    """
    TARGET = 224

    img_raw  = np.load(img_path).astype(np.float32)
    mask_raw = np.load(mask_path).astype(np.int32)

    # Normalizar igual que en entrenamiento del Swin
    if img_raw.max() > 10.0:
        img_raw = img_raw / 10000.0

    # Center crop 224x224
    h, w = img_raw.shape[:2]
    y0   = (h - TARGET) // 2
    x0   = (w - TARGET) // 2
    img  = img_raw[y0:y0+TARGET, x0:x0+TARGET, :]   # (224, 224, 4)

    mh, mw = mask_raw.shape
    my0    = (mh - TARGET) // 2
    mx0    = (mw - TARGET) // 2
    mask   = mask_raw[my0:my0+TARGET, mx0:mx0+TARGET]  # (224, 224)

    # Aplanar: cada píxel es una fila con 4 valores de banda
    # Orden npy: (B, G, R, NIR) → coincide con Blue, Green, Red, NIR del CSV
    X      = img.reshape(-1, 4)[:, BAND_INDICES]   # (224*224, 4)
    y_gt   = mask.reshape(-1)                       # (224*224,)

    rows, cols = np.meshgrid(np.arange(TARGET), np.arange(TARGET), indexing="ij")
    coords = np.stack([rows.reshape(-1), cols.reshape(-1)], axis=1)

    return X, y_gt, coords


# ══════════════════════════════════════════════════════════════════════
# PREDICCIÓN CON UN MODELO
# ══════════════════════════════════════════════════════════════════════

def predecir(modelo_info: dict, X: np.ndarray) -> np.ndarray:
    """
    Predice clase sargazo/no-sargazo para cada píxel.

    Returns:
        pred_sarg: (N,) bool — True si el modelo predice sargazo
    """
    scaler   = modelo_info["scaler"]
    model    = modelo_info["model"]
    sarg_idx = modelo_info["sarg_idx"]
    tipo     = modelo_info["tipo"]

    X_scaled = scaler.transform(X)

    if tipo == "sklearn":
        y_pred = model.predict(X_scaled)
        return y_pred == sarg_idx

    elif tipo == "keras":
        X_cnn  = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        probs  = model.predict(X_cnn, verbose=0).flatten()
        return probs >= 0.5

    return np.zeros(len(X), dtype=bool)


# ══════════════════════════════════════════════════════════════════════
# MÉTRICAS
# ══════════════════════════════════════════════════════════════════════

def calcular_metricas_tile(pred_sarg: np.ndarray, y_gt: np.ndarray) -> dict:
    """Calcula métricas para un tile."""
    gt_sarg = np.isin(y_gt, list(CLASES_SARGASSUM))

    tp = int((pred_sarg &  gt_sarg).sum())
    fp = int((pred_sarg & ~gt_sarg).sum())
    fn = int((~pred_sarg & gt_sarg).sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1   = 2 * prec * rec / (prec + rec) if (not np.isnan(prec) and
                                              not np.isnan(rec) and
                                              (prec + rec) > 0) else float("nan")

    # IoU sargazo
    iou_sarg = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")

    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
            "iou_sargassum": iou_sarg}


def agregar_metricas(resultados_tiles: list[dict]) -> dict:
    """Agrega métricas de todos los tiles."""
    tp_total = sum(r["tp"] for r in resultados_tiles)
    fp_total = sum(r["fp"] for r in resultados_tiles)
    fn_total = sum(r["fn"] for r in resultados_tiles)

    prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    rec  = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    iou  = tp_total / (tp_total + fp_total + fn_total) if (tp_total + fp_total + fn_total) > 0 else 0.0

    # IoU medio por tile (solo tiles con sargazo)
    ious_validos = [r["iou_sargassum"] for r in resultados_tiles
                    if not np.isnan(r["iou_sargassum"])]
    iou_medio = float(np.mean(ious_validos)) if ious_validos else 0.0

    return {
        "precision":        round(prec, 4),
        "recall":           round(rec,  4),
        "f1":               round(f1,   4),
        "iou_sargazo_global": round(iou,  4),
        "iou_sargazo_medio":  round(iou_medio, 4),
        "tp_total":         tp_total,
        "fp_total":         fp_total,
        "fn_total":         fn_total,
        "tiles_evaluados":  len(resultados_tiles),
    }


# ══════════════════════════════════════════════════════════════════════
# EVALUACIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

def evaluar(split: str = "test", n: int | None = None, solo_sargassum: bool = True) -> None:
    img_dir  = SARGASSUM_READY / split / "images"
    mask_dir = SARGASSUM_READY / split / "masks"

    if not img_dir.exists():
        print(f"[ERROR] No existe: {img_dir}")
        return

    # Seleccionar tiles
    todos = sorted(img_dir.glob("*.npy"))
    if solo_sargassum:
        tiles = []
        for p in todos:
            mp = mask_dir / p.name
            if mp.exists() and np.isin(np.load(mp), list(CLASES_SARGASSUM)).any():
                tiles.append(p)
    else:
        tiles = [p for p in todos if (mask_dir / p.name).exists()]

    if n:
        tiles = random.sample(tiles, min(n, len(tiles)))

    # Etiqueta para identificar qué modelos se están evaluando
    etiqueta = "mados" if "mados" in MODELS_DIR.name else "echevarria"
    print(f"\n[Evaluación] split={split}  tiles={len(tiles)}  "
          f"solo_sargazo={solo_sargassum}  modelos={etiqueta}")

    # Cargar modelos
    print("\n[Cargando modelos...]")
    modelos = cargar_modelos()
    if not modelos:
        return

    # Evaluar cada modelo
    resultados_globales = {}

    for nombre_modelo, modelo_info in modelos.items():
        print(f"\n── {nombre_modelo} ──────────────────────────")
        resultados_tiles = []

        for i, img_path in enumerate(tiles):
            mask_path = mask_dir / img_path.name
            try:
                X, y_gt, _ = extraer_pixeles(img_path, mask_path)

                # Predecir en lotes para no saturar memoria
                BATCH = 10000
                pred_sarg = np.zeros(len(X), dtype=bool)
                for b in range(0, len(X), BATCH):
                    pred_sarg[b:b+BATCH] = predecir(modelo_info, X[b:b+BATCH])

                metricas = calcular_metricas_tile(pred_sarg, y_gt)
                resultados_tiles.append(metricas)

                iou_str  = f"{metricas['iou_sargassum']:.4f}" if not np.isnan(metricas['iou_sargassum']) else "n/a"
                prec_str = f"{metricas['precision']:.2f}"     if not np.isnan(metricas['precision'])     else "n/a"
                rec_str  = f"{metricas['recall']:.2f}"        if not np.isnan(metricas['recall'])        else "n/a"
                print(f"  [{i+1}/{len(tiles)}] {img_path.name}  "
                      f"IoU={iou_str}  Prec={prec_str}  Rec={rec_str}  "
                      f"TP={metricas['tp']} FP={metricas['fp']} FN={metricas['fn']}")

            except Exception as e:
                print(f"  [ERROR] {img_path.name}: {e}")

        if resultados_tiles:
            resumen = agregar_metricas(resultados_tiles)
            resultados_globales[nombre_modelo] = resumen
            print(f"\n  RESUMEN {nombre_modelo}:")
            print(f"    Precision : {resumen['precision']:.4f}")
            print(f"    Recall    : {resumen['recall']:.4f}")
            print(f"    F1        : {resumen['f1']:.4f}")
            print(f"    IoU sarg. : {resumen['iou_sargazo_global']:.4f}")

    # Tabla comparativa final
    print("\n" + "═" * 65)
    print(f"  COMPARATIVA FINAL — split={split}  modelos={etiqueta}")
    print("═" * 65)
    print(f"  {'Modelo':<16} {'Precision':>10} {'Recall':>8} {'F1':>8} {'IoU Sarg':>10}")
    print("  " + "─" * 55)
    for nombre, res in resultados_globales.items():
        print(f"  {nombre:<16} {res['precision']:>10.4f} {res['recall']:>8.4f} "
              f"{res['f1']:>8.4f} {res['iou_sargazo_global']:>10.4f}")
    print("═" * 65)

    # Guardar JSON con nombre que identifica los modelos usados
    out_path = SCRIPT_DIR / f"evaluacion_{split}_{etiqueta}.json"
    with open(out_path, "w") as f:
        json.dump(resultados_globales, f, indent=2)
    print(f"\n[Guardado] {out_path}")


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evalúa modelos de Echevarría et al. sobre MADOS"
    )
    parser.add_argument("--split",  default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--n",      type=int, default=None,
                        help="Número de tiles aleatorios (default: todos con sargazo)")
    parser.add_argument("--todas",  action="store_true",
                        help="Evaluar también tiles sin sargazo")
    parser.add_argument(
        "--modelos-dir", type=Path, default=None,
        help=(
            "Carpeta con los modelos a evaluar. "
            "Por defecto: models_4bands/ (modelos originales Echevarría). "
            "Para los modelos reentrenados con MADOS: models_4bands_mados/"
        )
    )
    args = parser.parse_args()

    # Sobreescribir MODELS_DIR si se especifica
    global MODELS_DIR
    if args.modelos_dir is not None:
        MODELS_DIR = args.modelos_dir
    print(f"[info] Carpeta de modelos: {MODELS_DIR}")

    evaluar(
        split          = args.split,
        n              = args.n,
        solo_sargassum = not args.todas,
    )


if __name__ == "__main__":
    main()