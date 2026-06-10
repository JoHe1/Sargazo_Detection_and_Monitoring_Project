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
import matplotlib
matplotlib.use('Agg')   # backend no interactivo — evita errores tkinter en bucles largos
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from core.config.paths import SARGASSUM_READY

# ── Rutas ─────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
MODELS_DIR   = SCRIPT_DIR / "models_4bands"
RESULTS_DIR  = SCRIPT_DIR / "resultados"   # carpeta de salida para tablas y JSON

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

    # Limpiar NaN/Inf — píxeles de nube o borde sin datos
    img_raw = np.nan_to_num(img_raw, nan=0.0, posinf=1.0, neginf=0.0)

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
    # Limpiar NaN residuales tras el escalado (por si el scaler los introduce)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

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
# VISUALIZACIÓN DE TILES
# ══════════════════════════════════════════════════════════════════════

def visualizar_tile(
    img_path:      Path,
    mask_path:     Path,
    modelos:       dict,
    etiqueta:      str,
    save_dir:      Path,
    skip_show:     bool = False,
) -> None:
    """
    Genera una figura con un panel por modelo + GT.
    Paneles: pred_modelo1 | pred_modelo2 | ... | GT
    Guardada en save_dir/{escena}_{etiqueta}.png

    Si skip_show=True guarda la figura sin mostrarla en pantalla.
    """
    TARGET = 224

    # ── Cargar imagen ────────────────────────────────────────────────
    img_raw  = np.load(img_path).astype(np.float32)
    mask_raw = np.load(mask_path).astype(np.int32)

    if img_raw.max() > 10.0:
        img_raw = img_raw / 10000.0
    img_raw = np.nan_to_num(img_raw, nan=0.0, posinf=1.0, neginf=0.0)
    img_raw = np.clip(img_raw * 5.0, 0.0, 1.0)

    h, w = img_raw.shape[:2]
    y0   = (h - TARGET) // 2
    x0   = (w - TARGET) // 2
    img  = img_raw[y0:y0+TARGET, x0:x0+TARGET, :]

    mh, mw = mask_raw.shape
    my0    = (mh - TARGET) // 2
    mx0    = (mw - TARGET) // 2
    mask   = mask_raw[my0:my0+TARGET, mx0:mx0+TARGET]

    rgb     = img[:, :, [2, 1, 0]]
    gt_sarg = np.isin(mask, list(CLASES_SARGASSUM))

    # ── Predicciones ──────────────────────────────────────────────────
    img_norm_raw = np.load(img_path).astype(np.float32)
    if img_norm_raw.max() > 10.0:
        img_norm_raw = img_norm_raw / 10000.0
    img_norm_raw = np.nan_to_num(img_norm_raw, nan=0.0, posinf=1.0, neginf=0.0)
    img_crop_raw = img_norm_raw[y0:y0+TARGET, x0:x0+TARGET, :]
    X = img_crop_raw.reshape(-1, 4)[:, BAND_INDICES]

    predicciones = {}
    for nombre_modelo, modelo_info in modelos.items():
        BATCH = 10000
        pred_flat = np.zeros(TARGET * TARGET, dtype=bool)
        for b in range(0, len(X), BATCH):
            pred_flat[b:b+BATCH] = predecir(modelo_info, X[b:b+BATCH])
        predicciones[nombre_modelo] = pred_flat.reshape(TARGET, TARGET)

    # ── Layout: pred×N | GT ───────────────────────────────────────────
    n_modelos = len(modelos)
    n_paneles = n_modelos + 1   # modelos + GT
    escena    = img_path.stem

    fig, axes = plt.subplots(1, n_paneles, figsize=(6 * n_paneles, 6))
    fig.suptitle(
        f"{escena}  —  Modelos Echevarría et al. ({etiqueta})",
        fontsize=14, fontweight="bold"
    )

    # Paneles de predicción por modelo
    for idx, (nombre_modelo, pred_mapa) in enumerate(predicciones.items()):
        ax   = axes[idx]
        tp   = int(( pred_mapa &  gt_sarg).sum())
        fp   = int(( pred_mapa & ~gt_sarg).sum())
        fn   = int((~pred_mapa &  gt_sarg).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1   = 2*prec*rec/(prec+rec) if not (np.isnan(prec) or np.isnan(rec)) and (prec+rec) > 0 else float("nan")
        prec_str = f"{prec:.2f}" if not np.isnan(prec) else "n/a"
        rec_str  = f"{rec:.2f}"  if not np.isnan(rec)  else "n/a"
        f1_str   = f"{f1:.2f}"   if not np.isnan(f1)   else "n/a"

        comp = np.zeros((TARGET, TARGET, 4), dtype=np.float32)
        comp[ pred_mapa &  gt_sarg] = [0.0, 1.0, 0.0, 0.75]
        comp[ pred_mapa & ~gt_sarg] = [1.0, 0.0, 0.0, 0.60]
        comp[~pred_mapa &  gt_sarg] = [1.0, 1.0, 0.0, 0.75]

        ax.imshow(rgb)
        ax.imshow(comp, interpolation="nearest")
        ax.set_title(
            f"{nombre_modelo}\nPrec: {prec_str}  Rec: {rec_str}  F1: {f1_str}",
            fontsize=13, fontweight="bold"
        )
        ax.axis("off")
        ax.legend(
            handles=[
                mpatches.Patch(color=[0,1,0], label=f"TP = {tp}"),
                mpatches.Patch(color=[1,0,0], label=f"FP = {fp}"),
                mpatches.Patch(color=[1,1,0], label=f"FN = {fn}"),
            ],
            fontsize=9, loc="lower right", framealpha=0.85
        )

    # Último panel: Ground Truth
    ax_gt = axes[-1]
    gt_overlay = np.zeros((TARGET, TARGET, 4), dtype=np.float32)
    gt_overlay[gt_sarg] = [0.0, 0.8, 0.2, 0.75]
    ax_gt.imshow(rgb)
    ax_gt.imshow(gt_overlay, interpolation="nearest")
    ax_gt.set_title("Ground Truth\n(Sargazo MADOS)", fontsize=13, fontweight="bold")
    ax_gt.axis("off")
    ax_gt.legend(
        handles=[mpatches.Patch(color=[0, 0.8, 0.2], label="Sargazo GT")],
        fontsize=9, loc="lower right", framealpha=0.85
    )

    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{escena}_{etiqueta}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  [OK] Imagen guardada: {out_path}")
    if not skip_show:
        plt.show()
    plt.close(fig)




def generar_tabla_matplotlib(
    resultados_globales: dict,
    split: str,
    etiqueta: str,
    sufijo: str = "",
) -> None:
    """
    Genera y guarda una tabla matplotlib con los resultados de todos
    los modelos evaluados, lista para incluir en Overleaf.
    Guardada en RESULTS_DIR/tabla_{split}_{etiqueta}{sufijo}.png
    """
    if not resultados_globales:
        return

    modelos  = list(resultados_globales.keys())
    metricas = ["Precision", "Recall", "F1", "IoU Sargazo Global",
                "IoU Sargazo Medio", "TP Total", "FP Total", "FN Total"]

    # Construir filas: una fila por métrica, una columna por modelo
    claves = ["precision", "recall", "f1", "iou_sargazo_global",
              "iou_sargazo_medio", "tp_total", "fp_total", "fn_total"]

    filas = []
    for clave, metrica in zip(claves, metricas):
        fila = [metrica]
        for modelo in modelos:
            val = resultados_globales[modelo].get(clave, "n/a")
            if isinstance(val, float):
                fila.append(f"{val:.4f}")
            else:
                fila.append(str(val))
        filas.append(fila)

    col_labels = ["Métrica"] + modelos
    n_filas    = len(filas)
    n_cols     = len(col_labels)

    fig, ax = plt.subplots(figsize=(3 + 2.5 * len(modelos), n_filas * 0.55 + 1.5))
    ax.axis("off")

    tabla = ax.table(
        cellText   = filas,
        colLabels  = col_labels,
        cellLoc    = "center",
        loc        = "center",
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 1.6)

    # Estilo cabecera
    for col in range(n_cols):
        tabla[(0, col)].set_facecolor("#2c3e50")
        tabla[(0, col)].set_text_props(color="white", fontweight="bold")

    # Filas alternadas + destacar F1 e IoU global en verde
    destacadas = {3, 4}   # F1 (fila 3) e IoU global (fila 4) base-1
    for row in range(1, n_filas + 1):
        color_bg = "#eaf4fb" if row % 2 == 0 else "white"
        if row in destacadas:
            color_bg = "#d5f5e3"
        elif row in {6}:   # TP
            color_bg = "#d5f5e3"
        elif row in {7}:   # FP
            color_bg = "#fadbd8"
        elif row in {8}:   # FN
            color_bg = "#fef9e7"
        for col in range(n_cols):
            tabla[(row, col)].set_facecolor(color_bg)

    titulo = (
        f"Evaluación modelos Echevarría et al. — {etiqueta}\n"
        f"Split: {split}  |  Dataset: MADOS (Sentinel-2, 10m)"
    )
    fig.suptitle(titulo, fontsize=11, fontweight="bold", y=0.98)

    plt.tight_layout()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"tabla_{split}_{etiqueta}{sufijo}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  [OK] Tabla guardada: {out_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# EVALUACIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

def evaluar(
    split:          str = "test",
    n:              int | None = None,
    solo_sargassum: bool = True,
    tiles_fijos:    list[str] | None = None,
    skip_show:      bool = False,
) -> None:
    img_dir  = SARGASSUM_READY / split / "images"
    mask_dir = SARGASSUM_READY / split / "masks"

    if not img_dir.exists():
        print(f"[ERROR] No existe: {img_dir}")
        return

    etiqueta = "mados" if "mados" in MODELS_DIR.name else "echevarria"

    # ── Seleccionar tiles ─────────────────────────────────────────────
    if tiles_fijos is not None:
        # Modo --tiles: usar exactamente los tiles indicados, sin filtro sargazo
        tiles = []
        for nombre in tiles_fijos:
            p = img_dir / nombre
            if p.exists():
                tiles.append(p)
            else:
                print(f"[AVISO] Tile no encontrado: {nombre}")
        print(f"[Evaluación] Modo --tiles: {len(tiles)} tiles específicos")
    elif solo_sargassum:
        todos  = sorted(img_dir.glob("*.npy"))
        tiles  = []
        for p in todos:
            mp = mask_dir / p.name
            if mp.exists() and np.isin(np.load(mp), list(CLASES_SARGASSUM)).any():
                tiles.append(p)
    else:
        todos = sorted(img_dir.glob("*.npy"))
        tiles = [p for p in todos if (mask_dir / p.name).exists()]

    if n and tiles_fijos is None:
        tiles = random.sample(tiles, min(n, len(tiles)))

    print(f"\n[Evaluación] split={split}  tiles={len(tiles)}  modelos={etiqueta}")

    # ── Cargar modelos ────────────────────────────────────────────────
    print("\n[Cargando modelos...]")
    modelos = cargar_modelos()
    if not modelos:
        return

    # ── Evaluar cada modelo ───────────────────────────────────────────
    resultados_globales = {}

    for nombre_modelo, modelo_info in modelos.items():
        print(f"\n── {nombre_modelo} ──────────────────────────")
        resultados_tiles = []

        for i, img_path in enumerate(tiles):
            mask_path = mask_dir / img_path.name
            try:
                X, y_gt, _ = extraer_pixeles(img_path, mask_path)

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

    # ── Tabla comparativa en consola ──────────────────────────────────
    print("\n" + "═" * 65)
    print(f"  COMPARATIVA FINAL — split={split}  modelos={etiqueta}")
    print("═" * 65)
    print(f"  {'Modelo':<16} {'Precision':>10} {'Recall':>8} {'F1':>8} {'IoU Sarg':>10}")
    print("  " + "─" * 55)
    for nombre, res in resultados_globales.items():
        print(f"  {nombre:<16} {res['precision']:>10.4f} {res['recall']:>8.4f} "
              f"{res['f1']:>8.4f} {res['iou_sargazo_global']:>10.4f}")
    print("═" * 65)

    # ── Tabla matplotlib + JSON ───────────────────────────────────────
    sufijo = "" if tiles_fijos is not None else "_all"
    generar_tabla_matplotlib(resultados_globales, split, etiqueta, sufijo=sufijo)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sufijo = "" if tiles_fijos is not None else "_all"
    out_path = RESULTS_DIR / f"evaluacion_{split}_{etiqueta}{sufijo}.json"
    with open(out_path, "w") as f:
        json.dump(resultados_globales, f, indent=2)
    print(f"\n[Guardado] {out_path}")

    # ── Imágenes por tile (solo en modo --tiles) ──────────────────────
    if tiles_fijos is not None:
        print(f"\n[Generando imágenes por tile...]")
        vis_dir = RESULTS_DIR / f"visualizaciones_{etiqueta}"
        for img_path in tiles:
            mask_path = mask_dir / img_path.name
            if mask_path.exists():
                visualizar_tile(
                    img_path, mask_path, modelos,
                    etiqueta, vis_dir,
                    skip_show=skip_show,
                )
        print(f"[OK] Imágenes guardadas en: {vis_dir}")


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
    parser.add_argument("--tiles",  nargs="+", default=None,
                        help="Nombres exactos de tiles a procesar y visualizar. "
                             "Ej: Scene_135_10.npy Scene_141_22.npy. "
                             "Activa generación de imágenes por tile.")
    parser.add_argument("--skip",   action="store_true",
                        help="Guardar imágenes sin mostrarlas en pantalla (más rápido)")
    parser.add_argument(
        "--modelos-dir", type=Path, default=None,
        help=(
            "Carpeta con los modelos a evaluar. "
            "Por defecto: models_4bands/ (modelos originales Echevarría). "
            "Para los modelos reentrenados con MADOS: models_4bands_mados/"
        )
    )
    args = parser.parse_args()

    global MODELS_DIR
    if args.modelos_dir is not None:
        MODELS_DIR = args.modelos_dir
    print(f"[info] Carpeta de modelos: {MODELS_DIR}")

    evaluar(
        split          = args.split,
        n              = args.n,
        solo_sargassum = not args.todas,
        tiles_fijos    = args.tiles,
        skip_show      = args.skip,
    )


if __name__ == "__main__":
    main()