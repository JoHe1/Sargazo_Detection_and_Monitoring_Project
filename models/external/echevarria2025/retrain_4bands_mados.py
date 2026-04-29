"""
models/external/echevarria2025/retrain_4bands_mados.py
-------------------------------------------------------
Reentrenamiento de los modelos de Echevarría et al. (2025) usando los
mismos datos de entrenamiento que el Swin Transformer del TFG.

Diferencia principal respecto a retrain_4bands.py:
    - Los datos se extraen de los tiles .npy de MADOS (train/val/test)
      en lugar del sargassum_data.csv de Echevarría.
    - Esto permite una comparativa simétrica entre los modelos externos
      y el Swin Transformer, ya que todos se evalúan sobre los mismos
      datos de entrenamiento y test.

Extracción de píxeles:
    - Cada tile .npy de imágenes tiene shape (H, W, 4) en orden (B, G, R, NIR)
    - Cada tile .npy de máscaras tiene shape (H, W) con IDs de clase 0-15
    - Solo se extraen píxeles anotados (clase != 0)
    - Las clases de sargazo (2=Dense Sargassum, 3=Sparse Floating Algae)
      se etiquetan como "sargassum", el resto como "water_other"
    - Se aplica submuestreo de la clase mayoritaria para equilibrar el dataset
      (ratio configurable, por defecto 10:1 agua/sargazo)

Split:
    - Train: tiles del split train/ de MADOS
    - Test:  tiles del split test/ de MADOS
    - Val:   tiles del split val/ de MADOS (incluida en train para los modelos sklearn)

Salida (en models/external/echevarria2025/models_4bands_mados/):
    randomforest_4b_mados.joblib
    xgboost_4b_mados.joblib
    knn_4b_mados.joblib
    mlp_4b_mados.joblib
    cnn_4b_mados.keras
    scaler_4b_mados.joblib
    label_encoder_4b_mados.joblib
    resultados_4b_mados.json

Uso:
    # Todos los modelos con GridSearchCV
    python -m models.external.echevarria2025.retrain_4bands_mados

    # Solo modelos rápidos sin GridSearchCV
    python -m models.external.echevarria2025.retrain_4bands_mados --sin-gridsearch

    # Solo algunos modelos
    python -m models.external.echevarria2025.retrain_4bands_mados --solo xgboost knn

    # Ruta personalizada al dataset
    python -m models.external.echevarria2025.retrain_4bands_mados --dataset ruta/al/dataset
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False
    print("[AVISO] xgboost no instalado: pip install xgboost")

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import (
        Conv1D, Dense, Dropout, Flatten, Input, MaxPooling1D,
    )
    from tensorflow.keras.models import Sequential
    try:
        AdamOptimizer = tf.keras.optimizers.Adam
    except AttributeError:
        AdamOptimizer = tf.keras.optimizers.legacy.Adam
    tf.random.set_seed(42)
    TF_OK = True
except ImportError:
    TF_OK = False
    print("[AVISO] TensorFlow no instalado: pip install tensorflow")


# ── Configuración ──────────────────────────────────────────────────────────────
SEED           = 42
N_SPLITS_CV    = 5
FEATURE_COLUMNS = ["Blue", "Green", "Red", "NIR"]

# Clases MADOS que se consideran sargazo
SARGASSUM_CLASSES = {2, 3}   # 2=Dense Sargassum, 3=Sparse Floating Algae
POSITIVE_LABEL    = "sargassum"
NEGATIVE_LABEL    = "water_other"

# Ratio máximo agua/sargazo para submuestreo (evita colapso de modelos)
# Con ratio=10 → por cada píxel de sargazo se usan máximo 10 de agua
SUBSAMPLE_RATIO = 10

# Normalización igual que MADOSDataset
SCALE_FACTOR    = 5.0     # multiplicado tras dividir por 10000
MAX_REFLECTANCE = 10.0    # umbral para detectar valores sin normalizar

CNN_EPOCHS           = 100
CNN_BATCH_SIZE       = 64
CNN_VALIDATION_SPLIT = 0.20
CNN_PATIENCE         = 15

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "models_4bands_mados"
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACCIÓN DE PÍXELES DESDE TILES .NPY
# ══════════════════════════════════════════════════════════════════════════════

def extraer_pixeles_split(
    dataset_dir: Path,
    split: str,
    target_size: int = 224,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrae píxeles anotados de todos los tiles .npy de un split de MADOS.

    Args:
        dataset_dir: ruta raíz del dataset (contiene train/, val/, test/)
        split:       nombre del split ("train", "val", "test")
        target_size: tamaño del crop central aplicado en MADOSDataset

    Returns:
        X: (N, 4) float32 — bandas [B, G, R, NIR] normalizadas
        y: (N,)   int     — 1=sargazo, 0=agua/otro
    """
    img_dir  = dataset_dir / split / "images"
    mask_dir = dataset_dir / split / "masks"

    img_paths = sorted(img_dir.glob("*.npy"))
    if not img_paths:
        print(f"  [AVISO] No hay tiles en {img_dir}")
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int32)

    X_list, y_list = [], []
    n_sarg = 0
    n_agua = 0

    for img_path in img_paths:
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            continue

        img  = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.uint8)

        # Normalización idéntica a MADOSDataset.__getitem__
        if img.max() > MAX_REFLECTANCE:
            img = img / 10000.0
        img = np.clip(img * SCALE_FACTOR, 0.0, 1.0)

        # Crop central igual que MADOSDataset
        h, w = img.shape[:2]
        y0   = (h - target_size) // 2
        x0   = (w - target_size) // 2
        img  = img[y0:y0+target_size, x0:x0+target_size, :]    # (224, 224, 4)
        mask = mask[y0:y0+target_size, x0:x0+target_size]      # (224, 224)

        # Solo píxeles anotados (clase != 0)
        anotados = mask != 0
        if not anotados.any():
            continue

        # Extraer píxeles anotados
        # img orden: (B, G, R, NIR) — igual que MADOS original
        pixeles = img[anotados]                     # (N, 4)
        clases  = mask[anotados]                    # (N,)

        # Binarizar: sargazo vs resto
        etiquetas = np.isin(clases, list(SARGASSUM_CLASSES)).astype(np.int32)

        X_list.append(pixeles)
        y_list.append(etiquetas)

        n_sarg += etiquetas.sum()
        n_agua += (etiquetas == 0).sum()

    if not X_list:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int32)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    print(f"  Split {split:5s}: {len(X):>10,} píxeles "
          f"| sargazo={n_sarg:>8,} ({100*n_sarg/len(X):.3f}%) "
          f"| agua/otro={n_agua:>10,}")

    return X, y


def submuestrear(
    X: np.ndarray,
    y: np.ndarray,
    ratio: int = SUBSAMPLE_RATIO,
    seed:  int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Submuestrea la clase mayoritaria para equilibrar el dataset.

    Con ratio=10: por cada píxel de sargazo se mantienen máximo 10 de agua.
    Esto es necesario porque el desbalance extremo (~0.002%) hace que
    KNN y Random Forest colapsen a predecir siempre la clase mayoritaria.

    Args:
        X:     features (N, 4)
        y:     etiquetas (N,) binarias
        ratio: máximo ratio negativo/positivo
        seed:  semilla para reproducibilidad

    Returns:
        X_bal, y_bal balanceados
    """
    rng       = np.random.RandomState(seed)
    idx_sarg  = np.where(y == 1)[0]
    idx_agua  = np.where(y == 0)[0]

    n_sarg    = len(idx_sarg)
    n_agua_max = min(len(idx_agua), n_sarg * ratio)

    idx_agua_sub = rng.choice(idx_agua, size=n_agua_max, replace=False)
    idx_total    = np.concatenate([idx_sarg, idx_agua_sub])
    rng.shuffle(idx_total)

    print(f"  Submuestreo: {len(idx_sarg):,} sargazo + {n_agua_max:,} agua "
          f"= {len(idx_total):,} total (ratio 1:{ratio})")

    return X[idx_total], y[idx_total]


def cargar_datos_mados(dataset_dir: Path) -> tuple:
    """
    Carga y prepara todos los splits de MADOS para entrenamiento.

    Train + Val se combinan para entrenar los modelos sklearn.
    Test se usa para evaluación final.

    Returns:
        X_train_s, X_test_s: features normalizadas
        y_train, y_test:     etiquetas binarias
        scaler, le:          scaler y label encoder ajustados
        sarg_idx:            índice de la clase sargazo en le
        scale_pos_weight:    peso para XGBoost
    """
    print("\n[datos] Extrayendo píxeles de MADOS...")
    print(f"[datos] Dataset: {dataset_dir}")
    print(f"[datos] Bandas:  {FEATURE_COLUMNS} (orden B,G,R,NIR)")
    print(f"[datos] Clases sargazo: {SARGASSUM_CLASSES} "
          f"(Dense Sargassum + Sparse Floating Algae)")

    X_train_raw, y_train_raw = extraer_pixeles_split(dataset_dir, "train")
    X_val_raw,   y_val_raw   = extraer_pixeles_split(dataset_dir, "val")
    X_test_raw,  y_test_raw  = extraer_pixeles_split(dataset_dir, "test")

    # Combinar train + val para entrenamiento
    X_trainval = np.concatenate([X_train_raw, X_val_raw], axis=0)
    y_trainval = np.concatenate([y_train_raw, y_val_raw], axis=0)

    print(f"\n[datos] Total antes de submuestreo:")
    print(f"  Train+Val: {len(X_trainval):,} | Test: {len(X_test_raw):,}")

    # Submuestrear para equilibrar
    X_train_bal, y_train_bal = submuestrear(X_trainval, y_trainval)
    X_test_bal,  y_test_bal  = submuestrear(X_test_raw,  y_test_raw)

    # Label encoder — binario: 0=water_other, 1=sargassum
    le = LabelEncoder()
    le.fit([NEGATIVE_LABEL, POSITIVE_LABEL])
    sarg_idx = list(le.classes_).index(POSITIVE_LABEL)

    # Scaler ajustado sobre train
    scaler      = StandardScaler()
    X_train_s   = scaler.fit_transform(X_train_bal)
    X_test_s    = scaler.transform(X_test_bal)

    scale_pos_weight = (y_train_bal == 0).sum() / max((y_train_bal == 1).sum(), 1)

    print(f"\n[datos] Train balanceado: {len(X_train_s):,} "
          f"| sargazo={y_train_bal.sum():,} | agua={(y_train_bal==0).sum():,}")
    print(f"[datos] Test  balanceado: {len(X_test_s):,}  "
          f"| sargazo={y_test_bal.sum():,} | agua={(y_test_bal==0).sum():,}")
    print(f"[datos] scale_pos_weight para XGBoost: {scale_pos_weight:.2f}")

    return (X_train_s, X_test_s, y_train_bal, y_test_bal,
            scaler, le, sarg_idx, scale_pos_weight)


# ══════════════════════════════════════════════════════════════════════════════
# GRIDS DE HIPERPARÁMETROS — idénticos al paper original
# ══════════════════════════════════════════════════════════════════════════════

def get_param_grids(scale_pos_weight: float = 1.0) -> dict:
    return {
        "RandomForest": {
            "n_estimators":      [100, 150],
            "max_depth":         [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf":  [1, 3],
            "class_weight":      ["balanced", "balanced_subsample", None],
            "max_features":      ["sqrt", "log2"],
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 9],
            "weights":     ["uniform", "distance"],
            "metric":      ["euclidean", "manhattan", "minkowski"],
        },
        "XGBoost": {
            "n_estimators":     [100, 150],
            "learning_rate":    [0.05, 0.1],
            "max_depth":        [3, 5, 7],
            "subsample":        [0.7, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0],
            "gamma":            [0, 0.1, 0.2],
        },
        "MLP": {
            "hidden_layer_sizes": [(50,), (100,), (50, 25), (100, 50)],
            "activation":         ["relu", "tanh"],
            "solver":             ["adam"],
            "alpha":              [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.005],
            "early_stopping":     [True],
            "n_iter_no_change":   [10, 20],
            "batch_size":         [32, 64, "auto"],
        },
    }


def get_base_models(scale_pos_weight: float = 1.0) -> dict:
    """Modelos con hiperparámetros fijos (para --sin-gridsearch)."""
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=150, max_depth=20, min_samples_split=2,
            min_samples_leaf=1, class_weight="balanced",
            max_features="sqrt", random_state=SEED, n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5, weights="distance", metric="euclidean", n_jobs=-1,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(100, 50), activation="relu", solver="adam",
            alpha=0.0001, learning_rate_init=0.001, early_stopping=True,
            n_iter_no_change=10, batch_size=64, max_iter=300, random_state=SEED,
        ),
    }
    if XGB_OK:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=5,
            subsample=0.9, colsample_bytree=0.9, gamma=0,
            scale_pos_weight=scale_pos_weight,
            random_state=SEED, n_jobs=-1, eval_metric="logloss",
        )
    return models


def build_cnn(input_shape: tuple):
    """Arquitectura CNN idéntica al paper original."""
    if not TF_OK:
        return None
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        MaxPooling1D(pool_size=2),
        Dropout(0.4),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=AdamOptimizer(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def entrenar_con_gridsearch(nombre, modelo_base, param_grid,
                             X_train, y_train) -> object:
    print(f"\n[{nombre}] GridSearchCV {N_SPLITS_CV}-fold (F1 macro)...")
    t0 = time.time()
    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=SEED)
    gs = GridSearchCV(
        modelo_base, param_grid, cv=cv,
        scoring="f1_macro", n_jobs=-1, verbose=1,
    )
    gs.fit(X_train, y_train)
    print(f"[{nombre}] Mejores params: {gs.best_params_}")
    print(f"[{nombre}] Mejor F1 CV:    {gs.best_score_:.4f}")
    print(f"[{nombre}] Tiempo:         {(time.time()-t0)/60:.1f} min")
    return gs.best_estimator_


def entrenar_sin_gridsearch(nombre, modelo, X_train, y_train) -> object:
    print(f"\n[{nombre}] Entrenando con hiperparámetros fijos...")
    t0 = time.time()
    modelo.fit(X_train, y_train)
    print(f"[{nombre}] Completado en {(time.time()-t0)/60:.1f} min")
    return modelo


# ══════════════════════════════════════════════════════════════════════════════
# EVALUACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def evaluar(nombre, model, X_test, y_test, le, es_keras=False) -> dict:
    class_names = list(le.classes_)

    if es_keras:
        probs = model.predict(X_test, verbose=0).ravel()
        preds = (probs >= 0.5).astype(int)
    else:
        preds = model.predict(X_test)

    acc  = accuracy_score(y_test, preds)
    f1w  = f1_score(y_test, preds, average="weighted", zero_division=0)
    f1m  = f1_score(y_test, preds, average="macro",    zero_division=0)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec  = recall_score(y_test, preds, average="macro",    zero_division=0)

    print(f"\n[{nombre}] accuracy={acc:.4f} | F1_macro={f1m:.4f} | "
          f"F1_weighted={f1w:.4f}")
    print(classification_report(y_test, preds,
                                 target_names=class_names, digits=4))

    return {
        "modelo":           nombre,
        "fuente_datos":     "MADOS",
        "bandas":           FEATURE_COLUMNS,
        "accuracy":         round(acc,  4),
        "f1_macro":         round(f1m,  4),
        "f1_weighted":      round(f1w,  4),
        "precision_macro":  round(prec, 4),
        "recall_macro":     round(rec,  4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(dataset_dir: Path, modelos: list[str], usar_gridsearch: bool) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (X_train, X_test, y_train, y_test,
     scaler, le, sarg_idx, spw) = cargar_datos_mados(dataset_dir)

    # Guardar scaler y label encoder
    dump(scaler, OUTPUT_DIR / "scaler_4b_mados.joblib")
    dump(le,     OUTPUT_DIR / "label_encoder_4b_mados.joblib")
    print(f"\n[guardado] scaler_4b_mados.joblib | label_encoder_4b_mados.joblib")

    # Cargar resultados previos si existen
    out_json = OUTPUT_DIR / "resultados_4b_mados.json"
    resultados_previos = {}
    if out_json.exists():
        try:
            prev = json.loads(out_json.read_text())
            resultados_previos = {r["modelo"]: r for r in prev}
            print(f"[info] Resultados previos: {list(resultados_previos.keys())}")
        except Exception:
            pass

    resultados   = []
    param_grids  = get_param_grids(spw)
    base_models  = get_base_models(spw)

    sklearn_modelos = {
        "randomforest": ("RandomForest",
                         RandomForestClassifier(random_state=SEED, n_jobs=-1)),
        "knn":          ("KNN",
                         KNeighborsClassifier(n_jobs=-1)),
        "mlp":          ("MLP",
                         MLPClassifier(random_state=SEED, max_iter=300)),
    }
    if XGB_OK:
        sklearn_modelos["xgboost"] = (
            "XGBoost",
            xgb.XGBClassifier(random_state=SEED, n_jobs=-1,
                               eval_metric="logloss"),
        )

    for key, (nombre, modelo_base) in sklearn_modelos.items():
        if key not in modelos:
            continue

        if usar_gridsearch:
            model = entrenar_con_gridsearch(
                nombre, modelo_base, param_grids[nombre], X_train, y_train
            )
        else:
            model = entrenar_sin_gridsearch(
                nombre, base_models[nombre], X_train, y_train
            )

        out_path = OUTPUT_DIR / f"{key}_4b_mados.joblib"
        dump(model, out_path)
        print(f"[guardado] {out_path.name}")
        resultados.append(evaluar(nombre, model, X_test, y_test, le))

    # ── 1D-CNN ────────────────────────────────────────────────────────────────
    if "cnn" in modelos and TF_OK:
        print("\n[CNN] Entrenando 1D-CNN...")
        t0 = time.time()
        X_tr_cnn = X_train.reshape(-1, len(FEATURE_COLUMNS), 1)
        X_te_cnn = X_test.reshape(-1,  len(FEATURE_COLUMNS), 1)

        cnn = build_cnn(input_shape=(len(FEATURE_COLUMNS), 1))
        cnn.summary()
        cnn.fit(
            X_tr_cnn, y_train,
            epochs=CNN_EPOCHS, batch_size=CNN_BATCH_SIZE,
            validation_split=CNN_VALIDATION_SPLIT,
            callbacks=[EarlyStopping(
                monitor="val_loss", patience=CNN_PATIENCE,
                restore_best_weights=True, verbose=1,
            )],
            verbose=2,
        )
        print(f"[CNN] Completado en {(time.time()-t0)/60:.1f} min")

        out_path = OUTPUT_DIR / "cnn_4b_mados.keras"
        cnn.save(out_path)
        print(f"[guardado] {out_path.name}")
        resultados.append(evaluar("1D-CNN", cnn, X_te_cnn, y_test, le,
                                   es_keras=True))

    # ── Resumen final ─────────────────────────────────────────────────────────
    for r in resultados:
        resultados_previos[r["modelo"]] = r
    todos = list(resultados_previos.values())
    with open(out_json, "w") as f:
        json.dump(todos, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*65}")
    print(f"  RESUMEN — entrenado con datos MADOS ({', '.join(FEATURE_COLUMNS)})")
    print(f"{'='*65}")
    print(f"  {'Modelo':15}  {'Accuracy':>10}  {'F1 macro':>10}  "
          f"{'Precision':>10}  {'Recall':>8}")
    print(f"  {'-'*60}")
    for r in todos:
        print(f"  {r['modelo']:15}  {r['accuracy']:>10.4f}  "
              f"{r['f1_macro']:>10.4f}  "
              f"{r['precision_macro']:>10.4f}  "
              f"{r['recall_macro']:>8.4f}")
    print(f"\n  Modelos en : {OUTPUT_DIR}")
    print(f"  Métricas en: {out_json}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Ruta por defecto — ajustar si el script se ejecuta desde otro directorio
    DEFAULT_DATASET = (
        Path(__file__).parent.parent.parent.parent.parent
        / "datasets" / "data" / "Sargassum_Ready_Dataset"
    )

    parser = argparse.ArgumentParser(
        description="Reentrenamiento modelos externos con datos MADOS"
    )
    parser.add_argument(
        "--dataset", type=Path,
        default=DEFAULT_DATASET,
        help="Ruta al Sargassum_Ready_Dataset (contiene train/, val/, test/)",
    )
    parser.add_argument(
        "--solo", nargs="+",
        choices=["randomforest", "xgboost", "knn", "mlp", "cnn"],
        default=["randomforest", "xgboost", "knn", "mlp", "cnn"],
        help="Modelos a entrenar (default: todos)",
    )
    parser.add_argument(
        "--sin-gridsearch", action="store_true",
        help="Usar hiperparámetros fijos (mucho más rápido)",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"[ERROR] No se encuentra el dataset en: {args.dataset}")
        print(f"        Usa --dataset para especificar la ruta correcta")
        print(f"        Ejemplo:")
        print(f"          python -m models.external.echevarria2025.retrain_4bands_mados "
              f"--dataset C:/ruta/al/Sargassum_Ready_Dataset")
        exit(1)

    main(args.dataset, args.solo, not args.sin_gridsearch)
