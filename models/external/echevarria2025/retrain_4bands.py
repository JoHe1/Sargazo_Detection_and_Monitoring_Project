"""
models/external/echevarria2025/retrain_4bands.py
-------------------------------------------------
Adaptación del script original de Echevarría et al. (2025) para usar
4 bandas espectrales (Blue, Green, Red, NIR) en lugar de 5 (+ SWIR1).

El único cambio respecto al original es:
    FEATURE_COLUMNS = ["blue", "green", "red", "nir"]  # sin swir1

Todo lo demás es idéntico: mismo GridSearchCV, mismos hiperparámetros,
mismo split 70/30 estratificado, misma CNN, mismo random_state=42.

Justificación: comparativa homogénea con el Swin Transformer del TFG,
que opera sobre las mismas 4 bandas disponibles en la descarga Sentinel-2.

Referencia:
    Echevarría-Rubio, J.M.; Martínez-Flores, G.; Morales-Pérez, R.A.
    Data 2025, 10, 177. https://doi.org/10.3390/data10110177
    Código original: https://doi.org/10.5281/zenodo.17246345

Uso:
    # Todos los modelos (tarda ~2-3h por el GridSearchCV de RF y KNN)
    python -m models.external.echevarria2025.retrain_4bands

    # Solo los modelos rápidos para verificar
    python -m models.external.echevarria2025.retrain_4bands --solo xgboost mlp cnn

    # Saltarse GridSearchCV y usar hiperparámetros fijos (mucho más rápido)
    python -m models.external.echevarria2025.retrain_4bands --sin-gridsearch

Salida (en models/external/echevarria2025/models_4bands/):
    randomforest_4b.joblib
    xgboost_4b.joblib
    knn_4b.joblib
    mlp_4b.joblib
    cnn_4b.keras
    scaler_4b.joblib
    label_encoder_4b.joblib
    resultados_4b.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, train_test_split,
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

# ── Configuración ─────────────────────────────────────────────────────────────
SEED             = 42
TEST_SIZE        = 0.30
N_SPLITS_CV      = 5
CLASS_COLUMN     = "class"
POSITIVE_LABEL   = "sargassum"

# ← ÚNICO CAMBIO RESPECTO AL ORIGINAL: sin swir1
# Orden crítico: debe coincidir con el orden de bandas en inferencia
# Original: ['Blue', 'Green', 'Red', 'NIR', 'SWIR1']
# Este TFG: ['Blue', 'Green', 'Red', 'NIR']
FEATURE_COLUMNS = ["Blue", "Green", "Red", "NIR"]

CNN_EPOCHS             = 100
CNN_BATCH_SIZE         = 64
CNN_VALIDATION_SPLIT   = 0.20
CNN_PATIENCE           = 15

SCRIPT_DIR  = Path(__file__).parent
OUTPUT_DIR  = SCRIPT_DIR / "models_4bands"
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# GRIDS DE HIPERPARÁMETROS — idénticos al paper original
# ══════════════════════════════════════════════════════════════════════════════

def get_param_grids(scale_pos_weight: float = 1.0) -> dict:
    return {
        "RandomForest": {
            "n_estimators":    [100, 150],
            "max_depth":       [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf":  [1, 3],
            "class_weight":    ["balanced", "balanced_subsample", None],
            "max_features":    ["sqrt", "log2"],
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 9],
            "weights":     ["uniform", "distance"],
            "metric":      ["euclidean", "manhattan", "minkowski"],
        },
        "XGBoost": {
            "n_estimators":    [100, 150],
            "learning_rate":   [0.05, 0.1],
            "max_depth":       [3, 5, 7],
            "subsample":       [0.7, 0.9, 1.0],
            "colsample_bytree":[0.7, 0.9, 1.0],
            "gamma":           [0, 0.1, 0.2],
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
# CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def cargar_datos(csv_path: Path):
    print(f"\n[datos] Cargando: {csv_path}")
    df = pd.read_csv(csv_path)

    for col in FEATURE_COLUMNS + [CLASS_COLUMN]:
        if col not in df.columns:
            raise ValueError(
                f"Columna '{col}' no encontrada.\n"
                f"Columnas disponibles: {list(df.columns)}"
            )

    print(f"[datos] Muestras totales: {len(df):,}")
    print(f"[datos] Bandas usadas: {FEATURE_COLUMNS}")
    if "swir1" in df.columns:
        print(f"[datos] swir1 presente en el CSV pero EXCLUIDA de las features")
    print(f"[datos] Distribución:\n{df[CLASS_COLUMN].value_counts().to_string()}")

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y_raw = df[CLASS_COLUMN].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    sarg_idx = list(le.classes_).index(POSITIVE_LABEL)
    print(f"[datos] '{POSITIVE_LABEL}' codificado como: {sarg_idx}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print(f"[datos] Train: {len(X_train):,} | Test: {len(X_test):,}")

    scale_pos_weight = np.sum(y_train != sarg_idx) / max(np.sum(y_train == sarg_idx), 1)
    print(f"[datos] scale_pos_weight para XGBoost: {scale_pos_weight:.2f}")

    return X_train_s, X_test_s, y_train, y_test, scaler, le, sarg_idx, scale_pos_weight


# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO CON GRIDSEARCH
# ══════════════════════════════════════════════════════════════════════════════

def entrenar_con_gridsearch(nombre: str, modelo_base, param_grid: dict,
                             X_train, y_train) -> object:
    print(f"\n[{nombre}] GridSearchCV {N_SPLITS_CV}-fold (F1 weighted)...")
    t0 = time.time()
    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=SEED)
    gs = GridSearchCV(
        modelo_base, param_grid, cv=cv,
        scoring="f1_macro", n_jobs=-1, verbose=1,  # igual que el paper original
    )
    gs.fit(X_train, y_train)
    print(f"[{nombre}] Mejores params: {gs.best_params_}")
    print(f"[{nombre}] Mejor F1 CV: {gs.best_score_:.4f}")
    print(f"[{nombre}] Tiempo: {(time.time()-t0)/60:.1f} min")
    return gs.best_estimator_


def entrenar_sin_gridsearch(nombre: str, modelo, X_train, y_train) -> object:
    print(f"\n[{nombre}] Entrenando con hiperparámetros fijos...")
    t0 = time.time()
    modelo.fit(X_train, y_train)
    print(f"[{nombre}] Completado en {(time.time()-t0)/60:.1f} min")
    return modelo


# ══════════════════════════════════════════════════════════════════════════════
# EVALUACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def evaluar(nombre: str, model, X_test, y_test,
            le: LabelEncoder, es_keras=False) -> dict:
    class_names = list(le.classes_)

    if es_keras:
        probs = model.predict(X_test, verbose=0).ravel()
        preds = (probs >= 0.5).astype(int)
    else:
        preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1w = f1_score(y_test, preds, average="weighted", zero_division=0)
    f1m = f1_score(y_test, preds, average="macro",    zero_division=0)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec  = recall_score(y_test, preds, average="macro",    zero_division=0)

    print(f"\n[{nombre}] accuracy={acc:.4f} | F1_macro={f1m:.4f} | F1_weighted={f1w:.4f}")
    print(classification_report(y_test, preds, target_names=class_names, digits=4))

    return {
        "modelo":      nombre,
        "bandas":      FEATURE_COLUMNS,
        "accuracy":    round(acc,  4),
        "f1_macro":    round(f1m,  4),
        "f1_weighted": round(f1w,  4),
        "precision_macro": round(prec, 4),
        "recall_macro":    round(rec,  4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(csv_path: Path, modelos: list[str], usar_gridsearch: bool) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, scaler, le, sarg_idx, spw = cargar_datos(csv_path)

    # Guardar scaler y label encoder
    dump(scaler, OUTPUT_DIR / "scaler_4b.joblib")
    dump(le,     OUTPUT_DIR / "label_encoder_4b.joblib")
    print(f"\n[guardado] scaler_4b.joblib | label_encoder_4b.joblib")

    resultados = []
    param_grids  = get_param_grids(spw)
    base_models  = get_base_models(spw)

    sklearn_modelos = {
        "randomforest": ("RandomForest", RandomForestClassifier(random_state=SEED, n_jobs=-1)),
        "knn":          ("KNN",          KNeighborsClassifier(n_jobs=-1)),
        "mlp":          ("MLP",          MLPClassifier(random_state=SEED, max_iter=300)),
    }
    if XGB_OK:
        sklearn_modelos["xgboost"] = (
            "XGBoost",
            xgb.XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="logloss"),
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

        out_path = OUTPUT_DIR / f"{key}_4b.joblib"
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

        out_path = OUTPUT_DIR / "cnn_4b.keras"
        cnn.save(out_path)
        print(f"[guardado] {out_path.name}")
        resultados.append(evaluar("1D-CNN", cnn, X_te_cnn, y_test, le, es_keras=True))

    # ── Resumen final ─────────────────────────────────────────────────────────
    out_json = OUTPUT_DIR / "resultados_4b.json"
    with open(out_json, "w") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*65}")
    print(f"  RESUMEN — reentrenamiento con {len(FEATURE_COLUMNS)} bandas ({', '.join(FEATURE_COLUMNS)})")
    print(f"{'='*65}")
    print(f"  {'Modelo':15}  {'Accuracy':>10}  {'F1 macro':>10}  {'F1 weighted':>12}")
    print(f"  {'-'*55}")
    for r in resultados:
        print(f"  {r['modelo']:15}  {r['accuracy']:>10.4f}  "
              f"{r['f1_macro']:>10.4f}  {r['f1_weighted']:>12.4f}")
    print(f"\n  Modelos en : {OUTPUT_DIR}")
    print(f"  Métricas en: {out_json}")
    print(f"\n  Para usar en inferencia:")
    print(f"    from joblib import load")
    print(f"    scaler = load('{OUTPUT_DIR}/scaler_4b.joblib')")
    print(f"    model  = load('{OUTPUT_DIR}/xgboost_4b.joblib')")
    print(f"    probs  = model.predict_proba(scaler.transform(X))[:, {sarg_idx}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reentrenamiento 4 bandas — Echevarría et al. (2025)"
    )
    parser.add_argument(
        "--csv", type=Path,
        default=SCRIPT_DIR / "sargassum_data.csv",
        help="Ruta al sargassum_data.csv",
    )
    parser.add_argument(
        "--solo", nargs="+",
        choices=["randomforest", "xgboost", "knn", "mlp", "cnn"],
        default=["randomforest", "xgboost", "knn", "mlp", "cnn"],
        help="Modelos a entrenar (default: todos)",
    )
    parser.add_argument(
        "--sin-gridsearch", action="store_true",
        help="Usar hiperparámetros fijos en lugar de GridSearchCV (mucho más rápido)",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"[ERROR] No se encuentra: {args.csv}")
        print(f"        Descárgalo de: https://doi.org/10.5281/zenodo.17246345")
        exit(1)

    main(args.csv, args.solo, not args.sin_gridsearch)