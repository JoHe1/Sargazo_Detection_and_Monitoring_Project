"""
web/streamlit_app/pages/2_inference.py
----------------------------------------
Página de inferencia manual. Acepta dos tipos de entrada:

  1. Archivo .SAFE.zip  — producto Sentinel-2 L1C descargado desde
                          Copernicus Browser. Se extraen las bandas
                          B02/B03/B04/B08, se aplica el mismo preprocesado
                          que sentinel_pipeline.py y se infiere sobre todos
                          los patches oceánicos de 224×224 que contiene.

  2. Archivo .npy       — patch ya preprocesado (legacy / dataset MADOS).

En ambos casos el preprocesado es idéntico al entrenamiento:
    DN / 10000  →  orden (B, G, R, NIR)  →  ×5  →  clip(0, 1)
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COMPONENTS = Path(__file__).parent.parent / "components"
if str(COMPONENTS) not in sys.path:
    sys.path.insert(0, str(COMPONENTS))

from model_loader import load_model, DEVICE

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

NUM_CLASSES = 16
PATCH_SIZE  = 224


# ══════════════════════════════════════════════════════════════════════
# LECTURA DEL .SAFE ZIP
# ══════════════════════════════════════════════════════════════════════

def _find_band(zf: zipfile.ZipFile, band: str) -> str | None:
    """
    Busca el archivo .jp2 de una banda dentro del ZIP.

    Formato real del .SAFE de Copernicus:
        GRANULE/.../IMG_DATA/T19QEA_20250708T151741_B02.jp2

    La banda B08 en L1C está a 10m igual que B02/B03/B04.
    Se excluyen QI_DATA (máscaras de calidad) para no confundirlos.
    """
    names = zf.namelist()
    for name in names:
        p = PurePosixPath(name)
        if (
            "IMG_DATA" in p.parts          # solo bandas de imagen
            and "QI_DATA" not in p.parts   # excluir máscaras de calidad
            and p.stem.endswith(f"_{band}") # nombre acaba en _B02, _B08, etc.
            and p.suffix == ".jp2"
        ):
            return name
    return None


def load_safe_zip(uploaded_file) -> list[dict]:
    """
    Lee un .SAFE.zip de Sentinel-2 L1C y devuelve una lista de patches
    224×224 preprocesados (misma lógica que sentinel_pipeline.py).

    Returns:
        lista de dicts con keys: 'image' (224,224,4), 'row', 'col'
    """
    if not HAS_RASTERIO:
        st.error("rasterio no está instalado. Ejecuta: `pip install rasterio`")
        st.stop()

    raw = uploaded_file.read()
    zf  = zipfile.ZipFile(io.BytesIO(raw))

    # Localizar las 4 bandas
    band_keys = {"B02": None, "B03": None, "B04": None, "B08": None}
    for bk in band_keys:
        path = _find_band(zf, bk)
        if path is None:
            st.error(f"No se encontró la banda {bk} dentro del ZIP.")
            st.stop()
        band_keys[bk] = path

    # Leer cada banda con rasterio desde memoria
    arrays = {}
    for bk, zpath in band_keys.items():
        with zf.open(zpath) as f:
            data = f.read()
        with rasterio.open(io.BytesIO(data)) as src:
            arrays[bk] = src.read(1).astype(np.float32)  # (H, W) en DN

    # Todas deben tener la misma resolución — B08 en L1C ya es 10m
    # Si alguna es diferente, resamplear a la de B04
    H, W = arrays["B04"].shape
    for bk in ("B02", "B03", "B08"):
        if arrays[bk].shape != (H, W):
            from skimage.transform import resize
            arrays[bk] = resize(
                arrays[bk], (H, W),
                order=1, preserve_range=True, anti_aliasing=True
            ).astype(np.float32)

    # DN → ρtoa, orden MADOS (B, G, R, NIR), normalización ×5
    b   = arrays["B02"] / 10000.0
    g   = arrays["B03"] / 10000.0
    r   = arrays["B04"] / 10000.0
    nir = arrays["B08"] / 10000.0

    # (H, W, 4) en orden (B, G, R, NIR) — igual que MADOS
    full_img = np.stack([b, g, r, nir], axis=-1)
    full_img = np.nan_to_num(full_img, nan=0.0, posinf=1.0, neginf=0.0)
    full_img = np.clip(full_img * 5.0, 0.0, 1.0).astype(np.float32)

    # Dividir en patches 224×224 descartando los que sean >90% ceros
    patches = []
    for row in range(H // PATCH_SIZE):
        for col in range(W // PATCH_SIZE):
            y0    = row * PATCH_SIZE
            x0    = col * PATCH_SIZE
            patch = full_img[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE, :]
            if patch.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
                continue
            zeros_pct = (patch.sum(axis=2) == 0).mean()
            if zeros_pct > 0.90:
                continue
            patches.append({"image": patch, "row": row, "col": col})

    return patches


# ══════════════════════════════════════════════════════════════════════
# PREPROCESADO .npy (legacy — dataset MADOS)
# ══════════════════════════════════════════════════════════════════════

def load_npy(uploaded_file) -> list[dict]:
    """Carga un .npy de MADOS y devuelve un único patch preprocesado."""
    img = np.load(uploaded_file).astype(np.float32)
    if img.ndim != 3 or img.shape[2] != 4:
        st.error(f"Formato inesperado: shape={img.shape}. Se esperaba (H, W, 4).")
        st.stop()
    # Orden MADOS (B,G,R,NIR) → ya correcto; normalizar igual que base_dataset
    if img.max() > 10.0:
        img = img / 10000.0
    img = np.clip(img * 5.0, 0.0, 1.0)
    # Center crop 224×224
    H, W = img.shape[:2]
    y0 = (H - PATCH_SIZE) // 2
    x0 = (W - PATCH_SIZE) // 2
    img = img[y0:y0+PATCH_SIZE, x0:x0+PATCH_SIZE, :]
    return [{"image": img, "row": 0, "col": 0}]


# ══════════════════════════════════════════════════════════════════════
# INFERENCIA TTA
# ══════════════════════════════════════════════════════════════════════

def inferir_tta(model, img_array: np.ndarray, usar_tta: bool) -> tuple[np.ndarray, np.ndarray]:
    """Inferencia con TTA opcional (8 transformaciones + majority voting)."""
    tensor = torch.tensor(
        np.transpose(img_array, (2, 0, 1)), dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    if not usar_tta:
        with torch.no_grad():
            prob_todas = torch.softmax(model(tensor), dim=1).squeeze(0).cpu().numpy()
        return prob_todas.argmax(axis=0).astype(np.int32), prob_todas[2] + prob_todas[3]

    transformaciones = [
        (0, False), (0, True), (1, False), (1, True),
        (2, False), (2, True), (3, False), (3, True),
    ]
    acum  = np.zeros((NUM_CLASSES, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    votos = np.zeros((NUM_CLASSES, PATCH_SIZE, PATCH_SIZE), dtype=np.int32)

    with torch.no_grad():
        for num_rot, flip_h in transformaciones:
            t = tensor.clone()
            if flip_h:   t = torch.flip(t, dims=[3])
            if num_rot > 0: t = torch.rot90(t, k=num_rot, dims=[2, 3])
            probs = torch.softmax(model(t), dim=1).squeeze(0).cpu().numpy()
            if num_rot > 0: probs = np.rot90(probs, k=-num_rot, axes=(1, 2)).copy()
            if flip_h:      probs = probs[:, :, ::-1].copy()
            acum += probs
            for c in range(NUM_CLASSES):
                votos[c] += (probs.argmax(axis=0) == c).astype(np.int32)

    prob_todas = acum / len(transformaciones)
    return votos.argmax(axis=0).astype(np.int32), prob_todas[2] + prob_todas[3]


# ══════════════════════════════════════════════════════════════════════
# LAND MASK — idéntica a 1_monitor.py
# ══════════════════════════════════════════════════════════════════════

def aplicar_land_mask(img: np.ndarray, pred_mask: np.ndarray, prob_sarg: np.ndarray):
    """
    Enmascara tierra, nubes y no-data.
    Canales: (B=0, G=1, R=2, NIR=3) — orden MADOS.
    """
    nir_ch    = img[:, :, 3]
    r_ch      = img[:, :, 2]
    pixel_sum = img.sum(axis=2)
    eps       = 1e-6
    ndvi      = (nir_ch - r_ch) / (nir_ch + r_ch + eps)

    tierra_mask = (
        (ndvi > 0.30)
        | (nir_ch > 0.35)
        | (pixel_sum > 2.8)
        | (pixel_sum == 0.0)
    )
    prob_sarg[tierra_mask] = 0.0
    pred_mask[tierra_mask] = 7   # clase 7 = Deep Water en MADOS
    return pred_mask, prob_sarg


# ══════════════════════════════════════════════════════════════════════
# VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════

def mostrar_patch(img: np.ndarray, pred_mask: np.ndarray,
                  prob_sarg: np.ndarray, umbral: float,
                  titulo: str = "") -> None:
    deteccion = ((pred_mask == 2) | (pred_mask == 3)) & (prob_sarg >= umbral)
    n_px      = int(deteccion.sum())

    # RGB: canales (B=0,G=1,R=2) → reordenar a (R,G,B) para imshow
    rgb     = np.clip(img[:, :, [2, 1, 0]], 0, 1)
    nir     = img[:, :, 3]
    overlay = rgb.copy()
    overlay[deteccion, 0] = 0.1
    overlay[deteccion, 1] = 0.9
    overlay[deteccion, 2] = 0.2

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.patch.set_facecolor("#0e1117")
    if titulo:
        fig.suptitle(titulo, color="white", fontsize=9)
    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#0e1117")

    axes[0].imshow(rgb)
    axes[0].set_title("RGB Sentinel-2", color="white", fontsize=9)

    axes[1].imshow(nir, cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title("Canal NIR\n(sargazo=brillante, agua=oscura)", color="white", fontsize=9)

    im = axes[2].imshow(prob_sarg, cmap="YlGn", vmin=0, vmax=1)
    axes[2].set_title("P(sargazo)\nP(cl.2)+P(cl.3)", color="white", fontsize=9)
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(overlay)
    axes[3].set_title(
        f"Detección (u={umbral:.2f})\n{n_px} px · {100*n_px/(PATCH_SIZE**2):.1f}%",
        color="white", fontsize=9,
    )
    patch_leg = mpatches.Patch(color=(0.1, 0.9, 0.2), label=f"Sargazo ({n_px} px)")
    axes[3].legend(handles=[patch_leg], loc="lower right", fontsize=8,
                   facecolor="#1a1a2e", labelcolor="white")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# PÁGINA
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Inferencia Manual · Sargazo", page_icon="🔍", layout="wide")
st.title("🔍 Inferencia sobre imagen cargada")
st.markdown(
    "Acepta dos formatos:\n"
    "- **`.SAFE.zip`** — producto Sentinel-2 L1C descargado desde Copernicus Browser "
    "(se extraen y procesan las bandas automáticamente)\n"
    "- **`.npy`** — patch ya preprocesado del dataset MADOS"
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parámetros")
    umbral   = st.slider("Umbral de confianza", 0.50, 0.99, 0.95, 0.01)
    usar_tta = st.checkbox("TTA (8 transformaciones)", value=True)
    st.divider()
    st.markdown("**Formatos aceptados**")
    st.caption(
        "`.SAFE.zip` → Sentinel-2 L1C de Copernicus Browser\n\n"
        "`.npy` → Patches preprocesados del dataset MADOS"
    )

# ── Carga ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Sube un archivo `.SAFE.zip` (Sentinel-2 L1C) o `.npy` (MADOS)",
    type=["zip", "npy"],
    help="Producto L1C de Copernicus Browser o array numpy preprocesado",
)

if uploaded is None:
    st.info("Sube un archivo para iniciar la inferencia.", icon="📂")
    st.stop()

# ── Procesar según tipo ───────────────────────────────────────────────
fname = uploaded.name.lower()

with st.spinner("Leyendo y preprocesando imagen…"):
    if fname.endswith(".zip") or fname.endswith(".safe.zip"):
        patches = load_safe_zip(uploaded)
        st.success(f"Producto .SAFE cargado — {len(patches)} patches válidos de 224×224")
    elif fname.endswith(".npy"):
        patches = load_npy(uploaded)
        st.success(f"Array .npy cargado — {patches[0]['image'].shape}")
    else:
        st.error("Formato no reconocido. Sube un .zip (producto .SAFE) o un .npy")
        st.stop()

if not patches:
    st.warning("No se encontraron patches válidos en la imagen. Puede estar cubierta de nubes.")
    st.stop()

# ── Inferencia ────────────────────────────────────────────────────────
model = load_model()

results = []
prog    = st.progress(0, text="Infiriendo patches…")

for i, patch in enumerate(patches):
    img_arr   = patch["image"]
    pred_mask, prob_sarg = inferir_tta(model, img_arr, usar_tta)
    pred_mask, prob_sarg = aplicar_land_mask(img_arr, pred_mask, prob_sarg)

    has_sarg = bool(
        (((pred_mask == 2) | (pred_mask == 3)) & (prob_sarg >= umbral)).any()
    )
    results.append({
        **patch,
        "pred_mask": pred_mask,
        "prob_sarg": prob_sarg,
        "has_sarg":  has_sarg,
    })
    prog.progress(int(100 * (i + 1) / len(patches)))

prog.empty()

# ── Métricas globales ─────────────────────────────────────────────────
n_sarg   = sum(1 for r in results if r["has_sarg"])
total_px = sum(
    int(((r["pred_mask"] == 2) | (r["pred_mask"] == 3)) & (r["prob_sarg"] >= umbral)).sum()  # type: ignore
    for r in results
)

st.divider()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Patches analizados",  len(results))
col2.metric("Patches con sargazo", n_sarg)
col3.metric("Píxeles sargazo",     total_px)
col4.metric("Cobertura media",
            f"{100*total_px/(len(results)*PATCH_SIZE**2):.2f}%")

# ── Visualización ─────────────────────────────────────────────────────
st.subheader("🔬 Resultados por patch")

# Primero los patches con sargazo, luego el resto
ordenados = sorted(results, key=lambda r: not r["has_sarg"])

for r in ordenados:
    estado = "✅ Sargazo detectado" if r["has_sarg"] else "❌ Sin sargazo"
    titulo = f"Patch ({r['row']}, {r['col']}) — {estado}"
    mostrar_patch(r["image"], r["pred_mask"], r["prob_sarg"], umbral, titulo)

# ── Tabla resumen ─────────────────────────────────────────────────────
sarg_results = [r for r in results if r["has_sarg"]]
if sarg_results:
    st.subheader("📊 Resumen de detecciones")
    tabla = []
    for r in sarg_results:
        n = int(((r["pred_mask"] == 2) | (r["pred_mask"] == 3)).sum())
        tabla.append({
            "Patch":       f"({r['row']}, {r['col']})",
            "Píxeles":     n,
            "Cobertura":   f"{100*n/PATCH_SIZE**2:.1f}%",
            "Prob. máx":   f"{r['prob_sarg'].max():.3f}",
        })
    st.dataframe(tabla, use_container_width=True)