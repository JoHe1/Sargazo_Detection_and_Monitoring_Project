"""
web/streamlit_app/pages/3_compare.py
--------------------------------------
Página de comparativa de umbrales.

Muestra side-by-side el efecto de 3 umbrales diferentes sobre la misma imagen,
útil para explicar en la defensa del TFG cómo el umbral controla el equilibrio
entre precisión y recall en la detección de sargazo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COMPONENTS = Path(__file__).parent.parent / "components"
if str(COMPONENTS) not in sys.path:
    sys.path.insert(0, str(COMPONENTS))

from model_loader import load_model, DEVICE

NUM_CLASSES = 16

st.set_page_config(page_title="Comparativa Umbrales · Sargazo", page_icon="📊", layout="wide")
st.title("📊 Comparativa de umbrales de confianza")
st.markdown(
    "Visualiza cómo el umbral de confianza afecta a las detecciones. "
    "Umbral alto → menos FP, más FN. Umbral bajo → más recall, menos precisión."
)
st.divider()

# ── Carga ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Sube un archivo .npy (224×224×4)", type=["npy"])
if uploaded is None:
    st.info("Sube un archivo .npy para comparar umbrales.", icon="📂")
    st.stop()

try:
    img = np.load(uploaded).astype(np.float32)
except Exception as e:
    st.error(f"Error al cargar: {e}")
    st.stop()

if img.ndim == 3 and img.shape[2] == 4:
    img = img[:, :, [2, 1, 0, 3]]
    if img.max() > 10.0:
        img = img / 10000.0
    img = np.clip(img * 5.0, 0.0, 1.0)
    H, W = img.shape[:2]
    y0 = (H - 224) // 2
    x0 = (W - 224) // 2
    img = img[y0:y0+224, x0:x0+224, :]
else:
    st.error(f"Formato inesperado: {img.shape}")
    st.stop()

# ── Umbrales a comparar ───────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
u1 = col1.slider("Umbral A", 0.50, 0.99, 0.70, 0.01, key="u1")
u2 = col2.slider("Umbral B", 0.50, 0.99, 0.90, 0.01, key="u2")
u3 = col3.slider("Umbral C (óptimo)", 0.50, 0.99, 0.95, 0.01, key="u3")

# ── Inferencia ────────────────────────────────────────────────────────
with st.spinner("Ejecutando inferencia con TTA…"):
    model = load_model()
    tensor = torch.tensor(
        np.transpose(img, (2, 0, 1)), dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    transformaciones = [
        (0, False), (0, True), (1, False), (1, True),
        (2, False), (2, True), (3, False), (3, True),
    ]
    acum = np.zeros((NUM_CLASSES, 224, 224), dtype=np.float32)
    votos = np.zeros((NUM_CLASSES, 224, 224), dtype=np.int32)
    with torch.no_grad():
        for num_rot, flip_h in transformaciones:
            t = tensor.clone()
            if flip_h: t = torch.flip(t, dims=[3])
            if num_rot > 0: t = torch.rot90(t, k=num_rot, dims=[2, 3])
            probs = torch.softmax(model(t), dim=1).squeeze(0).cpu().numpy()
            if num_rot > 0: probs = np.rot90(probs, k=-num_rot, axes=(1,2)).copy()
            if flip_h: probs = probs[:, :, ::-1].copy()
            acum += probs
            for c in range(NUM_CLASSES):
                votos[c] += (probs.argmax(axis=0) == c).astype(np.int32)

    prob_todas = acum / len(transformaciones)
    prob_sarg  = prob_todas[2] + prob_todas[3]

# ── Visualización ─────────────────────────────────────────────────────
rgb = np.clip(img[:, :, :3], 0, 1)

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
fig.patch.set_facecolor("#0e1117")
for ax in axes:
    ax.set_facecolor("#0e1117")
    ax.axis("off")

# RGB original
axes[0].imshow(rgb)
axes[0].set_title("RGB original", color="white", fontsize=10)

for ax, umbral, label in zip(
    axes[1:],
    [u1, u2, u3],
    [f"Umbral A ({u1:.2f})", f"Umbral B ({u2:.2f})", f"Umbral C óptimo ({u3:.2f})"],
):
    det = ((prob_sarg >= umbral))
    n_px = int(det.sum())
    overlay = rgb.copy()
    overlay[det, 0] = 0.1
    overlay[det, 1] = 0.9
    overlay[det, 2] = 0.2
    ax.imshow(overlay)
    ax.set_title(f"{label}\n{n_px} px ({100*n_px/(224*224):.1f}%)",
                 color="white", fontsize=9)

plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── Tabla comparativa ─────────────────────────────────────────────────
st.subheader("Tabla comparativa")
tabla = []
for umbral, nombre in [(u1, "Umbral A"), (u2, "Umbral B"), (u3, "Umbral C (óptimo)")]:
    det = (prob_sarg >= umbral)
    n   = int(det.sum())
    tabla.append({
        "Umbral":          f"{umbral:.2f}",
        "Nombre":          nombre,
        "Píxeles detectados": n,
        "Cobertura (%)":   f"{100*n/(224*224):.2f}",
        "Prob. máxima":    f"{prob_sarg.max():.4f}",
        "Prob. media (det.)": f"{prob_sarg[det].mean():.4f}" if n > 0 else "—",
    })
st.dataframe(tabla, use_container_width=True)

st.markdown(
    """
    ---
    **Interpretación:**
    - **Umbral bajo (0.70)** → detecta más sargazo pero con más falsos positivos (agua, nubes)
    - **Umbral alto (0.95)** → solo activa la alarma con alta confianza, menos ruido, pero puede perder sargazo disperso
    - **Umbral óptimo (0.95)** → encontrado experimentalmente maximizando F1 en el conjunto de test del dataset MADOS
    """
)