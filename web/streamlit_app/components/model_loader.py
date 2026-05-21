"""
web/streamlit_app/components/model_loader.py
---------------------------------------------
Carga el mejor modelo entrenado (Swin-Tiny + Attention U-Net)
desde los pesos guardados en web/streamlit_app/model/weights.pth.

Usa st.cache_resource para cargar el modelo una sola vez en memoria
y reutilizarlo en todas las inferencias sin recargar pesos cada vez.

El modelo se ejecuta en CPU por defecto para compatibilidad con
Streamlit Cloud y entornos sin GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn

# ── Añadir raíz del proyecto al path para importar arquitecturas ─────
# Estructura esperada:
#   proyecto/
#   ├── models/architectures/swin_transformer_attention.py
#   ├── core/interfaces/base_model.py
#   └── web/streamlit_app/components/model_loader.py
ROOT = Path(__file__).resolve().parents[3]  # sube 3 niveles → raíz proyecto
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.registry import ModelRegistry

# ── Ruta a los pesos ─────────────────────────────────────────────────
WEIGHTS_PATH = Path(__file__).parent.parent / "model" / "weights.pth"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES  = 16


@st.cache_resource(show_spinner="Cargando modelo…")
def load_model() -> nn.Module:
    """
    Carga el Swin-Tiny Attention U-Net con los pesos del mejor experimento.
    Cacheado por Streamlit — solo se ejecuta una vez por sesión.

    Returns:
        modelo en modo eval() listo para inferencia
    """
    if not WEIGHTS_PATH.exists():
        st.error(
            f"No se encontraron los pesos del modelo en:\n`{WEIGHTS_PATH}`\n\n"
            "Copia `weights.pth` del mejor run a `web/streamlit_app/model/`"
        )
        st.stop()

    model = ModelRegistry.build("swin_transformer_attention", num_classes=NUM_CLASSES)
    state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model