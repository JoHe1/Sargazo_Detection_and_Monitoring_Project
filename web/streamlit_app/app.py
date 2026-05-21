"""
web/streamlit_app/app.py
-------------------------
Punto de entrada principal de la aplicación Streamlit.

Sistema de Monitorización NRT de Sargazo
TFG — Detección y Monitorización de Sargazo mediante Imágenes Satelitales
ULPGC · Grado en Ciencia e Ingeniería de Datos · 2026

La app se estructura en 3 páginas accesibles desde la barra lateral:
    1_monitor.py   — Monitorización NRT (página principal)
    2_inference.py — Inferencia sobre imagen cargada manualmente
    3_compare.py   — Comparativa de umbrales

Uso:
    streamlit run web/streamlit_app/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Monitorización de Sargazo · TFG ULPGC",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**Detección y Monitorización de Sargazo**\n\n"
            "TFG · Grado en Ciencia e Ingeniería de Datos · ULPGC 2026\n\n"
            "Autor: Jorge Lorenzo Lorenzo\n"
            "Tutores: Javier Sánchez Pérez · Giovanny A. Cuervo Londoño"
        )
    },
)

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Sargazo NRT")
    st.markdown(
        "Sistema de detección automática de sargazo mediante "
        "imágenes Sentinel-2 y aprendizaje profundo."
    )
    st.divider()
    st.markdown("**Navegación**")
    st.page_link("pages/1_monitor.py",   label="📡 Monitorización NRT",  icon="📡")
    st.page_link("pages/2_inference.py", label="🔍 Inferencia manual",   icon="🔍")
    st.page_link("pages/3_compare.py",   label="📊 Comparativa umbrales",icon="📊")
    st.divider()
    st.markdown(
        "<small>TFG · ULPGC · 2026<br>"
        "Modelo: Swin-Tiny + Attention U-Net<br>"
        "Dataset: MADOS · Sentinel-2 L2A</small>",
        unsafe_allow_html=True,
    )

# ── Página de inicio ──────────────────────────────────────────────────
st.title("🌿 Sistema de Monitorización de Sargazo")
st.markdown(
    """
    La creciente llegada masiva de sargazo a las costas del Atlántico representa una
    **problemática ambiental y económica** para el turismo y los ecosistemas costeros.
    La monitorización actual, a menudo manual o por avistamiento, requiere una modernización
    que permita **alertas tempranas fiables** y un análisis en **tiempo casi real (NRT)**.
    """
)

col1, col2, col3 = st.columns(3)
with col1:
    st.info(
        "**📡 Monitorización NRT**\n\n"
        "Descarga automática de la imagen Sentinel-2 más reciente "
        "sobre República Dominicana e inferencia inmediata.",
        icon="📡",
    )
with col2:
    st.info(
        "**🔍 Inferencia manual**\n\n"
        "Carga tu propia imagen preprocesada (.npy) "
        "y visualiza las detecciones del modelo.",
        icon="🔍",
    )
with col3:
    st.info(
        "**📊 Comparativa de umbrales**\n\n"
        "Compara el efecto de diferentes umbrales de confianza "
        "sobre la misma imagen.",
        icon="📊",
    )

st.divider()
st.markdown(
    """
    #### Modelo utilizado
    **Swin Transformer Tiny + Attention U-Net** · 31.9M parámetros  
    Entrenado sobre MADOS · 4 canales espectrales (RGB + NIR)  
    F1 = 0.650 · IoU sargazo = 0.482 · Umbral óptimo = 0.95 · TTA 8 transformaciones

    #### Región monitorizada
    **República Dominicana — Costa Norte, Mar Caribe**  
    Zona seleccionada por alta densidad documentada de sargazo pelágico (2018–2024).
    """
)