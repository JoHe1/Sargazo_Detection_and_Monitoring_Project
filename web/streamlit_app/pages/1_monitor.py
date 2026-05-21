"""
web/streamlit_app/pages/1_monitor.py
--------------------------------------
Monitorización NRT de sargazo en Cabo Engaño (Costa Este, Rep. Dom.)

Flujo:
    1. Usuario introduce credenciales Copernicus
    2. Pulsa "Ver imágenes disponibles" → lista fechas con baja nubosidad
    3. Elige fecha del desplegable
    4. Pulsa "Detectar sargazo"
    5. Se descarga imagen 448×448 (4 patches), se infiere y se muestra:
       - Métricas resumen
       - Mapa Folium con detecciones superpuestas
       - Paneles RGB + NIR + Probabilidad + Detección por patch
    6. Al cerrar la sesión todo se borra — nada en disco
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch

# ── Path setup ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COMPONENTS = Path(__file__).parent.parent / "components"
if str(COMPONENTS) not in sys.path:
    sys.path.insert(0, str(COMPONENTS))

from model_loader import load_model, DEVICE
from sentinel_pipeline import (
    run_pipeline, list_available_products, get_token,
    BBOX, BBOX_LABEL,
)
from map_viewer import build_map

try:
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

NUM_CLASSES = 16


# ══════════════════════════════════════════════════════════════════════
# TTA — idéntico al paper MADOS (Kikaki et al. 2024)
# ══════════════════════════════════════════════════════════════════════

def inferir_tta(model, img_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    8 transformaciones (rot0/90/180/270 × flip/no-flip) + majority voting.
    Igual que MariNeXt y el inference.py principal del proyecto.
    """
    transformaciones = [
        (0, False), (0, True),
        (1, False), (1, True),
        (2, False), (2, True),
        (3, False), (3, True),
    ]
    tensor = torch.tensor(
        np.transpose(img_array, (2, 0, 1)), dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    acum  = np.zeros((NUM_CLASSES, 224, 224), dtype=np.float32)
    votos = np.zeros((NUM_CLASSES, 224, 224), dtype=np.int32)

    with torch.no_grad():
        for num_rot, flip_h in transformaciones:
            t = tensor.clone()
            if flip_h:
                t = torch.flip(t, dims=[3])
            if num_rot > 0:
                t = torch.rot90(t, k=num_rot, dims=[2, 3])
            probs = torch.softmax(model(t), dim=1).squeeze(0).cpu().numpy()
            if num_rot > 0:
                probs = np.rot90(probs, k=-num_rot, axes=(1, 2)).copy()
            if flip_h:
                probs = probs[:, :, ::-1].copy()
            acum += probs
            for c in range(NUM_CLASSES):
                votos[c] += (probs.argmax(axis=0) == c).astype(np.int32)

    prob_todas     = acum / len(transformaciones)
    prob_sargassum = prob_todas[2] + prob_todas[3]
    clase_predicha = votos.argmax(axis=0).astype(np.int32)
    return clase_predicha, prob_sargassum


# ══════════════════════════════════════════════════════════════════════
# INSPECCIÓN VISUAL
# ══════════════════════════════════════════════════════════════════════

def mostrar_inspeccion(results: list, umbral: float) -> None:
    """Muestra RGB, NIR, P(sargazo) y detección para cada patch."""
    import matplotlib.pyplot as plt

    st.subheader("🔬 Inspección visual de patches")
    st.caption(
        "**RGB** — imagen Sentinel-2 con corrección Rayleigh  ·  "
        "**NIR** — infrarrojo cercano (sargazo brilla, agua es oscura)  ·  "
        "**P(sargazo)** — probabilidad del modelo  ·  "
        "**Detección** — píxeles clasificados como sargazo"
    )

    for r in results:
        img_s     = r["image"]
        pred_mask = r["pred_mask"]
        prob_sarg = r["prob_sarg"]
        det       = ((pred_mask == 2) | (pred_mask == 3)) & (prob_sarg >= umbral)
        n_px      = int(det.sum())

        rgb     = np.clip(img_s[:, :, :3], 0, 1)
        nir     = img_s[:, :, 3]
        overlay = rgb.copy()
        overlay[det, 0] = 0.1
        overlay[det, 1] = 0.9
        overlay[det, 2] = 0.1

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.patch.set_facecolor("#0e1117")
        estado = "✅ Sargazo detectado" if r["has_sargassum"] else "❌ Sin sargazo"
        fig.suptitle(
            f"Patch ({r['row']}, {r['col']}) — {estado}  |  "
            f"bounds: {[round(b, 2) for b in r['bounds']]}",
            color="white", fontsize=9,
        )
        for ax in axes:
            ax.axis("off")
            ax.set_facecolor("#0e1117")

        axes[0].imshow(rgb)
        axes[0].set_title("RGB Sentinel-2\n(corrección Rayleigh ≈ACOLITE)", color="white", fontsize=8)

        axes[1].imshow(nir, cmap="inferno", vmin=0, vmax=1)
        axes[1].set_title("Canal NIR\n(sargazo = brillante, agua = oscura)", color="white", fontsize=8)

        im = axes[2].imshow(prob_sarg, cmap="YlGn", vmin=0, vmax=1)
        axes[2].set_title("P(sargazo)\nP(cl.2) + P(cl.3)", color="white", fontsize=8)
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        axes[3].imshow(overlay)
        axes[3].set_title(
            f"Detección (u={umbral:.2f})\n{n_px} px · {100*n_px/(224*224):.1f}%",
            color="white", fontsize=8,
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Tabla resumen
    sarg = [r for r in results if r["has_sargassum"]]
    if sarg:
        st.subheader("📊 Resumen de detecciones")
        tabla = []
        for r in sarg:
            tp = int(((r["pred_mask"] == 2) | (r["pred_mask"] == 3)).sum())
            tabla.append({
                "Patch":     f"({r['row']}, {r['col']})",
                "Píxeles":   tp,
                "Cobertura": f"{100*tp/(224*224):.1f}%",
                "Prob. máx": f"{r['prob_sarg'].max():.3f}",
            })
        st.dataframe(tabla, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PÁGINA
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Monitor Sargazo · Cabo Engaño",
    page_icon="📡",
    layout="wide",
)

st.title("📡 Monitorización NRT — Cabo Engaño")
st.markdown(
    f"**Zona:** {BBOX_LABEL}  \n"
    f"**Bbox:** `{BBOX}`  \n"
    "Detección de sargazo sobre imagen Sentinel-2 descargada al momento. "
    "Los datos se procesan en memoria y no se guardan en disco."
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuración")

    # Paso 1 — credenciales
    st.markdown("**Paso 1 — Credenciales Copernicus**")
    client_id     = st.text_input("CLIENT_ID",     type="password")
    client_secret = st.text_input("CLIENT_SECRET", type="password")

    buscar = st.button(
        "🔎 Ver imágenes disponibles",
        disabled=not (client_id and client_secret),
        use_container_width=True,
    )
    if not (client_id and client_secret):
        st.caption("⚠️ Introduce las credenciales primero")

    # Buscar fechas disponibles
    if buscar:
        with st.spinner("Consultando Copernicus…"):
            try:
                token_tmp = get_token(client_id, client_secret)
                productos = list_available_products(token_tmp, days_back=90, max_cloud=30)
                if productos:
                    st.session_state["productos"] = productos
                    st.success(f"✅ {len(productos)} imágenes encontradas")
                else:
                    # Relajar nubosidad si no hay con <30%
                    productos = list_available_products(token_tmp, days_back=90, max_cloud=60)
                    if productos:
                        st.session_state["productos"] = productos
                        st.warning(f"⚠️ No hay imágenes con <30% nubes. Mostrando {len(productos)} con <60%.")
                    else:
                        st.warning("No se encontraron imágenes en los últimos 90 días.")
                        st.session_state["productos"] = []
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state["productos"] = []

    st.divider()

    # Paso 2 — elegir imagen
    st.markdown("**Paso 2 — Elegir imagen**")
    productos = st.session_state.get("productos", [])
    selected  = None

    if productos:
        opciones = {f"{p['date']}  ☁️ {p['cloud']}": p for p in productos}
        sel      = st.selectbox(
            "Imágenes disponibles:",
            options=list(opciones.keys()),
            help="Más reciente primero. ☁️ = nubosidad.",
        )
        selected = opciones[sel]
        try:
            if float(selected["cloud"].replace("%", "")) > 30:
                st.warning("☁️ Nubosidad alta — posibles falsos positivos.")
        except (ValueError, TypeError):
            pass
    else:
        st.caption("Pulsa 'Ver imágenes disponibles' para cargar las opciones.")

    st.divider()

    # Parámetros
    st.markdown("**Parámetros de detección**")
    umbral   = st.slider("Umbral de confianza", 0.50, 0.99, 0.95, 0.01,
                         help="0.95 es el umbral óptimo del modelo (F1=0.650 en MADOS).")
    usar_tta = st.checkbox("TTA (8 transformaciones)", value=True,
                           help="Majority voting sobre 8 augmentaciones — igual que MariNeXt.")

    ejecutar = st.button(
        "🔍 Detectar sargazo",
        type="primary",
        disabled=selected is None,
        use_container_width=True,
    )
    if selected is None:
        st.caption("Selecciona una imagen para continuar.")

    selected_date  = selected["date"]  if selected else None
    selected_cloud = selected["cloud"] if selected else "—"

# ── Área principal ────────────────────────────────────────────────────

# Restaurar resultados si Streamlit recargó la página
if not ejecutar and "results" in st.session_state:
    st.info("Mostrando resultados de la última detección.", icon="🔄")
    _r  = st.session_state["results"]
    _pi = st.session_state["product_info"]
    _u  = st.session_state["umbral_usado"]
    _ns = sum(1 for r in _r if r["has_sargassum"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Patches analizados", len(_r))
    col2.metric("Con sargazo", _ns)
    col3.metric("Cobertura", f"{100*_ns/max(len(_r),1):.1f}%")
    col4.metric("Umbral", f"{_u:.2f}")

    with st.expander("📄 Producto Sentinel-2"): st.json(_pi)

    st.subheader("🗺️ Mapa de detecciones")
    if HAS_FOLIUM:
        _m, _ = build_map(_r, _u, center=[18.45, -68.25], zoom_start=10)
        st_folium(_m, height=500, use_container_width=True)

    mostrar_inspeccion(_r, _u)

elif not ejecutar:
    # Estado inicial — mapa vacío
    st.info(
        "Introduce tus credenciales, elige una fecha y pulsa **Detectar sargazo**.",
        icon="ℹ️",
    )
    if HAS_FOLIUM:
        import folium
        m = folium.Map(location=[18.45, -68.25], zoom_start=10, tiles="CartoDB positron")
        folium.Rectangle(
            bounds=[[BBOX[1], BBOX[0]], [BBOX[3], BBOX[2]]],
            color="#22cc66", weight=2, fill=True, fill_opacity=0.07,
            tooltip=f"Zona: {BBOX_LABEL}",
        ).add_to(m)
        # Marcador Cabo Engaño
        folium.Marker(
            location=[18.62, -68.32],
            tooltip="Cabo Engaño",
            icon=folium.Icon(color="green", icon="leaf"),
        ).add_to(m)
        st_folium(m, height=450, use_container_width=True)

else:
    # ── Ejecución ─────────────────────────────────────────────────────
    status = st.empty()
    prog   = st.progress(0, text="Iniciando…")
    steps  = []

    def log_step(msg: str):
        steps.append(msg)
        status.markdown("\n\n".join(f"- {s}" for s in steps))

    try:
        log_step("Cargando modelo…")
        prog.progress(10, text="Cargando modelo…")
        model = load_model()

        prog.progress(20, text="Descargando imagen…")
        patches, product_info = run_pipeline(
            client_id=client_id,
            client_secret=client_secret,
            date_str=selected_date,
            cloud_pct=selected_cloud,
            progress_cb=log_step,
        )

        prog.progress(60, text=f"Infiriendo {len(patches)} patches…")
        log_step(f"Ejecutando inferencia ({len(patches)} patches)…")

        if len(patches) == 0:
            st.warning(
                "La imagen descargada no tiene datos válidos — "
                "posiblemente cubierta por nubes o fuera del área de cobertura del satélite. "
                "Prueba con otra fecha."
            )
            st.stop()

        results = []
        for i, patch in enumerate(patches):
            img_arr = patch["image"]

            # ── Canales: orden MADOS (B=0, G=1, R=2, NIR=3) ──────────
            nir_ch = img_arr[:, :, 3]
            r_ch   = img_arr[:, :, 2]

            # ── Máscara 1: píxeles sin dato ────────────────────────────
            # EvalScript devuelve [0,0,0,0] para nubes y no-data
            pixel_sum   = img_arr.sum(axis=2)
            nodata_mask = (pixel_sum == 0.0)

            # ── Máscara 2: tierra firme ────────────────────────────────
            # a) NDVI alto → vegetación terrestre (sargazo marino NDVI ~0.1-0.25)
            # b) NIR > 0.35 → suelo / tierra brillante (agua oceánica NIR ~0.03-0.10)
            # c) pixel_sum > 2.8 → nubes residuales muy brillantes
            eps  = 1e-6
            ndvi = (nir_ch - r_ch) / (nir_ch + r_ch + eps)

            tierra_mask = (
                (ndvi > 0.30)
                | (nir_ch > 0.35)
                | (pixel_sum > 2.8)
                | nodata_mask
            )

            if usar_tta:
                pred_mask, prob_sarg = inferir_tta(model, img_arr)
            else:
                tensor = torch.tensor(
                    np.transpose(img_arr, (2, 0, 1)), dtype=torch.float32
                ).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    probs     = torch.softmax(model(tensor), dim=1).squeeze(0).cpu().numpy()
                pred_mask = probs.argmax(axis=0).astype(np.int32)
                prob_sarg = probs[2] + probs[3]

            # Aplicar land mask + no-data: forzar a clase agua (7), prob = 0
            prob_sarg[tierra_mask] = 0.0
            pred_mask[tierra_mask] = 7  # clase 7 = Deep Water en MADOS

            has_sarg = bool(
                (((pred_mask == 2) | (pred_mask == 3)) & (prob_sarg >= umbral)).any()
            )

            results.append({
                "bounds":        patch["bounds"],
                "row":           patch["row"],
                "col":           patch["col"],
                "image":         patch["image"],
                "pred_mask":     pred_mask,
                "prob_sarg":     prob_sarg,
                "has_sargassum": has_sarg,
            })
            prog.progress(60 + int(35 * (i+1) / len(patches)))

        # Guardar en session_state para sobrevivir recargas
        st.session_state["results"]      = results
        st.session_state["product_info"] = product_info
        st.session_state["umbral_usado"] = umbral

        n_sarg = sum(1 for r in results if r["has_sargassum"])
        prog.progress(100, text="¡Completado!")
        log_step(f"✅ {n_sarg} patches con sargazo de {len(results)} analizados.")

        # ── Métricas ──────────────────────────────────────────────────
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Patches analizados", len(results))
        col2.metric("Con sargazo", n_sarg)
        col3.metric("Cobertura", f"{100*n_sarg/max(len(results),1):.1f}%")
        col4.metric("Umbral", f"{umbral:.2f}")

        with st.expander("📄 Información del producto"):
            st.json(product_info)

        # ── Mapa ──────────────────────────────────────────────────────
        st.subheader("🗺️ Mapa de detecciones — Cabo Engaño")
        if HAS_FOLIUM:
            mapa, _ = build_map(results, umbral, center=[18.45, -68.25], zoom_start=10)
            st_folium(mapa, height=500, use_container_width=True)

        # ── Inspección visual ─────────────────────────────────────────
        mostrar_inspeccion(results, umbral)

    except Exception as e:
        prog.empty()
        st.error(f"❌ Error: `{e}`")
        st.markdown(
            "**Posibles causas:**\n"
            "- Credenciales incorrectas\n"
            "- Imagen sin datos válidos para esa fecha\n"
            "- Conexión a internet no disponible\n\n"
            "Prueba con otra fecha o verifica las credenciales."
        )