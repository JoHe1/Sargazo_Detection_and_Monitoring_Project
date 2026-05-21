"""
web/streamlit_app/components/map_viewer.py
-------------------------------------------
Construye el mapa Folium interactivo con las detecciones de sargazo
reconstruidas desde los patches de inferencia.

Cada patch detectado se superpone como una capa de imagen georreferenciada
sobre el mapa base de Folium (OpenStreetMap / Stamen Toner).

La reconstrucción respeta las coordenadas geoespaciales calculadas en
sentinel_pipeline.split_into_patches() para cada patch.
"""

from __future__ import annotations

import base64
import io

import folium
import numpy as np
from folium import plugins
from PIL import Image

# ── Colores por clase MADOS ───────────────────────────────────────────
# Solo las clases relevantes para la visualización
COLOR_SARGAZO_DENSO  = (34,  139, 34,  200)  # verde bosque semitransparente
COLOR_SARGAZO_SPARSE = (144, 238, 144, 180)  # verde claro semitransparente
COLOR_AGUA           = (0,   0,   0,   0)    # transparente

CLASE_COLORS = {
    2: COLOR_SARGAZO_DENSO,   # Dense Sargassum
    3: COLOR_SARGAZO_SPARSE,  # Sparse Floating Algae
}


def mask_to_rgba(
    pred_mask:    np.ndarray,
    prob_sarg:    np.ndarray,
    umbral:       float,
    alpha_scale:  float = 0.8,
) -> np.ndarray:
    """
    Convierte la máscara de predicción a imagen RGBA para superponer en el mapa.

    Píxeles con clase 2 o 3 y prob_sarg > umbral se colorean en verde.
    La intensidad del verde refleja la probabilidad de sargazo.
    El resto es completamente transparente.

    Args:
        pred_mask:   (H, W) int   — clase predicha por majority voting
        prob_sarg:   (H, W) float — probabilidad P(sargazo) acumulada
        umbral:      umbral de confianza
        alpha_scale: escala de opacidad máxima [0,1]

    Returns:
        array RGBA (H, W, 4) uint8
    """
    H, W = pred_mask.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)

    # Máscara de sargazo: clase 2 o 3 y por encima del umbral
    sarg_mask = ((pred_mask == 2) | (pred_mask == 3)) & (prob_sarg >= umbral)

    if sarg_mask.any():
        # Color base verde sargazo
        rgba[sarg_mask, 0] = 34   # R
        rgba[sarg_mask, 1] = 180  # G
        rgba[sarg_mask, 2] = 80   # B
        # Alpha proporcional a la probabilidad (más confianza = más opaco)
        prob_norm = np.clip(prob_sarg[sarg_mask], 0, 1)
        rgba[sarg_mask, 3] = (prob_norm * 255 * alpha_scale).astype(np.uint8)

    return rgba


def array_to_base64_png(arr: np.ndarray) -> str:
    """Convierte array RGBA a PNG en base64 para Folium."""
    img = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_map(
    results:     list[dict],
    umbral:      float,
    center:      tuple[float, float] = (18.7, -70.2),
    zoom_start:  int = 8,
) -> folium.Map:
    """
    Construye el mapa Folium con todas las detecciones de sargazo.

    Args:
        results:    lista de dicts con keys:
                    'bounds', 'pred_mask', 'prob_sarg', 'has_sargassum'
        umbral:     umbral de confianza usado en inferencia
        center:     [lat, lon] centro del mapa
        zoom_start: zoom inicial

    Returns:
        objeto folium.Map listo para st_folium()
    """
    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles="CartoDB positron",
        attr="© CartoDB © OpenStreetMap",
    )

    # Capa base satélite (opcional)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satélite",
        overlay=False,
        control=True,
    ).add_to(m)

    # Grupo de capas para detecciones
    sarg_group  = folium.FeatureGroup(name="Detecciones de sargazo", show=True)
    patch_group = folium.FeatureGroup(name="Cuadrícula de patches", show=False)

    n_detecciones = 0

    for r in results:
        bounds     = r["bounds"]        # [lon_min, lat_min, lon_max, lat_max]
        pred_mask  = r["pred_mask"]     # (224, 224) int
        prob_sarg  = r["prob_sarg"]     # (224, 224) float
        has_sarg   = r.get("has_sargassum", False)

        # Bounds para Folium: [[lat_min, lon_min], [lat_max, lon_max]]
        folium_bounds = [
            [bounds[1], bounds[0]],
            [bounds[3], bounds[2]],
        ]

        # Cuadrícula de patches (siempre visible si la capa está activa)
        folium.Rectangle(
            bounds=folium_bounds,
            color="#444",
            weight=0.3,
            fill=False,
        ).add_to(patch_group)

        if not has_sarg:
            continue

        # Overlay RGBA del patch con sargazo
        rgba = mask_to_rgba(pred_mask, prob_sarg, umbral)
        if rgba[:, :, 3].max() == 0:
            continue  # transparente completo — nada que mostrar

        png_b64 = array_to_base64_png(rgba)
        img_url = f"data:image/png;base64,{png_b64}"

        folium.raster_layers.ImageOverlay(
            image=img_url,
            bounds=folium_bounds,
            opacity=1.0,
            interactive=False,
            cross_origin=False,
            zindex=10,
        ).add_to(sarg_group)

        # Tooltip sobre el patch
        tp = (pred_mask == 2).sum() + (pred_mask == 3).sum()
        folium.Rectangle(
            bounds=folium_bounds,
            color="#22cc66",
            weight=1,
            fill=False,
            tooltip=folium.Tooltip(
                f"<b>Sargazo detectado</b><br>"
                f"Píxeles: {tp}<br>"
                f"Prob. máx: {prob_sarg.max():.2f}<br>"
                f"Umbral: {umbral}"
            ),
        ).add_to(sarg_group)

        n_detecciones += 1

    sarg_group.add_to(m)
    patch_group.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # Leyenda
    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:12px 16px;border-radius:6px;
                border:1px solid #ccc;font-family:monospace;font-size:12px;
                box-shadow:2px 2px 6px rgba(0,0,0,.15)">
      <b>Leyenda</b><br>
      <span style="color:#22cc66">■</span> Sargazo detectado<br>
      <span style="color:#888">□</span> Sin sargazo<br>
      <hr style="margin:6px 0">
      Umbral: <b>{umbral}</b><br>
      Patches con sargazo: <b>{n_detecciones}</b>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m, n_detecciones