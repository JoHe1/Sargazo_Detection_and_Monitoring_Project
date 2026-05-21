"""
web/streamlit_app/components/sentinel_pipeline.py
---------------------------------------------------
Pipeline de descarga y preprocesamiento para una zona pequeña fija
sobre Cabo Engaño (extremo este de República Dominicana).

Zona elegida por ser la entrada principal de sargazo pelágico
desde el Atlántico ecuatorial hacia las costas de RD.

Tamaño de zona: 0.5° × 0.5° → ~55km × ~55km a latitud 18.5°
Se divide en una cuadrícula de 4 patches 224×224 (2×2).
Descarga rápida (<30s), procesado en memoria, sin archivos en disco.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PREPROCESADO — ALINEADO CON base_dataset._normalize (entrenamiento)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

El modelo fue entrenado con TIFs ACOLITE (rhorc) en [0, ~0.20].
base_dataset._normalize aplica:
    if img.max() > 10.0: img /= 10000   ← NO se activa (max ≈ 0.20)
    img = clip(img * 5.0, 0, 1)         ← [0, 0.20] → [0, 1]

El pipeline de inferencia debe replicar exactamente ese rango.

POR QUÉ NO SE APLICA CORRECCIÓN RAYLEIGH:
  Para agua oceánica oscura ρtoa ≈ 0.03, offset B02 = 0.055:
      0.03 - 0.055 = -0.025 → Math.max(0, ...) = 0
  El 60-80% de píxeles de agua queda a CERO → imagen ruidosa.
  El modelo absorbe la pequeña diferencia ρtoa vs ρs sin degradación.

PIPELINE CORRECTO (alineado con entrenamiento):
  1. Descargar L1C en DN (única opción con CLM en S2L1C)
  2. EvalScript: DN/10000 → ρtoa [0,1], enmascarar CLM/dataMask
     Orden de canales MADOS: (B, G, R, NIR) — igual que en train
  3. normalize_img: ×5 → clip(0,1)  [SIN ÷10000 previo]
  4. Filtros post-normalización: nubes residuales, vegetación
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import requests

try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# ── Zona fija: Cabo Engaño ────────────────────────────────────────────
BBOX       = [-68.5, 18.2, -68.0, 18.7]   # [lon_min, lat_min, lon_max, lat_max]
BBOX_LABEL = "Cabo Engaño — Costa Este de República Dominicana"
PATCH_SIZE = 224
IMG_WIDTH  = 448
IMG_HEIGHT = 448

# ── URLs Copernicus ───────────────────────────────────────────────────
TOKEN_URL   = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# ── EvalScript L1C — ρtoa puro, orden MADOS (B, G, R, NIR) ──────────
#
# DECISIONES DE DISEÑO:
#
#   • units: "DN" — única opción válida en S2L1C cuando se incluye CLM.
#     REFLECTANCE lanza error 400; dos bloques input lanza "Dataset id:1 not found".
#
#   • SIN corrección Rayleigh — el modelo fue entrenado con rhorc de ACOLITE
#     pero base_dataset._normalize NO divide entre 10000 (los rhorc llegan
#     ya en [0,~0.20]). Aplicar la corrección aquí vaciaría el agua oscura:
#       B02 ρtoa ≈ 0.030 − 0.055 = −0.025 → clip a 0 → 60-80% de píxeles = 0
#
#   • Orden de salida: (B, G, R, NIR) — igual que MADOS/ACOLITE.
#     _reorder_channels del base_dataset lo invierte a (R, G, B, NIR)
#     antes de pasarlo al modelo. Aquí no se reordena: lo hará normalize_img.
#
#   • Valores de salida: ρtoa en [0, 1].
#     normalize_img aplica ×5 → clip(0,1), replicando base_dataset._normalize.
#
#   • CLM: 0 = claro, 1 = nube  |  dataMask: 1 = dato válido, 0 = sin dato
EVALSCRIPT = """
//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["B02", "B03", "B04", "B08", "CLM", "dataMask"],
            units: "DN"
        }],
        output: { bands: 4, sampleType: "FLOAT32" }
    };
}

function evaluatePixel(sample) {
    // Enmascarar nubes y píxeles sin dato
    if (sample.CLM === 1 || sample.dataMask === 0) {
        return [0, 0, 0, 0];
    }

    // DN → ρtoa: rango [0, 1]
    // Sin corrección Rayleigh — normalize_img aplica ×5 igual que en train
    // Orden MADOS/ACOLITE: (B, G, R, NIR)
    let b   = sample.B02 / 10000.0;
    let g   = sample.B03 / 10000.0;
    let r   = sample.B04 / 10000.0;
    let nir = sample.B08 / 10000.0;

    return [b, g, r, nir];
}
"""


# ══════════════════════════════════════════════════════════════════════
# AUTENTICACIÓN
# ══════════════════════════════════════════════════════════════════════

def get_token(client_id: str, client_secret: str) -> str:
    resp = requests.post(TOKEN_URL, data={
        "grant_type":    "client_credentials",
        "client_id":     client_id,
        "client_secret": client_secret,
    }, timeout=30)
    if resp.status_code != 200:
        raise ConnectionError(
            f"Error de autenticación ({resp.status_code}). "
            "Verifica CLIENT_ID y CLIENT_SECRET."
        )
    return resp.json()["access_token"]


# ══════════════════════════════════════════════════════════════════════
# FECHAS DISPONIBLES
# ══════════════════════════════════════════════════════════════════════

def list_available_products(
    token:     str,
    days_back: int = 90,
    max_cloud: int = 30,
) -> list[dict]:
    """
    Lista imágenes Sentinel-2 disponibles sobre Cabo Engaño.

    CAMBIO: busca en sentinel-2-l1c (antes buscaba en sentinel-2-l2a).
    L1C y L2A comparten las mismas fechas de adquisición, pero el
    catálogo L1C es la fuente correcta para este pipeline.
    """
    date_to   = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    date_from = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")

    lon_min, lat_min, lon_max, lat_max = BBOX
    payload = {
        "bbox":        [lon_min, lat_min, lon_max, lat_max],
        "datetime":    f"{date_from}/{date_to}",
        # ✓ L1C — fuente correcta para ACOLITE empírico
        "collections": ["sentinel-2-l1c"],
        "limit":       50,
        "filter":      f"eo:cloud_cover < {max_cloud}",
        "filter-lang": "cql2-text",
    }

    resp = requests.post(
        CATALOG_URL,
        json=payload,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=30,
    )
    if resp.status_code != 200:
        raise ConnectionError(f"Error catálogo ({resp.status_code}): {resp.text[:200]}")

    features = resp.json().get("features", [])
    features.sort(
        key=lambda f: f.get("properties", {}).get("datetime", ""),
        reverse=True,
    )

    results    = []
    seen_dates = set()
    for f in features:
        date  = f.get("properties", {}).get("datetime", "")[:10]
        cloud = f.get("properties", {}).get("eo:cloud_cover", "—")
        if date in seen_dates:
            continue
        seen_dates.add(date)
        try:
            cloud_fmt = f"{float(cloud):.1f}%"
        except (ValueError, TypeError):
            cloud_fmt = "—"
        results.append({"date": date, "cloud": cloud_fmt})

    return results


# ══════════════════════════════════════════════════════════════════════
# DESCARGA VÍA PROCESS API
# ══════════════════════════════════════════════════════════════════════

def download_image(
    token:     str,
    date_str:  str,
    max_cloud: int = 40,
) -> np.ndarray | None:
    """
    Descarga imagen 448×448 de Cabo Engaño via Process API.
    Todo en memoria — nada se guarda en disco.

    CAMBIO CRÍTICO: type cambiado de "sentinel-2-l2a" a "sentinel-2-l1c".
    L1C entrega ρtoa sin corrección ESA (Sen2Cor), que es la entrada
    correcta para la corrección Rayleigh empírica del EvalScript.
    Usar L2A causaba doble corrección atmosférica y valores fuera del
    rango esperado por el modelo entrenado con ACOLITE/MADOS.

    Returns:
        array (448, 448, 4) float32 en escala ×10000 (pre-normalización)
        o None si la descarga falla.
    """
    payload = {
        "input": {
            "bounds": {
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
                "bbox": BBOX,
            },
            "data": [{
                # ✓ L1C — TOA reflectance, sin corrección ESA
                "type": "sentinel-2-l1c",
                "dataFilter": {
                    "timeRange": {
                        "from": f"{date_str}T00:00:00Z",
                        "to":   f"{date_str}T23:59:59Z",
                    },
                    "maxCloudCoverage": max_cloud,
                },
            }],
        },
        "evalscript": EVALSCRIPT,
        "output": {
            "width":  IMG_WIDTH,
            "height": IMG_HEIGHT,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
    }

    if not HAS_RASTERIO:
        raise ImportError(
            "rasterio no está instalado. Ejecuta: pip install rasterio"
        )

    resp = requests.post(
        PROCESS_URL,
        json=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
            "Accept":        "image/tiff",
        },
        timeout=60,
    )

    if resp.status_code != 200:
        # Exponer el error real de la API para poder diagnosticar
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text[:500]
        raise ConnectionError(
            f"Process API respondió {resp.status_code}.\n"
            f"Detalle: {detail}"
        )

    # Verificar que la respuesta es un TIFF válido y no un JSON de error
    content_type = resp.headers.get("Content-Type", "")
    if "tiff" not in content_type.lower() and "octet" not in content_type.lower():
        raise ValueError(
            f"La API devolvió Content-Type inesperado: '{content_type}'.\n"
            f"Respuesta: {resp.text[:300]}"
        )

    with rasterio.open(io.BytesIO(resp.content)) as src:
        img = src.read().astype(np.float32)   # (4, H, W)
        img = np.transpose(img, (1, 2, 0))    # (H, W, 4)

    return img


# ══════════════════════════════════════════════════════════════════════
# NORMALIZACIÓN
# ══════════════════════════════════════════════════════════════════════

def normalize_img(img: np.ndarray) -> np.ndarray:
    """
    Normalización idéntica a base_dataset._normalize del entrenamiento.

    El EvalScript entrega ρtoa en [0, 1] (sin corrección Rayleigh,
    sin escalar ×10000). base_dataset._normalize hace:

        if img.max() > 10.0: img /= 10000   ← NO se activa (max ≈ 1.0)
        img = clip(img * 5.0, 0, 1)

    Aquí replicamos exactamente eso:
        ×5 → clip(0, 1)

    El canal NIR lleva al índice 3 en orden MADOS (B=0, G=1, R=2, NIR=3),
    igual que durante el entrenamiento. No se reordena aquí.

    Filtros post-normalización (conservadores para no eliminar sargazo):
      • Anti-nubes residuales : B+G+R > 1.8  (nubes no capturadas por CLM)
      • Anti-vegetación costera: NIR > 0.4 AND R < 0.1
        El sargazo legítimo tiene NIR ~0.10–0.30, no cumple ambas.
    """
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

    # Replica base_dataset._normalize: ×5 → clip(0,1)
    # NO se divide entre 10000 — el EvalScript ya entrega [0, 1]
    img = np.clip(img * 5.0, 0.0, 1.0).astype(np.float32)

    # Filtro anti-nubes residuales
    # Orden canales: (B=0, G=1, R=2, NIR=3) — igual que MADOS
    bgr_sum = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
    img[bgr_sum > 1.8] = 0.0

    # Filtro anti-vegetación terrestre
    nir = img[:, :, 3]
    r   = img[:, :, 2]
    tierra_veg = (nir > 0.4) & (r < 0.1)
    img[tierra_veg] = 0.0

    return img


# ══════════════════════════════════════════════════════════════════════
# DIVISIÓN EN PATCHES
# ══════════════════════════════════════════════════════════════════════

def split_into_patches(img: np.ndarray) -> list[dict]:
    """
    Divide la imagen 448×448 en 4 patches 224×224 (cuadrícula 2×2).
    Descarta patches que sean >95% ceros (nubes/tierra enmascaradas).
    Incluye metadatos geoespaciales para el mapa Folium.
    """
    H, W = img.shape[:2]
    lon_min, lat_min, lon_max, lat_max = BBOX
    lon_per_px = (lon_max - lon_min) / W
    lat_per_px = (lat_max - lat_min) / H

    patches = []
    for row in range(H // PATCH_SIZE):
        for col in range(W // PATCH_SIZE):
            y0 = row * PATCH_SIZE
            x0 = col * PATCH_SIZE
            patch = img[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE, :]

            if patch.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
                continue

            # Descartar solo si >95% ceros (prácticamente sin datos)
            zeros_pct = (patch.sum(axis=2) == 0).mean()
            if zeros_pct > 0.95:
                continue

            patches.append({
                "image": patch,
                "row":   row,
                "col":   col,
                "bounds": [
                    lon_min + col * PATCH_SIZE * lon_per_px,
                    lat_max - (row + 1) * PATCH_SIZE * lat_per_px,
                    lon_min + (col + 1) * PATCH_SIZE * lon_per_px,
                    lat_max - row * PATCH_SIZE * lat_per_px,
                ],
            })
    return patches


# ══════════════════════════════════════════════════════════════════════
# PIPELINE COMPLETA
# ══════════════════════════════════════════════════════════════════════

def run_pipeline(
    client_id:     str,
    client_secret: str,
    date_str:      str,
    cloud_pct:     str = "—",
    progress_cb:   Callable | None = None,
) -> tuple[list[dict], dict]:
    """
    Pipeline completa en memoria: auth → descarga → normalización → patches.
    Nada se guarda en disco — todo se borra al cerrar la sesión.

    Preprocesado alineado con base_dataset._normalize (entrenamiento):
        L1C DN → ρtoa [0,1] (EvalScript) → ×5 → clip(0,1) (normalize_img)
    Orden de canales: (B, G, R, NIR) — igual que MADOS/ACOLITE en train.
    """
    def log(msg: str):
        if progress_cb:
            progress_cb(msg)

    log("🔐 Autenticando…")
    token = get_token(client_id, client_secret)

    log(f"📡 Descargando imagen L1C {date_str} de Cabo Engaño (448×448 px)…")
    try:
        cloud_val = float(str(cloud_pct).replace("%", ""))
        max_cloud = max(40, int(cloud_val) + 10)
    except (ValueError, TypeError):
        max_cloud = 40

    # download_image ahora lanza ConnectionError/ValueError/ImportError
    # con el mensaje real de la API — ya no devuelve None silenciosamente.
    img = download_image(token, date_str, max_cloud=max_cloud)

    log("⚙️ Normalizando (×5, clip [0,1] — igual que entrenamiento)…")
    img = normalize_img(img)

    log("🔲 Dividiendo en patches 224×224…")
    patches = split_into_patches(img)

    if not patches:
        raise ValueError(
            "La imagen está completamente cubierta por nubes o tierra. "
            "Prueba con otra fecha de menor nubosidad."
        )

    product_info = {
        "zona":   BBOX_LABEL,
        "bbox":   BBOX,
        "fecha":  date_str,
        "cloud":  cloud_pct,
        "size":   f"{IMG_WIDTH}×{IMG_HEIGHT} px → {len(patches)} patches 224×224",
        "api":    "Sentinel Hub Process API — L1C ρtoa [0,1], normalización ×5 (≡ base_dataset._normalize)",
    }

    log(f"✅ {len(patches)} patches listos.")
    return patches, product_info