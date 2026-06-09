"""
datasets/preprocessors/mados/analisis_distribucion.py
-------------------------------------------------------
Análisis exploratorio completo del dataset MADOS.
Cubre DOS etapas del pipeline:

  ETAPA 0 — Datos CRUDOS (TIFs de ACOLITE en MADOS_RAW_DIR)
    · Escenas disponibles, recortes por split según los .txt
    · Tamaños reales de los TIFs (H×W px)
    · Resumen pre-procesado: cuántos recortes hay, de qué tamaño

  ETAPA 1 — Datos PROCESADOS (arrays .npy en SARGASSUM_READY)
    · Tiles guardados por split tras MADOSPreprocessor
    · Distribución de píxeles por clase (todas + zoom minoritarias)
    · Desequilibrio de clases (pie + log-bar)
    · Tiles con sargazo vs. sin sargazo por split

Figuras generadas (fondo blanco, aptas para TFG):
    0_resumen_raw_vs_procesado.png   — tabla comparativa crudo / procesado
    1_distribucion_tiles_por_split.png
    2_distribucion_pixeles_por_clase_{split}.png
    3_desequilibrio_clases_{split}.png
    4_tiles_con_sargazo_por_split.png

Uso:
    python -m datasets.preprocessors.mados.analisis_distribucion
    python -m datasets.preprocessors.mados.analisis_distribucion --splits train val
    python -m datasets.preprocessors.mados.analisis_distribucion --out figures/
    python -m datasets.preprocessors.mados.analisis_distribucion --solo procesado
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from core.config.paths import MADOS_RAW_DIR, SARGASSUM_READY

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════

DATASET_DIR  = str(SARGASSUM_READY)
MADOS_RAW    = str(MADOS_RAW_DIR)
NUM_CLASSES  = 16
CLASES_SARG  = {2, 3}
TARGET       = 224   # tamaño del crop tras preprocesado

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#444444",
    "axes.labelcolor":   "#222222",
    "xtick.color":       "#222222",
    "ytick.color":       "#222222",
    "text.color":        "#222222",
    "grid.color":        "#cccccc",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "figure.dpi":        150,
})

CLASES = {
    0:  ("Non-annotated",            "#aaaaaa"),
    1:  ("Marine Debris",            "#e74c3c"),
    2:  ("Dense Sargassum",          "#1a7a3c"),
    3:  ("Sparse Floating Algae",    "#52c47a"),
    4:  ("Natural Organic Material", "#e67e22"),
    5:  ("Ship",                     "#7d3c98"),
    6:  ("Oil Spill",                "#2c3e50"),
    7:  ("Marine Water",             "#2980b9"),
    8:  ("Sediment-Laden Water",     "#c9a227"),
    9:  ("Foam",                     "#bdc3c7"),
    10: ("Turbid Water",             "#5dade2"),
    11: ("Shallow Water",            "#85c1e9"),
    12: ("Waves & Wakes",            "#76d7c4"),
    13: ("Oil Platform",             "#922b21"),
    14: ("Jellyfish",                "#f1948a"),
    15: ("Sea Snot",                 "#9a7d0a"),
}


# ══════════════════════════════════════════════════════════════
# UTILIDADES COMUNES
# ══════════════════════════════════════════════════════════════

def listar_mascaras(split: str) -> list[str]:
    mask_dir = os.path.join(DATASET_DIR, split, "masks")
    return sorted(glob.glob(os.path.join(mask_dir, "*.npy")))


def tiene_sargassum(mask: np.ndarray) -> bool:
    return bool(np.isin(mask, list(CLASES_SARG)).any())


def contar_pixeles_por_clase(mask_paths: list[str]) -> np.ndarray:
    conteo = np.zeros(NUM_CLASSES, dtype=np.int64)
    for mp in tqdm(mask_paths, desc="    Contando píxeles", leave=False):
        m = np.load(mp).astype(np.int32)
        m = np.clip(m, 0, NUM_CLASSES - 1)
        for c in range(NUM_CLASSES):
            conteo[c] += int((m == c).sum())
    return conteo


def contar_tiles(mask_paths: list[str]) -> tuple[int, int]:
    total    = len(mask_paths)
    con_sarg = 0
    for mp in tqdm(mask_paths, desc="    Contando tiles", leave=False):
        m = np.load(mp)
        if tiene_sargassum(m):
            con_sarg += 1
    return total, con_sarg


# ══════════════════════════════════════════════════════════════
# ETAPA 0 — INSPECCIÓN DE DATOS CRUDOS (MADOS RAW)
# ══════════════════════════════════════════════════════════════

def _leer_split_txt(split: str) -> list[str]:
    """Lee el .txt de un split y devuelve la lista de nombres de recorte."""
    ruta = os.path.join(MADOS_RAW, "splits", f"{split}_X.txt")
    if not os.path.exists(ruta):
        return []
    with open(ruta) as f:
        return [l.strip() for l in f if l.strip()]


def _tif_size(tif_path: str) -> tuple[int, int] | None:
    """Devuelve (H, W) de un TIF sin cargarlo entero (solo cabecera)."""
    try:
        import tifffile as tiff
        with tiff.TiffFile(tif_path) as tf:
            page = tf.pages[0]
            return page.shape[0], page.shape[1]
    except Exception:
        return None


def analizar_datos_crudos(splits: list[str]) -> dict:
    """
    Recorre los splits .txt de MADOS crudo y recopila:
      - n_recortes: recortes listados en el .txt
      - n_tif_encontrados: recortes con todos sus TIFs presentes
      - tamanios: lista de (H, W) de las máscaras encontradas
    No carga los TIFs enteros, solo lee cabeceras.
    """
    print("\n" + "═" * 60)
    print("  ETAPA 0 — DATOS CRUDOS (MADOS RAW TIFs)")
    print(f"  Ruta: {os.path.abspath(MADOS_RAW)}")
    print("═" * 60)

    resultado = {}

    for split in splits:
        lineas = _leer_split_txt(split)
        n_raw  = len(lineas)

        encontrados = 0
        tamanios    = []
        descartados_size  = 0
        descartados_miss  = 0

        for crop_line in tqdm(lineas, desc=f"  Inspeccionando {split} (raw)", leave=False):
            partes = crop_line.split("_")
            if len(partes) < 3:
                continue
            id_rec    = partes[-1]
            carpeta   = "_".join(partes[:-1])
            path_10m  = os.path.join(MADOS_RAW, carpeta, "10")

            mask_tif = os.path.join(path_10m, f"{carpeta}_L2R_cl_{id_rec}.tif")
            if not os.path.exists(mask_tif):
                descartados_miss += 1
                continue

            size = _tif_size(mask_tif)
            if size is None:
                descartados_miss += 1
                continue

            h, w = size
            if h < TARGET or w < TARGET:
                descartados_size += 1
                continue

            tamanios.append((h, w))
            encontrados += 1

        resultado[split] = {
            "n_raw":             n_raw,
            "n_encontrados":     encontrados,
            "n_miss":            descartados_miss,
            "n_size":            descartados_size,
            "tamanios":          tamanios,
        }

        areas = [h * w for h, w in tamanios] if tamanios else [0]
        hs    = [h for h, w in tamanios] if tamanios else [0]
        ws    = [w for h, w in tamanios] if tamanios else [0]

        print(f"\n  {split.upper()} — {n_raw} recortes en splits/{split}_X.txt")
        print(f"    TIFs encontrados y válidos : {encontrados}")
        print(f"    Descartados (TIF ausente)  : {descartados_miss}")
        print(f"    Descartados (< {TARGET}px)   : {descartados_size}")
        if tamanios:
            print(f"    Tamaño H  min/max/media   : {min(hs)}  /  {max(hs)}  /  {np.mean(hs):.0f} px")
            print(f"    Tamaño W  min/max/media   : {min(ws)}  /  {max(ws)}  /  {np.mean(ws):.0f} px")
            print(f"    Área      min/max/media   : {min(areas):,}  /  {max(areas):,}  /  {np.mean(areas):,.0f} px²")

    return resultado


# ══════════════════════════════════════════════════════════════
# FIGURA 0 — TABLA COMPARATIVA RAW vs PROCESADO
# ══════════════════════════════════════════════════════════════

def fig_raw_vs_procesado(raw: dict, procesado: dict, out_dir: str) -> None:
    """
    Tabla visual + barras comparando:
      · Recortes originales (splits .txt)
      · TIFs válidos encontrados en disco
      · Tiles .npy guardados tras preprocesado
      · Tiles con sargazo
    Una fila por split. Fondo blanco para TFG.
    """
    splits = [s for s in ("train", "val", "test") if s in raw and s in procesado]
    if not splits:
        print("  [AVISO] Sin datos para fig_raw_vs_procesado.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [1.2, 1]})
    fig.suptitle(
        "Pipeline MADOS: datos crudos → preprocesado → listos para entrenar",
        fontweight="bold", fontsize=14
    )

    # ── Panel izquierdo: tabla ────────────────────────────────
    ax_tab = axes[0]
    ax_tab.axis("off")

    col_labels = ["Split", "Recortes\nen .txt", "TIFs\nválidos",
                  "Tiles .npy\nguardados", "Tiles con\nsargazo",
                  "% sargazo\n/ total"]
    filas = []
    for s in splits:
        r  = raw[s]
        p  = procesado[s]
        porc = f"{100 * p['con_sarg'] / max(p['total'], 1):.1f}%"
        filas.append([
            s.upper(),
            f"{r['n_raw']:,}",
            f"{r['n_encontrados']:,}",
            f"{p['total']:,}",
            f"{p['con_sarg']:,}",
            porc,
        ])

    tabla = ax_tab.table(
        cellText=filas,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(11)
    tabla.scale(1.3, 2.2)

    # Estilo cabecera
    for j in range(len(col_labels)):
        tabla[0, j].set_facecolor("#2c3e50")
        tabla[0, j].set_text_props(color="white", fontweight="bold")

    # Colorear columna sargazo (verde suave) y filas alternas
    colores_fila = ["#f7f9fc", "#eaf4ee"]
    for i, s in enumerate(splits):
        for j in range(len(col_labels)):
            tabla[i + 1, j].set_facecolor(colores_fila[i % 2])
        # Columna "Tiles con sargazo"
        tabla[i + 1, 4].set_facecolor("#d5f0e0")
        tabla[i + 1, 4].set_text_props(color="#1a7a3c", fontweight="bold")
        tabla[i + 1, 5].set_facecolor("#d5f0e0")
        tabla[i + 1, 5].set_text_props(color="#1a7a3c", fontweight="bold")

    ax_tab.set_title("Recuento de tiles por etapa del pipeline",
                     fontsize=11, pad=12)

    # ── Panel derecho: barras agrupadas raw / npy / sarg ─────
    ax_bar = axes[1]
    x      = np.arange(len(splits))
    ancho  = 0.22

    vals_raw  = [raw[s]["n_encontrados"] for s in splits]
    vals_npy  = [procesado[s]["total"]   for s in splits]
    vals_sarg = [procesado[s]["con_sarg"] for s in splits]

    b1 = ax_bar.bar(x - ancho, vals_raw,  ancho, label="TIFs válidos (crudo)",
                    color="#85c1e9", edgecolor="#2980b9", linewidth=0.8)
    b2 = ax_bar.bar(x,         vals_npy,  ancho, label="Tiles .npy (procesado)",
                    color="#f0a500", edgecolor="#b7770d", linewidth=0.8)
    b3 = ax_bar.bar(x + ancho, vals_sarg, ancho, label="Tiles con sargazo",
                    color="#1a7a3c", edgecolor="#145a32", linewidth=0.8)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax_bar.text(bar.get_x() + bar.get_width() / 2, h + max(vals_raw) * 0.01,
                            f"{int(h):,}", ha="center", va="bottom", fontsize=9)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([s.upper() for s in splits], fontsize=12)
    ax_bar.set_ylabel("Número de tiles / recortes")
    ax_bar.set_title("Comparativa por etapa del pipeline", fontsize=11)
    ax_bar.legend(fontsize=9, framealpha=0.9)
    ax_bar.grid(axis="y")
    ax_bar.set_ylim(0, max(vals_raw + [1]) * 1.18)

    plt.tight_layout()
    ruta = os.path.join(out_dir, "0_resumen_raw_vs_procesado.pdf")
    plt.savefig(ruta, format="pdf", bbox_inches="tight", facecolor="white")
    print(f"  [OK] {ruta}")
    plt.close()


def fig_tamanios_tiles(raw: dict, out_dir: str) -> None:
    """
    Histograma de tamaños (H y W) de los TIFs originales de MADOS.
    Muestra la variabilidad de resolución espacial antes del crop a 224px.
    """
    todos_h, todos_w = [], []
    for s, d in raw.items():
        todos_h += [h for h, w in d["tamanios"]]
        todos_w += [w for h, w in d["tamanios"]]

    if not todos_h:
        print("  [AVISO] Sin datos de tamaño de TIFs crudos.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        "Distribución de tamaños de los TIFs crudos de MADOS\n"
        f"(antes del crop a {TARGET}×{TARGET} px)",
        fontweight="bold", fontsize=13
    )

    for ax, vals, dim, color in [
        (axes[0], todos_h, "Alto (H)", "#5dade2"),
        (axes[1], todos_w, "Ancho (W)", "#52c47a"),
    ]:
        ax.hist(vals, bins=20, color=color, edgecolor="#333333",
                linewidth=0.6, alpha=0.85)
        ax.axvline(np.mean(vals), color="#c0392b", linestyle="--",
                   linewidth=1.4, label=f"Media: {np.mean(vals):.0f} px")
        ax.axvline(TARGET, color="#1a7a3c", linestyle=":",
                   linewidth=1.4, label=f"Crop target: {TARGET} px")
        ax.set_xlabel(f"{dim} (px)")
        ax.set_ylabel("Número de tiles")
        ax.set_title(f"Distribución de {dim}")
        ax.legend(fontsize=9)
        ax.grid(axis="y")

        # Estadísticas como texto
        stats = (f"min={min(vals)}  max={max(vals)}\n"
                 f"media={np.mean(vals):.0f}  mediana={int(np.median(vals))}")
        ax.text(0.97, 0.95, stats, transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                          edgecolor="#aaaaaa", alpha=0.9))

    plt.tight_layout()
    ruta = os.path.join(out_dir, "0b_tamanios_tifs_crudos.pdf")
    plt.savefig(ruta, format="pdf", bbox_inches="tight", facecolor="white")
    print(f"  [OK] {ruta}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# FIGURA 1 — TILES POR SPLIT (procesado)
# ══════════════════════════════════════════════════════════════

def fig_tiles_por_split(datos_splits: dict, out_dir: str) -> None:
    splits   = list(datos_splits.keys())
    totales  = [datos_splits[s]["total"]    for s in splits]
    con_sarg = [datos_splits[s]["con_sarg"] for s in splits]
    sin_sarg = [t - c for t, c in zip(totales, con_sarg)]

    x     = np.arange(len(splits))
    ancho = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Distribución de tiles por split (datos procesados)",
                 fontweight="bold", fontsize=14)

    barras_sin = ax.bar(x - ancho/2, sin_sarg, ancho,
                        label="Sin sargazo", color="#5dade2",
                        edgecolor="#2980b9", linewidth=0.8)
    barras_con = ax.bar(x + ancho/2, con_sarg, ancho,
                        label="Con sargazo (cl. 2 ó 3)",
                        color="#1a7a3c", edgecolor="#145a32", linewidth=0.8)

    for bar in barras_sin:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 5,
                f"{int(h):,}", ha="center", va="bottom", fontsize=10)
    for bar in barras_con:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 5,
                f"{int(h):,}", ha="center", va="bottom", fontsize=10,
                color="#1a7a3c", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in splits], fontsize=12)
    ax.set_ylabel("Número de tiles (parches 224×224 px)")
    ax.set_xlabel("Subconjunto del dataset")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y")
    ax.set_ylim(0, max(totales) * 1.15)

    for i, (s, t) in enumerate(zip(splits, totales)):
        ax.text(x[i], max(totales) * 1.08, f"Total: {int(t):,}",
                ha="center", fontsize=9, color="#444444")

    plt.tight_layout()
    ruta = os.path.join(out_dir, "1_distribucion_tiles_por_split.pdf")
    plt.savefig(ruta, format="pdf", bbox_inches="tight", facecolor="white")
    print(f"  [OK] {ruta}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# FIGURA 2 — PÍXELES POR CLASE
# ══════════════════════════════════════════════════════════════

def fig_pixeles_por_clase(conteo: np.ndarray, split: str, out_dir: str) -> None:
    total  = conteo.sum()
    porcs  = 100 * conteo / total
    clases_presentes = [c for c in range(NUM_CLASSES) if conteo[c] > 0]

    # Solo gráfica de zoom — clases minoritarias
    clases_zoom = [c for c in clases_presentes if c not in {0, 7} and conteo[c] > 0]

    if not clases_zoom:
        print(f"  [AVISO] Sin clases minoritarias en {split}. Saltando.")
        return

    fig, ax_zoom = plt.subplots(1, 1, figsize=(14, 7))
    fig.suptitle(
        f"Clases minoritarias — {split.upper()} "
        f"(excluye Non-annotated y Marine Water)\n"
        f"Total píxeles del split: {int(total):,}",
        fontweight="bold", fontsize=14
    )

    y_zoom = [porcs[c] for c in clases_zoom]
    col_z  = [CLASES[c][1] for c in clases_zoom]
    bord_z = ["#145a32" if c in CLASES_SARG else "#888888" for c in clases_zoom]
    gros_z = [1.5 if c in CLASES_SARG else 0.5 for c in clases_zoom]

    barras_z = ax_zoom.bar(
    range(len(clases_zoom)), y_zoom,
    color=col_z, edgecolor=bord_z, linewidth=gros_z,
    width=0.6
    )

    for bar, p, c in zip(barras_z, y_zoom, clases_zoom):
        ax_zoom.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(y_zoom) * 0.015,
            f"{p:.3f}%\n({int(conteo[c]):,} px)",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
            color="#145a32" if c in CLASES_SARG else "#333333",
        )

    ax_zoom.set_xticks(range(len(clases_zoom)))
    ax_zoom.set_xticklabels(
        [f"Cl.{c}: {CLASES[c][0]}" for c in clases_zoom],
        rotation=45, ha="right", fontsize=14, fontweight="bold"
    )
    ax_zoom.set_ylabel("Porcentaje de píxeles (%)", fontsize=12)
    ax_zoom.set_title(
        "Zoom — clases minoritarias (excluye Non-annotated y Marine Water)",
        fontsize=13, fontweight="bold"
    )
    ax_zoom.grid(axis="y")
    ax_zoom.legend(handles=[
        mpatches.Patch(color=CLASES[2][1], edgecolor="#145a32", linewidth=1.5,
                       label="Cl. 2 — Dense Sargassum (★)"),
        mpatches.Patch(color=CLASES[3][1], edgecolor="#145a32", linewidth=1.5,
                       label="Cl. 3 — Sparse Floating Algae (★)"),
    ], loc="upper right", framealpha=0.9, fontsize=11)

    ax_zoom.set_ylim(0, max(y_zoom) * 1.45)

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    ruta = os.path.join(out_dir, f"2_distribucion_pixeles_por_clase_{split}.pdf")
    plt.savefig(ruta, format="pdf", bbox_inches="tight", facecolor="white")
    print(f"  [OK] {ruta}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# FIGURA 3 — DESEQUILIBRIO DE CLASES
# ══════════════════════════════════════════════════════════════

def fig_desequilibrio(conteo: np.ndarray, split: str, out_dir: str) -> None:
    total    = conteo.sum()
    porcs    = 100 * conteo / total

    clases_pres = [(c, conteo[c]) for c in range(NUM_CLASSES) if conteo[c] > 0]
    clases_pres.sort(key=lambda x: x[1], reverse=True)

    ids_ord  = [c for c, _ in clases_pres]
    vals_ord = [v for _, v in clases_pres]
    col_ord  = [CLASES[c][1] if c != 0 else "#cccccc" for c in ids_ord]
    bord_ord = ["#145a32" if c in CLASES_SARG else "#888888" for c in ids_ord]
    gros_ord = [1.8 if c in CLASES_SARG else 0.5 for c in ids_ord]
    etq_ord  = [f"Cl.{c}: {CLASES[c][0][:20]}" for c in ids_ord]

    fig, ax_bar = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle(
        f"Desequilibrio de clases — {split.upper()}\n"
        f"Total: {int(total):,} píxeles",
        fontweight="bold", fontsize=14
    )

    barras_h = ax_bar.barh(
        range(len(ids_ord)), vals_ord,
        color=col_ord, edgecolor=bord_ord,
        linewidth=gros_ord, height=0.65
    )
    ax_bar.set_xscale("log")
    ax_bar.set_yticks(range(len(ids_ord)))
    ax_bar.set_yticklabels(etq_ord, fontsize=11, fontweight="bold")
    ax_bar.tick_params(axis="y", labelsize=11, pad=4)
    ax_bar.set_xlabel("Número de píxeles (escala logarítmica)", fontsize=11)
    ax_bar.set_title(
        "Píxeles por clase (escala log)\nOrdenado de mayor a menor",
        fontsize=14, fontweight="bold"
    )
    ax_bar.grid(axis="x")
    ax_bar.invert_yaxis()

    # Ampliar el eje X a la derecha para que los valores no se salgan
    ax_bar.set_xlim(right=max(vals_ord) * 8)

    for bar, v, c in zip(barras_h, vals_ord, ids_ord):
        ax_bar.text(
            v * 1.05, bar.get_y() + bar.get_height() / 2,
            f"{int(v):,}", va="center", fontsize=11,
            color="#145a32" if c in CLASES_SARG else "#333333",
            fontweight="bold"
        )

    # Anotación destacada del sargazo
    px_sarg   = int(conteo[2] + conteo[3])
    porc_sarg = 100 * px_sarg / total
    ax_bar.text(
        0.99, 0.01,
        f"Sargazo total (cl.2+3): {px_sarg:,} px = {porc_sarg:.4f}%",
        transform=ax_bar.transAxes,
        ha="right", va="bottom", fontsize=10,
        color="#1a7a3c", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0faf5",
                  edgecolor="#1a7a3c", alpha=0.9)
    )

    plt.tight_layout()
    ruta = os.path.join(out_dir, f"3_desequilibrio_clases_{split}.pdf")
    plt.savefig(ruta, format="pdf", bbox_inches="tight", facecolor="white")
    print(f"  [OK] {ruta}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# FIGURA 4 — BARRAS APILADAS SARGAZO / SIN SARGAZO
# ══════════════════════════════════════════════════════════════

def fig_tiles_apiladas(datos_splits: dict, out_dir: str) -> None:
    splits   = list(datos_splits.keys())
    totales  = np.array([datos_splits[s]["total"]    for s in splits])
    con_sarg = np.array([datos_splits[s]["con_sarg"] for s in splits])
    sin_sarg = totales - con_sarg
    porcs    = 100 * con_sarg / np.maximum(totales, 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("Tiles con sargazo vs. sin sargazo por subconjunto",
                 fontweight="bold", fontsize=14)

    x  = np.arange(len(splits))
    b1 = ax.bar(x, sin_sarg, label="Sin sargazo",
                color="#5dade2", edgecolor="#2980b9", linewidth=0.8)
    b2 = ax.bar(x, con_sarg, bottom=sin_sarg,
                label="Con sargazo (cl. 2 ó 3)",
                color="#1a7a3c", edgecolor="#145a32", linewidth=0.8)

    for bar, c, t, p in zip(b2, con_sarg, totales, porcs):
        if c > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(c)} tiles\n({float(p):.1f}%)",
                    ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold")

    for xi, t in zip(x, totales):
        ax.text(xi, t + max(totales) * 0.01,
                f"Total: {int(t)}", ha="center", fontsize=9.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in splits], fontsize=12)
    ax.set_ylabel("Número de tiles")
    ax.set_xlabel("Subconjunto del dataset")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y")
    ax.set_ylim(0, max(totales) * 1.12)

    plt.tight_layout()
    ruta = os.path.join(out_dir, "4_tiles_con_sargazo_por_split.pdf")
    plt.savefig(ruta, format="pdf", bbox_inches="tight", facecolor="white")
    print(f"  [OK] {ruta}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# FIGURA 5 — PROPORCIONES TRAIN / VAL / TEST
# ══════════════════════════════════════════════════════════════

def fig_proporciones_split(datos_splits: dict, raw_data: dict, out_dir: str) -> None:
    """
    Muestra cómo se divide el dataset en train / val / test con tres paneles:

      Izquierda  — Pie chart de tiles totales por split.
      Centro     — Pie chart de tiles CON sargazo por split.
      Derecha    — Tabla resumen con nº de tiles, % del total, tiles con
                   sargazo y % de sargazo dentro de cada split.

    Útil para responder en la defensa del TFG:
      "¿Qué porcentaje usas para entrenamiento / validación / test?"
    """
    splits  = list(datos_splits.keys())
    totales = [datos_splits[s]["total"]    for s in splits]
    sargs   = [datos_splits[s]["con_sarg"] for s in splits]
    gran_t  = sum(totales)

    # También recortes crudos si están disponibles
    raw_totales = [raw_data[s]["n_raw"] if s in raw_data else 0 for s in splits]
    gran_raw    = sum(raw_totales)

    colores_split = {
        "train": "#2980b9",
        "val":   "#e67e22",
        "test":  "#27ae60",
    }
    col_list = [colores_split.get(s, "#aaaaaa") for s in splits]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6),
                             gridspec_kw={"width_ratios": [1, 1, 1.4]})
    fig.suptitle(
        "División del dataset MADOS en subconjuntos de entrenamiento\n"
        "(partición definida por los ficheros splits/ del dataset original)",
        fontweight="bold", fontsize=13
    )

    # ── Pie 1: tiles totales ──────────────────────────────────
    ax1 = axes[0]
    wedges1, texts1, auto1 = ax1.pie(
        totales,
        labels=[s.upper() for s in splits],
        colors=col_list,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 11, "fontweight": "bold"},
    )
    for at in auto1:
        at.set_fontsize(10)
    ax1.set_title(f"Tiles procesados (.npy)\nTotal: {gran_t:,}", fontsize=11)

    # ── Pie 2: tiles con sargazo ──────────────────────────────
    ax2 = axes[1]
    gran_sarg = sum(sargs)
    # Evitar pie vacío si algún split no tiene sargazo
    sargs_safe = [max(s, 0) for s in sargs]
    wedges2, texts2, auto2 = ax2.pie(
        sargs_safe,
        labels=[s.upper() for s in splits],
        colors=col_list,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 11, "fontweight": "bold"},
    )
    for at in auto2:
        at.set_fontsize(10)
    ax2.set_title(f"Tiles CON sargazo\nTotal: {gran_sarg:,}", fontsize=11)

    # ── Tabla resumen ────────────────────────────────────────
    ax3 = axes[2]
    ax3.axis("off")

    col_labels = [
        "Split",
        "Recortes\noriginales",
        "Tiles\n.npy",
        "% del\ntotal",
        "Tiles con\nsargazo",
        "% sargazo\nen split",
    ]

    filas = []
    for s, tot, sarg, raw_t in zip(splits, totales, sargs, raw_totales):
        porc_tot  = f"{100 * tot  / max(gran_t, 1):.1f}%"
        porc_sarg = f"{100 * sarg / max(tot,    1):.1f}%"
        raw_str   = f"{raw_t:,}" if raw_t > 0 else "—"
        filas.append([
            s.upper(),
            raw_str,
            f"{tot:,}",
            porc_tot,
            f"{sarg:,}",
            porc_sarg,
        ])

    # Fila de totales
    porc_sarg_global = f"{100 * gran_sarg / max(gran_t, 1):.1f}%"
    filas.append([
        "TOTAL",
        f"{gran_raw:,}" if gran_raw > 0 else "—",
        f"{gran_t:,}",
        "100%",
        f"{gran_sarg:,}",
        porc_sarg_global,
    ])

    tabla = ax3.table(
        cellText=filas,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10.5)
    tabla.scale(1.25, 2.4)

    # Estilo cabecera
    for j in range(len(col_labels)):
        tabla[0, j].set_facecolor("#2c3e50")
        tabla[0, j].set_text_props(color="white", fontweight="bold")

    # Colores por split
    for i, s in enumerate(splits):
        base_color = colores_split.get(s, "#eeeeee")
        # Versión muy suave del color del split
        r = int(base_color[1:3], 16)
        g = int(base_color[3:5], 16)
        b = int(base_color[5:7], 16)
        pastel = f"#{min(r+180,255):02x}{min(g+180,255):02x}{min(b+180,255):02x}"
        for j in range(len(col_labels)):
            tabla[i + 1, j].set_facecolor(pastel)
        # Columna sargazo: siempre verde suave
        tabla[i + 1, 4].set_facecolor("#d5f0e0")
        tabla[i + 1, 4].set_text_props(color="#1a7a3c", fontweight="bold")
        tabla[i + 1, 5].set_facecolor("#d5f0e0")
        tabla[i + 1, 5].set_text_props(color="#1a7a3c", fontweight="bold")

    # Fila TOTAL
    fila_tot = len(splits) + 1
    for j in range(len(col_labels)):
        tabla[fila_tot, j].set_facecolor("#dde3ea")
        tabla[fila_tot, j].set_text_props(fontweight="bold")

    ax3.set_title("Resumen cuantitativo de la partición", fontsize=11, pad=14)

    # Nota aclaratoria
    fig.text(
        0.5, 0.01,
        "Nota: la partición train/val/test está definida por los ficheros splits/*.txt "
        "del dataset MADOS original (Kikaki et al., 2022), no es una decisión propia.",
        ha="center", fontsize=8.5, color="#555555",
        style="italic"
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    ruta = os.path.join(out_dir, "5_proporciones_train_val_test.pdf")
    plt.savefig(ruta, format="pdf", bbox_inches="tight", facecolor="white")
    print(f"  [OK] {ruta}")
    plt.close()


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main() -> None:
    global DATASET_DIR, MADOS_RAW

    parser = argparse.ArgumentParser(
        description="Análisis completo (crudo + procesado) del dataset MADOS para TFG",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dataset", default=str(SARGASSUM_READY),
                        help="Ruta raíz del dataset procesado (.npy)")
    parser.add_argument("--mados-raw", default=str(MADOS_RAW_DIR),
                        help="Ruta raíz de MADOS crudo (TIFs de ACOLITE)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        choices=["train", "val", "test"])
    parser.add_argument("--out", default=".",
                        help="Directorio de salida para las figuras")
    parser.add_argument("--solo", default=None,
                        choices=["crudo", "procesado"],
                        help="Ejecutar solo una etapa (default: ambas)")
    args = parser.parse_args()

    DATASET_DIR = args.dataset
    MADOS_RAW   = args.mados_raw
    out_dir     = args.out
    os.makedirs(out_dir, exist_ok=True)

    print("═" * 60)
    print("  ANÁLISIS DE DISTRIBUCIÓN — MADOS (crudo + procesado)")
    print(f"  Raw     : {os.path.abspath(MADOS_RAW)}")
    print(f"  Procesado: {os.path.abspath(DATASET_DIR)}")
    print(f"  Splits  : {args.splits}")
    print(f"  Salida  : {os.path.abspath(out_dir)}")
    print("═" * 60)

    raw_data = {}
    if args.solo in (None, "crudo"):
        if os.path.exists(MADOS_RAW):
            raw_data = analizar_datos_crudos(args.splits)
        else:
            print(f"\n  [AVISO] MADOS raw no encontrado en {MADOS_RAW}.")
            print("  Saltando análisis de datos crudos.")

    # ── Datos procesados ─────────────────────────────────────
    datos_splits = {}
    conteos_px   = {}

    if args.solo in (None, "procesado"):
        if not os.path.exists(DATASET_DIR):
            print(f"\n[ERROR] Dataset procesado no encontrado en: {DATASET_DIR}")
            sys.exit(1)

        for split in args.splits:
            print(f"\n  Procesando {split.upper()} (npy)...")
            mask_paths = listar_mascaras(split)
            if not mask_paths:
                print(f"  [AVISO] Sin máscaras en {split}. Saltando.")
                continue

            total, con_sarg = contar_tiles(mask_paths)
            conteo_px       = contar_pixeles_por_clase(mask_paths)
            datos_splits[split] = {"total": total, "con_sarg": con_sarg}
            conteos_px[split]   = conteo_px

            t_px = conteo_px.sum()
            print(f"\n  {split.upper()} — {total} tiles | {t_px:,} píxeles totales")
            print(f"  {'ID':>3}  {'Clase':<30}  {'Píxeles':>15}  {'%':>7}")
            print(f"  {'-'*60}")
            for c in range(NUM_CLASSES):
                if conteo_px[c] > 0:
                    marca = " ★" if c in CLASES_SARG else "  "
                    print(f"  {c:>3}{marca}  {CLASES[c][0]:<28}  "
                          f"{conteo_px[c]:>15,}  {100*conteo_px[c]/t_px:>6.3f}%")
            print(f"\n  ► Tiles con sargazo : {con_sarg} / {total} "
                  f"({100*con_sarg/max(total,1):.1f}%)")
            print(f"  ► Píxeles cl.2+3    : "
                  f"{conteo_px[2]+conteo_px[3]:,} / {t_px:,} "
                  f"({100*(conteo_px[2]+conteo_px[3])/max(t_px,1):.4f}%)")

    # ── Figuras ──────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  GENERANDO FIGURAS...")
    print("═" * 60)

    # Figuras que necesitan AMBOS (raw + procesado)
    if raw_data and datos_splits:
        fig_raw_vs_procesado(raw_data, datos_splits, out_dir)

    # Figuras solo con raw
    if raw_data:
        fig_tamanios_tiles(raw_data, out_dir)

    # Figuras solo con procesado
    if datos_splits:
        fig_tiles_por_split(datos_splits, out_dir)
        fig_tiles_apiladas(datos_splits, out_dir)
        fig_proporciones_split(datos_splits, raw_data, out_dir)
        for split, conteo in conteos_px.items():
            fig_pixeles_por_clase(conteo, split, out_dir)
            fig_desequilibrio(conteo, split, out_dir)

    print("\n" + "═" * 60)
    print("  ANÁLISIS COMPLETADO")
    print(f"  Figuras en: {os.path.abspath(out_dir)}")
    print("═" * 60)


if __name__ == "__main__":
    main()