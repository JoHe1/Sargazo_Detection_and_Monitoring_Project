"""
verificar_dataset.py
=====================
Herramienta completa de verificación y visualización del dataset MADOS preprocesado.

Ejecuta TODO de una vez (recomendado antes de entrenar):
    python verificar_dataset.py

O por secciones:
    python verificar_dataset.py --solo integridad
    python verificar_dataset.py --solo muestras  --split train --n 6
    python verificar_dataset.py --solo sargassum              # Solo tiles con sargazo
    python verificar_dataset.py --solo estadisticas
    python verificar_dataset.py --solo bandas     --idx 0     # Inspección profunda de un tile
"""

import os
import sys
import glob
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ══════════════════════════════════════════════════════════════
from core.config.paths import SARGASSUM_READY
DATASET_DIR = str(SARGASSUM_READY)
NUM_CLASSES = 16

# Mapping oficial MADOS con colores distintivos
CLASES = {
    0:  ("Non-annotated",            "#1a1a1a"),  # Gris muy oscuro
    1:  ("Marine Debris",            "#e74c3c"),  # Rojo
    2:  ("Dense Sargassum",          "#1e8449"),  # Verde oscuro   ★ OBJETIVO
    3:  ("Sparse Floating Algae",    "#58d68d"),  # Verde claro    ★ OBJETIVO
    4:  ("Natural Organic Material", "#f39c12"),  # Naranja
    5:  ("Ship",                     "#8e44ad"),  # Morado
    6:  ("Oil Spill",                "#17202a"),  # Negro azulado
    7:  ("Marine Water",             "#2e86c1"),  # Azul marino
    8:  ("Sediment-Laden Water",     "#d4ac0d"),  # Amarillo-marrón
    9:  ("Foam",                     "#f2f3f4"),  # Blanco roto
    10: ("Turbid Water",             "#5499c7"),  # Azul grisáceo
    11: ("Shallow Water",            "#a9cce3"),  # Azul claro
    12: ("Waves & Wakes",            "#abebc6"),  # Verde agua
    13: ("Oil Platform",             "#922b21"),  # Rojo oscuro
    14: ("Jellyfish",                "#f1948a"),  # Rosa
    15: ("Sea Snot",                 "#b7950b"),  # Marrón dorado
}

COLORES_LISTA  = [CLASES[i][1] for i in range(NUM_CLASSES)]
CMAP_CLASES    = ListedColormap(COLORES_LISTA)
BOUNDS         = np.arange(-0.5, NUM_CLASSES + 0.5, 1)
NORM_CLASES    = BoundaryNorm(BOUNDS, CMAP_CLASES.N)

# Clases objetivo (sargazo)
CLASES_SARGASSUM = {2, 3}
# Todas las clases de interés para el TFT
CLASES_INTERES   = {1, 2, 3, 5, 6}


# ══════════════════════════════════════════════════════════════
# UTILIDADES DE CARGA
# ══════════════════════════════════════════════════════════════

def listar_pares(split: str):
    """Devuelve lista de (img_path, mask_path) para el split dado."""
    img_dir  = os.path.join(DATASET_DIR, split, "images")
    mask_dir = os.path.join(DATASET_DIR, split, "masks")
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.npy")))
    pares = []
    for ip in img_paths:
        mp = os.path.join(mask_dir, os.path.basename(ip))
        pares.append((ip, mp))
    return pares


def cargar_muestra(img_path: str, mask_path: str):
    """
    Carga un tile y su máscara aplicando exactamente el mismo
    preprocesamiento que SargassoDataset (center crop 224×224).
    """
    img  = np.load(img_path).astype(np.float32)   # (H, W, 4) orden B,G,R,NIR
    mask = np.load(mask_path).astype(np.int32)    # (H, W) valores 0-15

    img  = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

    # Reordenar: (B, G, R, NIR) → (R, G, B, NIR)
    img = img[:, :, [2, 1, 0, 3]]

    # Normalizar igual que en el Dataset
    if img.max() > 10.0:
        img = img / 10000.0
    img = np.clip(img * 5.0, 0.0, 1.0)

    # Center crop a 224×224 (igual que val/test en el Dataset)
    TARGET = 224
    h, w   = mask.shape
    y0 = (h - TARGET) // 2
    x0 = (w - TARGET) // 2
    img  = img [y0:y0+TARGET, x0:x0+TARGET, :]
    mask = mask[y0:y0+TARGET, x0:x0+TARGET]

    return img, mask


def clases_presentes(mask: np.ndarray) -> set:
    return set(np.unique(mask).tolist())


def tiene_sargassum(mask: np.ndarray) -> bool:
    return bool(CLASES_SARGASSUM & clases_presentes(mask))


def parches_leyenda(mask: np.ndarray, destacar: set = None):
    patches = []
    for cid in sorted(clases_presentes(mask)):
        nombre, color = CLASES[cid]
        label = f"{'★ ' if destacar and cid in destacar else ''}{cid}: {nombre}"
        borde = "gold" if destacar and cid in destacar else "none"
        patches.append(mpatches.Patch(
            facecolor=color, edgecolor=borde, linewidth=1.5, label=label))
    return patches


# ══════════════════════════════════════════════════════════════
# SECCIÓN 1 — INTEGRIDAD DEL DATASET
# ══════════════════════════════════════════════════════════════

def verificar_integridad():
    print("\n" + "═"*60)
    print("  SECCIÓN 1 — INTEGRIDAD DEL DATASET")
    print("═"*60)

    todo_ok = True
    resumen = {}

    for split in ("train", "val", "test"):
        img_dir  = os.path.join(DATASET_DIR, split, "images")
        mask_dir = os.path.join(DATASET_DIR, split, "masks")

        imgs  = sorted(glob.glob(os.path.join(img_dir,  "*.npy")))
        masks = sorted(glob.glob(os.path.join(mask_dir, "*.npy")))

        n_imgs  = len(imgs)
        n_masks = len(masks)

        # Verificar que cada imagen tiene su máscara y viceversa
        nombres_img  = {os.path.basename(p) for p in imgs}
        nombres_mask = {os.path.basename(p) for p in masks}
        sin_mascara  = nombres_img  - nombres_mask
        sin_imagen   = nombres_mask - nombres_img

        # Verificar dimensiones y rango de valores en una muestra (10 tiles)
        errores_dim   = []
        errores_rango = []
        muestra_check = random.sample(imgs, min(10, n_imgs)) if n_imgs > 0 else []

        for ip in muestra_check:
            mp = os.path.join(mask_dir, os.path.basename(ip))
            if not os.path.exists(mp):
                continue
            img_raw  = np.load(ip)
            mask_raw = np.load(mp)

            # Dimensiones esperadas: imagen (H,W,4), máscara (H,W)
            if img_raw.ndim != 3 or img_raw.shape[2] != 4:
                errores_dim.append(f"{os.path.basename(ip)}: forma imagen={img_raw.shape}")
            if mask_raw.ndim != 2:
                errores_dim.append(f"{os.path.basename(ip)}: forma máscara={mask_raw.shape}")
            if img_raw.shape[:2] != mask_raw.shape:
                errores_dim.append(f"{os.path.basename(ip)}: imagen {img_raw.shape[:2]} ≠ máscara {mask_raw.shape}")
            if mask_raw.min() < 0 or mask_raw.max() > 15:
                errores_rango.append(f"{os.path.basename(ip)}: rango máscara [{mask_raw.min()},{mask_raw.max()}]")

        # Contar tiles con sargazo
        n_sarg = 0
        for ip in tqdm(imgs, desc=f"  Analizando {split}", leave=False):
            mp = os.path.join(mask_dir, os.path.basename(ip))
            if os.path.exists(mp):
                m = np.load(mp)
                if tiene_sargassum(m):
                    n_sarg += 1

        resumen[split] = {
            "imgs": n_imgs, "masks": n_masks,
            "sin_mascara": sin_mascara, "sin_imagen": sin_imagen,
            "errores_dim": errores_dim, "errores_rango": errores_rango,
            "tiles_sargassum": n_sarg,
        }

        ok_emparejado = len(sin_mascara) == 0 and len(sin_imagen) == 0
        ok_dim        = len(errores_dim) == 0
        ok_rango      = len(errores_rango) == 0
        estado        = "✔ OK" if (ok_emparejado and ok_dim and ok_rango) else "✗ PROBLEMA"
        if not (ok_emparejado and ok_dim and ok_rango):
            todo_ok = False

        print(f"\n  [{estado}] {split.upper()}")
        print(f"    Imágenes  : {n_imgs}")
        print(f"    Máscaras  : {n_masks}")
        print(f"    Con sargazo (clases 2 ó 3): {n_sarg}  "
              f"({100*n_sarg/max(n_imgs,1):.1f}% del split)")

        if sin_mascara:
            print(f"    ⚠ Sin máscara : {len(sin_mascara)} archivos")
            todo_ok = False
        if sin_imagen:
            print(f"    ⚠ Sin imagen  : {len(sin_imagen)} archivos")
            todo_ok = False
        if errores_dim:
            print(f"    ⚠ Errores dim : {errores_dim}")
            todo_ok = False
        if errores_rango:
            print(f"    ⚠ Rango máscara fuera de [0,15]: {errores_rango}")
            todo_ok = False

    print(f"\n  {'✔ Dataset íntegro y listo para entrenar.' if todo_ok else '✗ Se encontraron problemas.'}")
    return resumen


# ══════════════════════════════════════════════════════════════
# SECCIÓN 2 — VISUALIZACIÓN DE MUESTRAS
# ══════════════════════════════════════════════════════════════

def visualizar_muestras(split: str, indices: list):
    """
    Por cada muestra muestra 5 paneles:
      RGB | NIR | Máscara GT | RGB+Máscara superpuesta | Distribución de píxeles
    """
    pares = listar_pares(split)
    if not pares:
        print(f"[ERROR] No hay datos en {split}")
        return

    indices = [i % len(pares) for i in indices]
    n = len(indices)

    print(f"\n{'═'*60}")
    print(f"  SECCIÓN 2 — MUESTRAS ({split.upper()}, {n} tiles)")
    print(f"{'═'*60}")

    fig = plt.figure(figsize=(22, 4.5 * n))
    fig.suptitle(
        f"Dataset MADOS — {split.upper()}  |  {len(pares)} tiles totales\n"
        f"★ = clases con sargazo   "
        f"Columnas: RGB · NIR · Ground Truth · GT superpuesto · Distribución",
        fontsize=11, fontweight="bold", y=1.01
    )

    for fila, idx in enumerate(indices):
        ip, mp = pares[idx]
        img, mask = cargar_muestra(ip, mp)
        nombre    = os.path.basename(ip).replace(".npy", "")
        rgb = img[:, :, :3]
        nir = img[:, :, 3]

        gs = gridspec.GridSpec(n, 5, figure=fig,
                               hspace=0.35, wspace=0.25)

        ax_rgb    = fig.add_subplot(gs[fila, 0])
        ax_nir    = fig.add_subplot(gs[fila, 1])
        ax_mask   = fig.add_subplot(gs[fila, 2])
        ax_over   = fig.add_subplot(gs[fila, 3])
        ax_bar    = fig.add_subplot(gs[fila, 4])

        # — Título de fila —
        hay_sarg = tiene_sargassum(mask)
        etiqueta = f"[{idx}] {nombre[:28]}"
        if hay_sarg:
            etiqueta += "  ★ SARGAZO"
        ax_rgb.set_title(etiqueta, fontsize=8, loc="left", color="darkgreen" if hay_sarg else "black")

        # — RGB —
        ax_rgb.imshow(rgb)
        ax_rgb.set_xlabel("RGB (R-G-B)", fontsize=8)
        ax_rgb.axis("off")

        # — NIR —
        ax_nir.imshow(nir, cmap="inferno")
        ax_nir.set_xlabel("NIR 833 nm", fontsize=8)
        ax_nir.axis("off")

        # — Ground Truth —
        ax_mask.imshow(mask, cmap=CMAP_CLASES, norm=NORM_CLASES, interpolation="nearest")
        ax_mask.set_xlabel("Ground Truth (máscara)", fontsize=8)
        ax_mask.axis("off")
        leyenda = parches_leyenda(mask, destacar=CLASES_SARGASSUM)
        ax_mask.legend(handles=leyenda, fontsize=6, loc="lower right",
                       framealpha=0.85, ncol=1,
                       title="Clases presentes", title_fontsize=6)

        # — RGB + GT superpuesto —
        ax_over.imshow(rgb)
        mask_rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        for cid in range(NUM_CLASSES):
            if cid == 7:   # Marine Water: transparente para no tapar imagen
                continue
            hex_c = CLASES[cid][1].lstrip("#")
            r, g, b = [int(hex_c[i:i+2], 16) / 255 for i in (0, 2, 4)]
            alpha   = 0.6 if cid in CLASES_SARGASSUM else 0.35
            pixeles = mask == cid
            mask_rgba[pixeles] = [r, g, b, alpha]
        ax_over.imshow(mask_rgba, interpolation="nearest")
        ax_over.set_xlabel("RGB + GT superpuesto", fontsize=8)
        ax_over.axis("off")

        # — Distribución de píxeles por clase —
        total = mask.size
        ids, porcs, colores = [], [], []
        for cid in range(NUM_CLASSES):
            n_px = int((mask == cid).sum())
            if n_px > 0:
                ids.append(cid)
                porcs.append(100 * n_px / total)
                c_hex = CLASES[cid][1]
                colores.append("#555555" if c_hex == "#1a1a1a" else c_hex)

        etqs = [str(i) for i in ids]
        barras = ax_bar.barh(etqs[::-1], porcs[::-1],
                             color=colores[::-1], edgecolor="grey", linewidth=0.4)
        for bar, cid_r, p_r in zip(barras, ids[::-1], porcs[::-1]):
            if p_r > 1.5:
                ax_bar.text(p_r + 0.3, bar.get_y() + bar.get_height()/2,
                            f"{p_r:.1f}%", va="center", fontsize=6.5)
            if cid_r in CLASES_SARGASSUM:
                bar.set_linewidth(1.5)
                bar.set_edgecolor("gold")

        # Etiquetas del eje Y con nombre de clase
        ax_bar.set_yticks(range(len(ids)))
        ax_bar.set_yticklabels(
            [f"{i}: {CLASES[i][0][:14]}" for i in ids[::-1]],
            fontsize=6.5
        )
        ax_bar.set_xlabel("% píxeles", fontsize=8)
        ax_bar.set_title("Distribución GT", fontsize=8)
        ax_bar.tick_params(axis="x", labelsize=7)

    plt.tight_layout()
    out = f"muestras_{split}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  [OK] Guardado: {out}")
    plt.show()


# ══════════════════════════════════════════════════════════════
# SECCIÓN 3 — TILES CON SARGAZO
# ══════════════════════════════════════════════════════════════

def visualizar_sargassum(split: str, n: int = 6):
    """Busca y muestra tiles que contienen sargazo (clases 2 ó 3)."""
    print(f"\n{'═'*60}")
    print(f"  SECCIÓN 3 — TILES CON SARGAZO ({split.upper()})")
    print(f"{'═'*60}")

    pares = listar_pares(split)
    pares_sarg = []
    for ip, mp in tqdm(pares, desc="  Buscando sargazo"):
        if os.path.exists(mp):
            m = np.load(mp)
            if tiene_sargassum(m):
                pares_sarg.append((ip, mp))

    print(f"  Tiles con sargazo en {split}: {len(pares_sarg)} / {len(pares)}")

    if not pares_sarg:
        print("  No se encontraron tiles con sargazo en este split.")
        return

    seleccion = random.sample(pares_sarg, min(n, len(pares_sarg)))
    visualizar_muestras.__wrapped__ = True  # flag interno para no reimprimir cabecera

    n_sel = len(seleccion)
    fig, axes = plt.subplots(n_sel, 3, figsize=(15, 4.5 * n_sel))
    if n_sel == 1:
        axes = [axes]

    fig.suptitle(
        f"Tiles con SARGAZO — {split.upper()} "
        f"({len(pares_sarg)} tiles con sargazo de {len(pares)} totales)\n"
        f"Verde oscuro = Dense Sargassum (cl.2) · Verde claro = Sparse Algae (cl.3)",
        fontsize=11, fontweight="bold"
    )

    for fila, (ip, mp) in enumerate(seleccion):
        img, mask = cargar_muestra(ip, mp)
        nombre    = os.path.basename(ip).replace(".npy", "")
        rgb = img[:, :, :3]
        nir = img[:, :, 3]

        ax_rgb, ax_gt, ax_over = axes[fila]

        # Sargassum en números
        n_dense  = int((mask == 2).sum())
        n_sparse = int((mask == 3).sum())
        total    = mask.size

        titulo = (f"{nombre[:35]}  |  "
                  f"Denso: {100*n_dense/total:.1f}%  "
                  f"Escaso: {100*n_sparse/total:.1f}%")
        ax_rgb.set_title(titulo, fontsize=8, loc="left")

        ax_rgb.imshow(rgb)
        ax_rgb.set_xlabel("RGB", fontsize=8)
        ax_rgb.axis("off")

        ax_gt.imshow(mask, cmap=CMAP_CLASES, norm=NORM_CLASES, interpolation="nearest")
        ax_gt.set_xlabel("Ground Truth", fontsize=8)
        ax_gt.axis("off")
        ax_gt.legend(handles=parches_leyenda(mask, CLASES_SARGASSUM),
                     fontsize=6.5, loc="lower right", framealpha=0.85,
                     title="Clases", title_fontsize=6)

        ax_over.imshow(rgb)
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        # Destacar solo las clases de sargazo con colores fuertes
        for cid, color_hex, alpha in [(2, "#1e8449", 0.75), (3, "#58d68d", 0.65)]:
            if (mask == cid).any():
                hex_c = color_hex.lstrip("#")
                r, g, b = [int(hex_c[i:i+2], 16)/255 for i in (0,2,4)]
                overlay[mask == cid] = [r, g, b, alpha]
        ax_over.imshow(overlay, interpolation="nearest")
        ax_over.set_xlabel("RGB + Sargazo resaltado", fontsize=8)
        ax_over.axis("off")

    plt.tight_layout()
    out = f"sargazo_{split}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  [OK] Guardado: {out}")
    plt.show()


# ══════════════════════════════════════════════════════════════
# SECCIÓN 4 — ESTADÍSTICAS GLOBALES (LOS 3 SPLITS)
# ══════════════════════════════════════════════════════════════

def estadisticas_globales():
    print(f"\n{'═'*60}")
    print(f"  SECCIÓN 4 — ESTADÍSTICAS GLOBALES (train / val / test)")
    print(f"{'═'*60}")

    datos = {}  # split → array de píxeles por clase

    for split in ("train", "val", "test"):
        mask_dir   = os.path.join(DATASET_DIR, split, "masks")
        mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.npy")))
        if not mask_paths:
            print(f"  [AVISO] Sin máscaras en {split}")
            continue

        conteo = np.zeros(NUM_CLASSES, dtype=np.int64)
        for mp in tqdm(mask_paths, desc=f"  Contando {split}", leave=False):
            m = np.load(mp).astype(np.int32)
            for c in range(NUM_CLASSES):
                conteo[c] += int((m == c).sum())
        datos[split] = conteo

        total = conteo.sum()
        print(f"\n  {split.upper()} — {len(mask_paths)} tiles  |  {total:,} píxeles totales")
        print(f"  {'ID':>3}  {'Clase':<28}  {'Píxeles':>12}  {'%':>7}")
        print(f"  {'-'*55}")
        for c in range(NUM_CLASSES):
            if conteo[c] > 0:
                marca = " ★" if c in CLASES_SARGASSUM else ""
                print(f"  {c:>3}  {CLASES[c][0]:<28}{marca}  "
                      f"{conteo[c]:>12,}  {100*conteo[c]/total:>6.2f}%")

    if not datos:
        return

    # ── Gráfico comparativo train / val / test ──
    splits_disp = list(datos.keys())
    n_splits    = len(splits_disp)
    x           = np.arange(NUM_CLASSES)
    ancho       = 0.25

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # Panel superior: todas las clases
    ax_all = axes[0]
    for i, sp in enumerate(splits_disp):
        total = datos[sp].sum()
        porcs = 100 * datos[sp] / total
        ax_all.bar(x + i * ancho, porcs, ancho,
                   label=sp.upper(), alpha=0.85, edgecolor="grey", linewidth=0.3)

    ax_all.set_xticks(x + ancho)
    ax_all.set_xticklabels(
        [f"{c}\n{CLASES[c][0][:10]}" for c in range(NUM_CLASSES)],
        fontsize=7, rotation=30, ha="right")
    ax_all.set_ylabel("% píxeles totales del split")
    ax_all.set_title("Distribución de clases por split (todas las clases)", fontsize=11)
    ax_all.legend(fontsize=9)
    ax_all.grid(axis="y", alpha=0.3)

    # Panel inferior: solo clases de interés (sin clase 0 y sin Marine Water)
    clases_zoom = [c for c in range(NUM_CLASSES) if c not in {0, 7}]
    ax_zoom = axes[1]
    x2 = np.arange(len(clases_zoom))
    for i, sp in enumerate(splits_disp):
        total = datos[sp].sum()
        porcs = [100 * datos[sp][c] / total for c in clases_zoom]
        ax_zoom.bar(x2 + i * ancho, porcs, ancho,
                    label=sp.upper(), alpha=0.85, edgecolor="grey", linewidth=0.3)

    ax_zoom.set_xticks(x2 + ancho)
    ax_zoom.set_xticklabels(
        [f"{c}: {CLASES[c][0][:12]}" for c in clases_zoom],
        fontsize=8, rotation=35, ha="right")
    ax_zoom.set_ylabel("% píxeles totales del split")
    ax_zoom.set_title("Zoom — clases minoritarias (sin Non-annotated ni Marine Water)",
                      fontsize=11)
    ax_zoom.legend(fontsize=9)
    ax_zoom.grid(axis="y", alpha=0.3)

    # Marcar barras de sargazo
    for ax in (ax_all, ax_zoom):
        for patch in ax.patches:
            pass  # matplotlib no expone a qué clase pertenece cada patch fácilmente

    plt.tight_layout()
    out = "estadisticas_globales.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  [OK] Guardado: {out}")
    plt.show()


# ══════════════════════════════════════════════════════════════
# SECCIÓN 5 — INSPECCIÓN PROFUNDA DE UN TILE (4 BANDAS)
# ══════════════════════════════════════════════════════════════

def inspeccionar_bandas(split: str, idx: int):
    """
    Muestra las 4 bandas espectrales crudas (sin normalizar) de un tile
    junto con la máscara GT. Útil para verificar que los valores de reflectancia
    son correctos antes de entrenar.
    """
    print(f"\n{'═'*60}")
    print(f"  SECCIÓN 5 — INSPECCIÓN DE BANDAS (tile [{idx}] de {split.upper()})")
    print(f"{'═'*60}")

    pares = listar_pares(split)
    if not pares:
        print("[ERROR] No hay tiles.")
        return

    ip, mp = pares[idx % len(pares)]
    nombre = os.path.basename(ip).replace(".npy", "")

    img_raw  = np.load(ip).astype(np.float32)   # SIN normalizar
    mask_raw = np.load(mp).astype(np.int32)

    # Center crop para consistencia
    TARGET = 224
    h, w   = mask_raw.shape
    y0 = (h - TARGET) // 2
    x0 = (w - TARGET) // 2
    img_raw  = img_raw [y0:y0+TARGET, x0:x0+TARGET, :]
    mask_raw = mask_raw[y0:y0+TARGET, x0:x0+TARGET]

    # Reordenar a (R,G,B,NIR) para el display
    img_reord = img_raw[:, :, [2, 1, 0, 3]]

    nombres_bandas = ["Azul (492 nm)", "Verde (560 nm)", "Rojo (665 nm)", "NIR (833 nm)"]
    cmaps_bandas   = ["Blues", "Greens", "Reds", "inferno"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(
        f"Inspección de bandas crudas — {nombre}\n"
        f"Fila superior: bandas individuales (valores rhorc)  |  "
        f"Fila inferior: composiciones y GT",
        fontsize=11, fontweight="bold"
    )

    # Fila 1: 4 bandas individuales
    for i, (nom, cmap) in enumerate(zip(nombres_bandas, cmaps_bandas)):
        banda = img_reord[:, :, i]
        ax = axes[0, i]
        im = ax.imshow(banda, cmap=cmap)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(nom, fontsize=9)
        ax.axis("off")
        # Estadísticas de la banda
        validos = banda[banda > 0]
        if validos.size > 0:
            ax.set_xlabel(
                f"min={banda.min():.4f}  max={banda.max():.4f}\n"
                f"media={banda.mean():.4f}  σ={banda.std():.4f}",
                fontsize=7
            )

    # Fila 2: RGB natural | NDVI-like | GT | RGB + GT
    # — RGB normalizado —
    img_norm = np.clip(img_reord[:, :, :3] / 10000.0 * 5.0, 0, 1) \
               if img_reord.max() > 10 \
               else np.clip(img_reord[:, :, :3] * 5.0, 0, 1)

    axes[1, 0].imshow(img_norm)
    axes[1, 0].set_title("RGB normalizado (x5)", fontsize=9)
    axes[1, 0].axis("off")

    # — FAI (Floating Algae Index): NIR - (Red + (SWIR-Red)*coef)
    # Usamos aproximación: NIR - Red como proxy simple del índice de algas flotantes
    nir_band = img_reord[:, :, 3]
    red_band = img_reord[:, :, 2]
    fai = nir_band - red_band
    ax_fai = axes[1, 1]
    im_fai = ax_fai.imshow(fai, cmap="RdYlGn", vmin=np.percentile(fai, 2),
                            vmax=np.percentile(fai, 98))
    plt.colorbar(im_fai, ax=ax_fai, fraction=0.046, pad=0.04)
    ax_fai.set_title("FAI proxy (NIR - Rojo)\nVerde = posible alga/sargazo", fontsize=8)
    ax_fai.axis("off")

    # — Ground Truth —
    axes[1, 2].imshow(mask_raw, cmap=CMAP_CLASES, norm=NORM_CLASES,
                      interpolation="nearest")
    axes[1, 2].set_title("Ground Truth", fontsize=9)
    axes[1, 2].axis("off")
    axes[1, 2].legend(handles=parches_leyenda(mask_raw, CLASES_SARGASSUM),
                      fontsize=6, loc="lower right", framealpha=0.85,
                      title="Clases", title_fontsize=6)

    # — RGB + GT superpuesto —
    axes[1, 3].imshow(img_norm)
    overlay = np.zeros((*mask_raw.shape, 4), dtype=np.float32)
    for cid in range(NUM_CLASSES):
        if cid == 7: continue
        hex_c = CLASES[cid][1].lstrip("#")
        r, g, b = [int(hex_c[i:i+2], 16)/255 for i in (0, 2, 4)]
        alpha = 0.7 if cid in CLASES_SARGASSUM else 0.3
        overlay[mask_raw == cid] = [r, g, b, alpha]
    axes[1, 3].imshow(overlay, interpolation="nearest")
    axes[1, 3].set_title("RGB + GT superpuesto", fontsize=9)
    axes[1, 3].axis("off")

    plt.tight_layout()
    out = f"bandas_{split}_{idx}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  [OK] Guardado: {out}")
    plt.show()


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════


def main():
    global DATASET_DIR  # primera linea obligatoria antes de cualquier uso

    parser = argparse.ArgumentParser(
        description="Verificacion y visualizacion completa del dataset MADOS preprocesado",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dataset", default="Sargassum_Ready_Dataset",
                        help="Ruta raiz del dataset (default: Sargassum_Ready_Dataset)")
    parser.add_argument("--solo", default=None,
                        choices=["integridad", "muestras", "sargassum",
                                 "estadisticas", "bandas"],
                        help="Ejecutar solo una seccion concreta")
    parser.add_argument("--split", default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=4,
                        help="Numero de muestras a mostrar (default: 4)")
    parser.add_argument("--idx", type=int, nargs="+", default=None,
                        help="Indices concretos de tiles a mostrar")
    args = parser.parse_args()

    # DESPUÉS
    DATASET_DIR = args.dataset if args.dataset != "Sargassum_Ready_Dataset" else str(SARGASSUM_READY)

    sep = chr(9608) * 60
    print("\n" + sep)
    print("  VERIFICADOR DEL DATASET MADOS")
    print("  Ruta: " + os.path.abspath(DATASET_DIR))
    print(sep)

    if not os.path.exists(DATASET_DIR):
        print("\n[ERROR FATAL] No se encuentra el dataset en: " + DATASET_DIR)
        sys.exit(1)

    solo = args.solo

    # Seccion 1: Integridad
    if solo in (None, "integridad"):
        verificar_integridad()

    # Seccion 2: Muestras generales
    if solo in (None, "muestras"):
        pares = listar_pares(args.split)
        if args.idx:
            indices = args.idx
        else:
            indices = random.sample(range(len(pares)), min(args.n, len(pares)))
        visualizar_muestras(args.split, indices)

    # Seccion 3: Tiles con sargassum
    if solo in (None, "sargassum"):
        visualizar_sargassum(args.split, n=args.n)

    # Seccion 4: Estadisticas globales
    if solo in (None, "estadisticas"):
        estadisticas_globales()

    # Seccion 5: Inspeccion de bandas
    if solo in (None, "bandas"):
        idx = args.idx[0] if args.idx else 0
        inspeccionar_bandas(args.split, idx)

    print("\n" + sep)
    print("  Verificacion completada.")
    print("  Imagenes guardadas en el directorio actual.")
    print(sep + "\n")


if __name__ == "__main__":
    main()