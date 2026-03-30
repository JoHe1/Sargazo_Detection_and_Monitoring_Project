"""
core/utils/visualization.py
-----------------------------
Helpers de visualización reutilizables en todo el proyecto.

Funciones disponibles:
    show_prediction()       — imagen + GT + predicción en 3 paneles
    show_spectral_bands()   — las 4 bandas espectrales de un tile
    show_class_distribution() — histograma de clases en un split
    overlay_mask()          — superpone una máscara coloreada sobre una imagen RGB
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

# Mapping oficial MADOS — se puede sobreescribir pasando class_map a cada función
MADOS_CLASSES: dict[int, tuple[str, str]] = {
    0:  ("Non-annotated",            "#1a1a1a"),
    1:  ("Marine Debris",            "#e74c3c"),
    2:  ("Dense Sargassum",          "#1e8449"),
    3:  ("Sparse Floating Algae",    "#58d68d"),
    4:  ("Natural Organic Material", "#f39c12"),
    5:  ("Ship",                     "#8e44ad"),
    6:  ("Oil Spill",                "#17202a"),
    7:  ("Marine Water",             "#2e86c1"),
    8:  ("Sediment-Laden Water",     "#d4ac0d"),
    9:  ("Foam",                     "#f2f3f4"),
    10: ("Turbid Water",             "#5499c7"),
    11: ("Shallow Water",            "#a9cce3"),
    12: ("Waves & Wakes",            "#abebc6"),
    13: ("Oil Platform",             "#922b21"),
    14: ("Jellyfish",                "#f1948a"),
    15: ("Sea Snot",                 "#b7950b"),
}


def _build_cmap(class_map: dict) -> tuple[ListedColormap, BoundaryNorm]:
    """Construye colormap y norma a partir de un dict de clases."""
    n = max(class_map.keys()) + 1
    colors = [class_map.get(i, ("?", "#888888"))[1] for i in range(n)]
    cmap   = ListedColormap(colors)
    norm   = BoundaryNorm(np.arange(-0.5, n + 0.5, 1), cmap.N)
    return cmap, norm


def overlay_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    class_map: dict = MADOS_CLASSES,
    highlight_classes: set | None = None,
    alpha_highlight: float = 0.7,
    alpha_rest: float = 0.3,
    ignore_classes: set | None = None,
) -> np.ndarray:
    """
    Superpone una máscara semántica coloreada sobre una imagen RGB.

    Args:
        image_rgb:         imagen normalizada (H, W, 3) con valores en [0, 1]
        mask:              máscara (H, W) con IDs de clase
        class_map:         dict {id: (nombre, hex_color)}
        highlight_classes: clases a resaltar con mayor opacidad (ej: {2, 3} para sargazo)
        alpha_highlight:   opacidad para clases resaltadas
        alpha_rest:        opacidad para el resto de clases
        ignore_classes:    clases que NO se pintan (ej: {7} para agua, {0} para fondo)

    Returns:
        imagen RGBA (H, W, 4) con la máscara superpuesta
    """
    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    ignore_classes = ignore_classes or set()

    for cid, (_, hex_color) in class_map.items():
        if cid in ignore_classes:
            continue
        r, g, b = [int(hex_color.lstrip("#")[i:i+2], 16) / 255 for i in (0, 2, 4)]
        alpha   = alpha_highlight if (highlight_classes and cid in highlight_classes) else alpha_rest
        overlay[mask == cid] = [r, g, b, alpha]

    return overlay


def show_prediction(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    class_map: dict = MADOS_CLASSES,
    highlight_classes: set | None = None,
    title: str = "",
    save_path: str | Path | None = None,
) -> None:
    """
    Muestra imagen original, GT y predicción en 3 paneles lado a lado.

    Args:
        image:             array (H, W, 4) o (H, W, 3) normalizado en [0, 1]
        ground_truth:      máscara GT (H, W) con IDs de clase
        prediction:        máscara predicha (H, W) con IDs de clase
        class_map:         dict de clases para colorear
        highlight_classes: clases a resaltar en la leyenda (ej: sargazo)
        title:             título general de la figura
        save_path:         si se especifica, guarda la figura en esa ruta
    """
    cmap, norm = _build_cmap(class_map)

    # Construir imagen RGB para mostrar (usa canales R, G, B)
    if image.shape[2] >= 3:
        rgb = np.clip(image[:, :, :3], 0, 1)
    else:
        rgb = np.stack([image[:, :, 0]] * 3, axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")

    axes[0].imshow(rgb)
    axes[0].set_title("Imagen (RGB)", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(ground_truth, cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title("Ground Truth", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(prediction, cmap=cmap, norm=norm, interpolation="nearest")
    axes[2].set_title("Predicción", fontsize=10)
    axes[2].axis("off")

    # Leyenda con las clases presentes
    present = set(np.unique(ground_truth).tolist()) | set(np.unique(prediction).tolist())
    patches = []
    for cid in sorted(present):
        if cid not in class_map:
            continue
        name, color = class_map[cid]
        star  = "★ " if (highlight_classes and cid in highlight_classes) else ""
        edge  = "gold" if (highlight_classes and cid in highlight_classes) else "none"
        patches.append(mpatches.Patch(
            facecolor=color, edgecolor=edge, linewidth=1.5,
            label=f"{star}{cid}: {name}"
        ))
    if patches:
        axes[2].legend(handles=patches, fontsize=7, loc="lower right",
                       framealpha=0.85, title="Clases presentes", title_fontsize=7)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"[visualization] Guardado: {save_path}")

    plt.show()
    plt.close(fig)


def show_spectral_bands(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    band_names: list[str] | None = None,
    title: str = "",
    save_path: str | Path | None = None,
) -> None:
    """
    Muestra las bandas espectrales de un tile junto con su GT opcional.

    Args:
        image:      array (H, W, C) con las bandas espectrales (sin normalizar o normalizado)
        mask:       máscara GT opcional (H, W). Si se pasa, se muestra como panel extra.
        band_names: nombres de las bandas. Por defecto ["B", "G", "R", "NIR"] para Sentinel-2
        title:      título de la figura
        save_path:  ruta para guardar la figura
    """
    if band_names is None:
        band_names = [f"Banda {i}" for i in range(image.shape[2])]
        if image.shape[2] == 4:
            band_names = ["Azul (492nm)", "Verde (560nm)", "Rojo (665nm)", "NIR (833nm)"]

    n_bands  = image.shape[2]
    n_panels = n_bands + (1 if mask is not None else 0)
    cmaps    = ["Blues", "Greens", "Reds", "inferno"] + ["gray"] * 10

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")

    for i in range(n_bands):
        banda = image[:, :, i]
        axes[i].imshow(banda, cmap=cmaps[i])
        axes[i].set_title(band_names[i], fontsize=9)
        axes[i].axis("off")
        axes[i].set_xlabel(
            f"min={banda.min():.4f}  max={banda.max():.4f}\nμ={banda.mean():.4f}",
            fontsize=7
        )

    if mask is not None:
        cmap, norm = _build_cmap(MADOS_CLASSES)
        axes[-1].imshow(mask, cmap=cmap, norm=norm, interpolation="nearest")
        axes[-1].set_title("Ground Truth", fontsize=9)
        axes[-1].axis("off")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=130, bbox_inches="tight")

    plt.show()
    plt.close(fig)