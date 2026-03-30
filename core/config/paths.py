"""
core/config/paths.py
----------------------
Rutas centralizadas del proyecto.

NUNCA escribas rutas hardcodeadas en otros archivos.
Siempre importa desde aquí:

    from core.config.paths import DATA_ROOT, CHECKPOINTS_DIR

Cómo funciona:
    Path(__file__) apunta a este archivo (core/config/paths.py)
    .parent      → core/config/
    .parent      → core/
    .parent      → raíz del proyecto  ← ROOT

    Así las rutas funcionan en cualquier máquina sin cambiar nada,
    independientemente de dónde esté clonado el repositorio.
"""

from pathlib import Path

# ── Raíz del proyecto ──────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
# Ejemplo: /home/jorge/proyectos/SARGAZO_DETECTION_PROJECT/

# ── Datos ──────────────────────────────────────────────────────────
DATA_ROOT     = ROOT / "datasets" / "data"
# Contiene: MADOS/, Sargassum_Ready_Dataset/, sentinel_downloads/, sentinel_caribbean/

MADOS_RAW_DIR    = DATA_ROOT / "MADOS" / "MADOS"
# Carpeta raíz de los TIFs crudos de MADOS (con subcarpetas por escena)

SARGASSUM_READY  = DATA_ROOT / "Sargassum_Ready_Dataset"
# Dataset ya preprocesado (.npy) con estructura train/val/test

SENTINEL_DIR     = DATA_ROOT / "sentinel_downloads"
# Imágenes descargadas de la API de Copernicus para inferencia

# ── Preprocesamiento ───────────────────────────────────────────────
PREPROCESSORS_DIR = ROOT / "datasets" / "preprocessors"
LAND_MASK_DIR     = PREPROCESSORS_DIR / "land_mask"
# Contiene: ne_10m_land.shp, ne_10m_minor_islands.shp, etc.

# ── Modelos ────────────────────────────────────────────────────────
MODELS_DIR       = ROOT / "models"
CHECKPOINTS_DIR  = ROOT / "experiments" / "checkpoints"
# Estructura dentro de checkpoints/:
#   {nombre_experimento}/
#       weights.pth
#       metadata.json

# ── Experimentos ───────────────────────────────────────────────────
EXPERIMENTS_DIR  = ROOT / "experiments"
RUNS_DIR         = EXPERIMENTS_DIR / "runs"
RESULTS_DIR      = EXPERIMENTS_DIR / "results"
CONFIGS_DIR      = EXPERIMENTS_DIR / "configs"

# ── Web / App ──────────────────────────────────────────────────────
DOCS_DIR         = ROOT / "docs"
# Carpeta para GitHub Pages (HTML/CSS/JS estático)


# ══════════════════════════════════════════════════════════════════
# Utilidad — verificar que las rutas críticas existen
# ══════════════════════════════════════════════════════════════════

def check_paths() -> None:
    """
    Verifica que las carpetas críticas existen en el disco.
    Llama a esta función al inicio de train.py para detectar
    problemas de configuración antes de empezar el entrenamiento.

    Uso:
        from core.config.paths import check_paths
        check_paths()
    """
    rutas_criticas = {
        "Raíz del proyecto":         ROOT,
        "Datos crudos":              DATA_ROOT,
        "MADOS crudo":               MADOS_RAW_DIR,
        "Sargassum Ready Dataset":   SARGASSUM_READY,
        "Land mask shapefiles":      LAND_MASK_DIR,
    }

    print("\n[paths] Verificando rutas del proyecto...")
    todo_ok = True
    for nombre, ruta in rutas_criticas.items():
        estado = "✔" if ruta.exists() else "✗ NO EXISTE"
        print(f"  {estado}  {nombre}: {ruta}")
        if not ruta.exists():
            todo_ok = False

    if todo_ok:
        print("[paths] Todas las rutas críticas existen.\n")
    else:
        print("[paths] AVISO: Algunas rutas no existen. Revisa la estructura de carpetas.\n")


if __name__ == "__main__":
    check_paths()