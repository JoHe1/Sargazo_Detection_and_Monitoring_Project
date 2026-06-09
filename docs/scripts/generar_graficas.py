"""
generar_graficas.py
--------------------
Genera todos los PDF vectoriales para el TFG de detección de sargazo.

Salida:
    graficas_pdf/
    ├── Exp01/
    │   ├── perdida_01_nombre.pdf         ← curva pérdida
    │   ├── miou_01_nombre.pdf            ← curva mIoU
    │   └── inference_1.pdf              ← barras Recall/Precision/F1/IoU
    ├── Exp02/ ...
    ├── Comparativas/
    │   ├── Comparativa_originales.pdf    ← vs Echevarría datos originales
    │   └── Comparativa_reentrenados.pdf  ← vs Echevarría reentrenado MADOS
    ├── umbral_095.pdf                   ← ablation todos los modelos umbral 0.95
    └── umbral_070.pdf                   ← ablation todos los modelos umbral 0.70

Uso:
    python generar_graficas.py
    python generar_graficas.py --runs docs/assets/data/runs --test docs/assets/data/test_final --out graficas_pdf
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Estilo global ────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'axes.edgecolor':    '#444444',
    'axes.labelcolor':   '#222222',
    'xtick.color':       '#222222',
    'ytick.color':       '#222222',
    'text.color':        '#222222',
    'grid.color':        '#dddddd',
    'grid.linestyle':    '--',
    'grid.linewidth':    0.6,
    'font.family':       'DejaVu Sans',
    'axes.titlesize':    14,
    'axes.labelsize':    13,
    'xtick.labelsize':   12,
    'ytick.labelsize':   12,
    'legend.fontsize':   12,
    'figure.dpi':        150,
})

COLORES = {
    'train':     '#3498db',
    'val':       '#e74c3c',
    'miou':      '#27ae60',
    'recall':    'rgba(52,152,219,.85)',
    'precision': 'rgba(231,76,60,.85)',
    'f1':        'rgba(39,174,96,.85)',
    'iou':       'rgba(243,156,18,.85)',
}

BAR_COLORS  = ['#3498db', '#e74c3c', '#27ae60', '#f39c12']
BAR_LABELS  = ['Recall', 'Precisión', 'F1', 'IoU']
BAR_EDGEC   = ['#2980b9', '#c0392b', '#1e8449', '#d68910']


# ── Utilidades ───────────────────────────────────────────────────────

def _filter_nan(epochs: list, values: list) -> tuple:
    """Elimina pares donde value es None o NaN — equivalente a spanGaps:true."""
    ep_clean, val_clean = [], []
    for e, v in zip(epochs, values):
        if v is not None and not math.isnan(v):
            ep_clean.append(e)
            val_clean.append(v)
    return ep_clean, val_clean


def load_json(path: Path) -> dict:
    """Lee un JSON probando múltiples encodings (utf-8, latin-1, cp1252)."""
    for enc in ('utf-8', 'utf-8-sig', 'latin-1', 'cp1252'):
        try:
            with open(path, 'r', encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError(f'No se pudo leer {path} con ningún encoding conocido')


def parse_csv(csv_path: Path) -> dict:
    epochs, train, val, miou = [], [], [], []
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    if not lines:
        return {}
    header = [h.strip() for h in lines[0].split(',')]
    try:
        idx_epoch = header.index('epoch')
        idx_train = header.index('train_loss')
        idx_val   = header.index('val_loss')
        idx_miou  = None
        for candidate in ['mIoU', 'miou', 'iou_sargassum_combinado']:
            if candidate in header:
                idx_miou = header.index(candidate)
                break
        if idx_miou is None:
            idx_miou = 6 if len(header) > 6 else 5
    except ValueError:
        idx_epoch, idx_train, idx_val, idx_miou = 0, 1, 2, 6

    for line in lines[1:]:
        if not line.strip():
            continue
        v = line.split(',')
        try:
            epochs.append(int(float(v[idx_epoch])))
            train.append(float(v[idx_train]) if v[idx_train].strip() else None)
            val.append(float(v[idx_val])     if v[idx_val].strip()   else None)
            m = v[idx_miou].strip() if idx_miou < len(v) else ''
            miou.append(float(m) * 100 if m else None)
        except (ValueError, IndexError):
            continue
    return {'epochs': epochs, 'train': train, 'val': val, 'miou': miou}


def get_eval_metrics(eval_path):
    """Lee un JSON de evaluación y devuelve las métricas principales."""
    if eval_path is None or not eval_path.exists():
        return None
    try:
        data = load_json(eval_path)
        key  = next((k for k in data if k != '_experimento'), None)
        if not key:
            return None
        d = data[key]
        recall    = d.get('recall')
        precision = d.get('precision')
        f1        = d.get('f1')
        iou       = d.get('iou_sargazo_global')
        # Validar que al menos f1 tenga valor
        if f1 is None:
            return None
        return {
            'recall':    recall,
            'precision': precision,
            'f1':        f1,
            'iou':       iou,
        }
    except Exception as e:
        print(f'    [WARN] Error leyendo {eval_path.name}: {e}')
        return None


# ── Gráficas de curvas ───────────────────────────────────────────────

def plot_loss(data: dict, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    epochs = data['epochs']
    ep_tr,  train = _filter_nan(epochs[:len(data['train'])], data['train'])
    ep_val, val   = _filter_nan(epochs[:len(data['val'])],   data['val'])
    ax.plot(ep_tr,  train, color=COLORES['train'], linewidth=2.5, label='Train Loss', zorder=3)
    ax.plot(ep_val, val,   color=COLORES['val'],   linewidth=2.5, label='Val Loss',   zorder=3)
    ax.set_xlabel('Época',   fontsize=14, fontweight='bold')
    ax.set_ylabel('Pérdida', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=13, pad=10)
    ax.legend(framealpha=0.9)
    ax.grid(True, zorder=0)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)


def plot_miou(data: dict, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    epochs = data['epochs']
    ep_miou, miou = _filter_nan(epochs[:len(data['miou'])], data['miou'])
    ax.fill_between(ep_miou, miou, color=COLORES['miou'], alpha=0.12, zorder=2)
    ax.plot(ep_miou, miou, color=COLORES['miou'], linewidth=2.5, label='mIoU (%)', zorder=3)
    ax.set_xlabel('Época',            fontsize=14, fontweight='bold')
    ax.set_ylabel('mIoU Sargazo (%)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_ylim(bottom=0)
    ax.legend(framealpha=0.9)
    ax.grid(True, zorder=0)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)


# ── Gráfica de barras de inferencia (1 modelo) ──────────────────────

def plot_inference(metrics: dict, title: str, out_path: Path) -> None:
    """Barras Recall / Precisión / F1 / IoU para un solo modelo."""
    valores = [
        metrics.get('recall')    or 0,
        metrics.get('precision') or 0,
        metrics.get('f1')        or 0,
        metrics.get('iou')       or 0,
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(BAR_LABELS))
    bars = ax.bar(x, valores, color=BAR_COLORS, edgecolor=BAR_EDGEC,
                  linewidth=1.2, width=0.55, zorder=3)

    # Etiquetas sobre las barras
    for bar, v in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{v*100:.1f}%',
                ha='center', va='bottom',
                fontsize=12, fontweight='bold', color='#222')

    ax.set_xticks(x)
    ax.set_xticklabels(BAR_LABELS, fontsize=13, fontweight='bold')
    ax.set_ylabel('Valor', fontsize=13, fontweight='bold')
    ax.set_ylim(0, min(1.0, max(valores) * 1.25 + 0.05))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v*100:.0f}%'))
    ax.tick_params(axis='y', labelsize=12)
    ax.set_title(title, fontsize=13, pad=10)
    ax.grid(axis='y', zorder=0)
    plt.tight_layout()
    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)


# ── Gráfica comparativa multi-modelo ────────────────────────────────

def plot_comparativa_simple(models_data: dict, title: str, out_path: Path) -> None:
    """
    Barras agrupadas con nombres completos en eje X.
    Usada para Comparativa_originales y Comparativa_reentrenados.
    """
    nombres  = list(models_data.keys())
    n        = len(nombres)
    metricas = ['recall', 'precision', 'f1', 'iou']
    n_met    = len(metricas)

    x      = np.arange(n)
    width  = 0.18
    offset = np.linspace(-(n_met-1)/2 * width, (n_met-1)/2 * width, n_met)

    fig, ax = plt.subplots(figsize=(max(10, n * 1.8), 5.5))

    for i, (met, lbl, col, edge) in enumerate(zip(metricas, BAR_LABELS, BAR_COLORS, BAR_EDGEC)):
        vals = [models_data[m].get(met) or 0 for m in nombres]
        ax.bar(x + offset[i], vals, width, label=lbl,
               color=col, edgecolor=edge, linewidth=0.8, zorder=3)

    short = [n[:22] + '…' if len(n) > 24 else n for n in nombres]
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=25, ha='right', fontsize=11, fontweight='bold')
    ax.set_ylabel('Valor', fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v*100:.0f}%'))
    ax.set_ylim(0, 1.15)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_title(title, fontsize=13, pad=10)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(axis='y', zorder=0)
    plt.tight_layout()
    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)


def plot_comparativa(models_data: dict, title: str, out_path: Path) -> None:
    """
    Barras agrupadas para múltiples modelos.
    Nombres completos en eje X rotados, figura cuadrada para que se vea bien en Overleaf.
    """
    nombres  = list(models_data.keys())
    n        = len(nombres)
    metricas = ['recall', 'precision', 'f1', 'iou']
    n_met    = len(metricas)

    x      = np.arange(n)
    width  = 0.18
    offset = np.linspace(-(n_met-1)/2 * width, (n_met-1)/2 * width, n_met)

    # Figura más alta para que los nombres del eje X tengan espacio
    fig, ax = plt.subplots(figsize=(max(12, n * 1.4), 8))

    for i, (met, lbl, col, edge) in enumerate(zip(metricas, BAR_LABELS, BAR_COLORS, BAR_EDGEC)):
        vals = [models_data[m].get(met) or 0 for m in nombres]
        ax.bar(x + offset[i], vals, width, label=lbl,
               color=col, edgecolor=edge, linewidth=0.8, zorder=3)

    # Nombres completos en eje X rotados 40 grados
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, rotation=40, ha='right',
                       fontsize=18, fontweight='bold')
    ax.set_ylabel('Valor', fontsize=20, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v*100:.0f}%'))
    ax.set_ylim(0, 1.15)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_title(title, fontsize=18, pad=14)
    ax.legend(fontsize=16, framealpha=0.9, loc='upper right',
              prop={'weight': 'bold', 'size': 16})
    ax.grid(axis='y', zorder=0)

    # Margen inferior extra para que los nombres no se corten
    plt.subplots_adjust(bottom=0.35)
    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Genera PDFs vectoriales de curvas y métricas para el TFG.'
    )
    parser.add_argument('--runs', default='docs/assets/data/runs',
                        help='Carpeta runs con index.json')
    parser.add_argument('--test', default='docs/assets/data/test_final',
                        help='Carpeta test_final con JSONs de evaluación')
    parser.add_argument('--out',  default='graficas_pdf',
                        help='Carpeta de salida')
    args = parser.parse_args()

    runs_dir = Path(args.runs)
    test_dir = Path(args.test)
    out_dir  = Path(args.out)

    # ── Validar rutas ────────────────────────────────────────────────
    index_path = runs_dir / 'index.json'
    if not index_path.exists():
        print(f'[ERROR] No se encontró: {index_path}')
        sys.exit(1)

    run_list = load_json(index_path)
    print(f'[OK] {len(run_list)} experimentos en {index_path}')
    print(f'[OK] Salida: {out_dir.resolve()}\n')

    errores = []

    # ════════════════════════════════════════════════════════════════
    # 1. Por cada experimento: pérdida + mIoU + inference
    # ════════════════════════════════════════════════════════════════
    print('─' * 60)
    print('  [1/4] Curvas de entrenamiento e inference por experimento')
    print('─' * 60)

    for i, run_name in enumerate(run_list, start=1):
        num      = str(i).zfill(2)
        exp_dir  = runs_dir / run_name
        carpeta  = out_dir / f'Exp{num}'
        carpeta.mkdir(parents=True, exist_ok=True)
        safe     = re.sub(r'[^a-zA-Z0-9_\-]', '_', run_name)
        titulo   = run_name.replace(f'{num}_', '', 1).replace('_', ' ')
        if 'BEST' in run_name.upper():
            titulo += ' ★'

        # ── Curvas pérdida y mIoU ────────────────────────────────
        csv_path = exp_dir / 'metrics.csv'
        if csv_path.exists():
            data = parse_csv(csv_path)
            if data and data.get('epochs'):
                try:
                    plot_loss(data, f'Curvas de pérdida — {titulo}',
                              carpeta / f'perdida_{safe}.pdf')
                    plot_miou(data, f'Evolución mIoU — {titulo}',
                              carpeta / f'miou_{safe}.pdf')
                except Exception as e:
                    print(f'  [{num}] ERROR curvas: {e}')
                    errores.append(run_name)
        else:
            print(f'  [{num}] SKIP — sin metrics.csv')

        # ── Inference (umbral 95 por defecto) ────────────────────
        # Buscar el JSON de evaluación — puede estar en el run o en test_final
        eval_path = exp_dir / 'evaluacion_test_umbral95.json'
        if not eval_path.exists():
            # Intentar también en test_final por si el mejor modelo está ahí
            alt = test_dir / 'evaluacion_test_umbral95.json'
            if alt.exists():
                eval_path = alt
            else:
                print(f'  [{num}] SKIP inference — no se encontró:')
                print(f'         {eval_path}')
                eval_path = None

        metrics = get_eval_metrics(eval_path) if eval_path else None
        if metrics:
            try:
                plot_inference(metrics,
                               f'Métricas de rendimiento — {titulo}',
                               carpeta / f'inference_{i}.pdf')
                print(f'  [{num}] OK — {run_name}')
            except Exception as e:
                print(f'  [{num}] ERROR inference: {e}')
                errores.append(run_name)
        elif eval_path:
            print(f'  [{num}] SKIP inference — archivo vacío o sin métricas')

    # ════════════════════════════════════════════════════════════════
    # 2. Comparativas con Echevarría
    # ════════════════════════════════════════════════════════════════
    print('\n' + '─' * 60)
    print('  [2/4] Comparativas con estado del arte')
    print('─' * 60)

    comp_dir = out_dir / 'Comparativas'
    comp_dir.mkdir(parents=True, exist_ok=True)

    # Cargar métricas del mejor modelo
    best_path = test_dir / 'evaluacion_swin_att_best.json'
    best_metrics = get_eval_metrics(best_path) if best_path.exists() else None
    if not best_metrics:
        print('  AVISO — sin evaluacion_swin_att_best.json, se omiten comparativas')
    else:
        swin_label = 'Swin-Att ★ (propuesto)'

        for sufijo, fname in [('echevarria', 'Comparativa_originales'),
                               ('echevarria_mados', 'Comparativa_reentrenados')]:
            ech_path = test_dir / f'evaluacion_{sufijo}.json'
            if not ech_path.exists():
                print(f'  SKIP — sin {ech_path.name}')
                continue
            try:
                ech_data = load_json(ech_path)
                models   = {swin_label: best_metrics}
                for nombre, vals in ech_data.items():
                    if nombre.startswith('_'):
                        continue
                    models[nombre] = {
                        'recall':    vals.get('recall'),
                        'precision': vals.get('precision'),
                        'f1':        vals.get('f1'),
                        'iou':       vals.get('iou_sargazo_global'),
                    }
                titulo_comp = ('Comparativa con Echevarría — datos originales'
                               if sufijo == 'echevarria'
                               else 'Comparativa con Echevarría — reentrenado MADOS')
                plot_comparativa_simple(models, titulo_comp, comp_dir / f'{fname}.pdf')
                print(f'  OK — {fname}.pdf ({len(models)} modelos)')
            except Exception as e:
                print(f'  ERROR {fname}: {e}')

    # ════════════════════════════════════════════════════════════════
    # 3. Ablation — umbral 0.95 y 0.70
    # ════════════════════════════════════════════════════════════════
    print('\n' + '─' * 60)
    print('  [3/4] Ablation study — umbrales 0.95 y 0.70')
    print('─' * 60)

    for umbral_str, umbral_val in [('95', '0.95'), ('70', '0.70')]:
        models = {}
        for i, run_name in enumerate(run_list, start=1):
            eval_path = runs_dir / run_name / f'evaluacion_test_umbral{umbral_str}.json'
            metrics   = get_eval_metrics(eval_path)
            if not metrics:
                continue
            label = run_name.replace(f'{str(i).zfill(2)}_', '', 1).replace('_', ' ')
            if 'BEST' in run_name.upper():
                label += ' ★'
            models[label] = metrics

        if models:
            fname = out_dir / f'umbral_0{umbral_str}.pdf'
            try:
                plot_comparativa(
                    models,
                    f'Ablation study — todos los modelos (umbral {umbral_val})',
                    fname
                )
                print(f'  OK — umbral_0{umbral_str}.pdf ({len(models)} modelos)')
            except Exception as e:
                print(f'  ERROR umbral_{umbral_str}: {e}')
        else:
            print(f'  SKIP umbral {umbral_val} — ningún run tiene evaluacion_test_umbral{umbral_str}.json')
            print(f'  Ruta buscada: {runs_dir}/<run_name>/evaluacion_test_umbral{umbral_str}.json')

    # ════════════════════════════════════════════════════════════════
    # Resumen
    # ════════════════════════════════════════════════════════════════
    print('\n' + '═' * 60)
    print(f'  Completado — PDFs en: {out_dir.resolve()}')
    if errores:
        print(f'  Errores en: {errores}')
    print('═' * 60)


if __name__ == '__main__':
    main()