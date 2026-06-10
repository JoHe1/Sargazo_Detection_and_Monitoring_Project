# Detección y Monitorización de Sargazo mediante Imágenes Satelitales y Aprendizaje Profundo
**TFG · Grado en Ciencia e Ingeniería de Datos · ULPGC · 2026**  
**Autor:** Jorge Lorenzo Lorenzo  
**Tutores:** Javier Sánchez Pérez · Giovanny A. Cuervo Londoño

---

## Descripción del proyecto

El sargazo pelágico (*Sargassum fluitans* y *S. natans*) ha experimentado proliferaciones masivas en el Atlántico tropical desde 2011, provocando un impacto económico y ambiental severo en el litoral del Caribe, el Golfo de México y las Canarias. Su monitorización actual, basada en avistamiento visual o datos de baja frecuencia, no permite emitir alertas tempranas fiables.

Este proyecto se encarga de hacer un estudio de la viabilidad de las nuevas tecnologías de aprendizaje profundo para la detección y monitorización de sargazo utilizando imágenes satelitales Sentinel-2. El objetivo es desarrollar un sistema de segmentación semántica capaz de identificar la presencia de sargazo en zonas costeras, con especial atención a las clases de sargazo denso y algas flotantes dispersas.

El sistema distingue entre **16 clases marinas** definidas por el dataset MADOS, con foco especial en:
- **Clase 2** — Sargazo denso (*Dense Sargassum*)
- **Clase 3** — Algas flotantes dispersas (*Sparse Floating Algae*)

### Características principales

- Arquitectura **Swin Transformer + Attention U-Net** con 31.9M parámetros
- Test-Time Augmentation (TTA) con 8 transformaciones + majority voting
- Loss personalizado como **Focal Loss + Dice Loss**
- Uso del EMA (Exponential Moving Average) para estabilizar el entrenamiento
- Uso del VSCP (Virtual Sargassum Copy-Paster) para aumentar la presencia de sargazo en los batches

---

## Resultados del modelo final

El modelo final me dio unos resultados de exaustividad del 66.9%, precisión del 63,2%, el F1 del 65% y el IoU del 48.2% en las clases 2 y 3 de sargazo, sobre los datos de test que son 45, sobre los de Mados.El modelo encuentra gran cantidad de 

---

## Estructura del proyecto

```
Sargazo_Detection_and_Monitoring_Project/
│
├── train.py                          ← Entrenamiento principal
├── inference.py                      ← Inferencia y evaluación sobre MADOS
│
├── core/
│   ├── config/
│   │   ├── experiment_config.py      ← Hiperparámetros del experimento
│   │   └── paths.py                  ← Rutas del proyecto
│   ├── interfaces/
│   │   ├── base_dataset.py           ← ABC del dataset
|   |   ├── base_preprocessor.py     ← ABC del preprocesador
|   |   ├── base_trainer.py          ← ABC del entrenador
│   │   └── base_model.py             ← ABC del modelo
│   └── utils/
│       ├── metrics.py                ← F1, IoU, etc.
│       └── visualization.py          ← Paleta de colores MADOS
│
├── datasets/
│   ├── base/
│   │   └── base_dataset.py           ← Normalización y crop compartidos
│   ├── preprocessors/mados/
│   │   ├── mados_preprocessor.py         ← Preprocesador MADOS 4 bandas
│   │   ├── mados_preprocessor_swir.py    ← Preprocesador MADOS 6 bandas (+ SWIR)
│   │   └── mados_preprocessor_11bands.py ← Preprocesador MADOS 11 bandas
│   └── sources/
│       ├── mados_dataset.py              ← Dataset PyTorch 4 bandas (principal)
│       ├── mados_dataset_swir.py         ← Dataset PyTorch 6 bandas
│       └── mados_dataset_11bands.py      ← Dataset PyTorch 11 bandas
│
├── models/
│   ├── architectures/
│   │   ├── swin_transformer.py                        ← Swin-Tiny + UNet (baseline)
│   │   ├── swin_transformer_attention.py              ← Swin-Tiny + Attention U-Net 
│   │   ├── swin_transformer_attention_swir.py         ← Swin-Tiny + Attention U-Net + SWIR
│   │   ├── swin_transformer_attention_base_11bands.py ← Swin-Base + Attention U-Net + 11 bandas
│   │   └── segformer.py                               ← SegFormer mit-b2 (alternativa)
│   ├── losses/
│   │   ├── cross_entropy_dice.py         ← CE + Dice (v3, pesos sargazo=7)
│   │   ├── cross_entropy_dice_tversky.py ← CE + Dice + Tversky (reduce halo)
│   │   └── focal_dice.py                 ← Focal + Dice ✅ mejor loss
│   └── registry.py                       ← Registro de modelos
│
├── tools/
│   ├── analisis_distribucion.py      ← Análisis de clases del dataset
│   └── gt_editor.py                  ← Editor manual de Ground Truth
│
└── web/streamlit_app/
    ├── app.py                        ← Punto de entrada de la web
    ├── pages/
    │   ├── 1_monitor.py              ← Monitorización NRT (página principal)
    │   ├── 2_inference.py            ← Inferencia manual (.SAFE.zip o .npy)
    │   └── 3_compare.py              ← Comparativa de umbrales
    └── components/
        ├── sentinel_pipeline.py      ← Descarga y preprocesado Sentinel Hub
        ├── map_viewer.py             ← Mapa Folium con detecciones
        └── model_loader.py           ← Carga del modelo (cacheado)
```

---

## Descripción de los archivos principales

### Entrenamiento e inferencia

| Archivo | Descripción |
|---|---|
| `train.py` | Bucle de entrenamiento completo. Incluye EMA (α=0.995), early stopping por `val_loss`, scheduler `ReduceLROnPlateau`, VSCP a nivel de batch y soporte para los tres datasets (4/6/11 bandas). |
| `inference.py` | Evaluación sobre el split `val` o `test` de MADOS. Aplica TTA (8 transformaciones), suavizado gaussiano, limpieza de componentes pequeños y genera visualizaciones con las 16 clases. |

### Preprocesadores del dataset MADOS

| Archivo | Descripción |
|---|---|
| `mados_preprocessor.py` | Convierte los TIFs ACOLITE de MADOS a arrays `.npy` de 4 bandas (B, G, R, NIR). Se ejecuta una sola vez antes de entrenar. |
| `mados_preprocessor_swir.py` | Igual que el anterior pero añade SWIR1 (1610nm) y SWIR2 (2190nm) desde la carpeta `20/`, con upsample bilineal 2× a 10m. Genera arrays de 6 canales. |
| `mados_preprocessor_11bands.py` | Genera arrays de 11 canales con todas las bandas Sentinel-2 a 10m (bandas a 20m y 60m resampladas). Bandas ausentes se rellenan con ceros en lugar de descartar el tile. |

### Datasets PyTorch

| Archivo | Descripción |
|---|---|
| `mados_dataset.py` | Dataset principal de 4 bandas. Implementa VSCP a nivel de batch (MariNeXt, Kikaki et al. 2024), `WeightedRandomSampler` con peso 10× para tiles de sargazo, random/center crop y augmentations con Albumentations. |
| `mados_dataset_swir.py` | Hereda de `MADOSDataset`. Sobreescribe `_reorder_channels()` para el orden correcto de 6 canales: (R, G, B, NIR, SWIR1, SWIR2). |
| `mados_dataset_11bands.py` | Hereda de `MADOSDataset`. Reordena los 11 canales para que R, G, B queden en las posiciones 0-2 (compatibilidad con pesos ImageNet). |

### Funciones de pérdida

| Archivo | Descripción |
|---|---|
| `cross_entropy_dice.py` | CrossEntropy ponderada (sargazo w=7, agua w=2) + Dice solo en clases 2 y 3. `smooth=0.1` para Dice más estricto en bordes. |
| `cross_entropy_dice_tversky.py` | CE + Dice + Tversky. La Tversky con α=0.7, β=0.3 penaliza los Falsos Positivos más que los Falsos Negativos, reduciendo el halo de sobredetección. |
| `focal_dice.py` | **Loss del modelo final.** Focal Loss (γ=2.0) + Dice con label smoothing (ε=0.05). La Focal reduce el peso de píxeles fáciles (agua) y concentra el gradiente en sargazo difícil. |

### Arquitecturas

| Archivo | Descripción |
|---|---|
| `swin_transformer.py` | Baseline: Swin-Tiny + decoder UNet con skip connections y `ConvTranspose2d`. Adaptado para 4 canales (canal NIR inicializado con la media de los pesos RGB de ImageNet). |
| `swin_transformer_attention.py` | **Arquitectura final del TFG.** Swin-Tiny + Attention U-Net. Las `AttentionGate` filtran las skip connections para suprimir píxeles de agua y reforzar bordes de sargazo. |
| `swin_transformer_attention_swir.py` | Igual que el anterior pero con 6 canales (+ SWIR). Los canales SWIR se inicializan con los pesos del canal NIR. |
| `swin_transformer_attention_base_11bands.py` | Swin-Base (87M parámetros) + Attention U-Net + Stage-1 HR Features (MariNeXt) + 11 bandas. Mayor capacidad pero mayor coste computacional. |
| `segformer.py` | Alternativa con backbone SegFormer mit-b2 y decoder MLP nativo. Menos halo que UNet por el upsample bilineal directo desde cada escala. |

### Aplicación web

| Archivo | Descripción |
|---|---|
| `app.py` | Punto de entrada de Streamlit. Define la estructura de 3 páginas y la barra lateral de navegación. |
| `1_monitor.py` | Página principal de monitorización NRT. Descarga la imagen Sentinel-2 más reciente de la zona configurada, infiere con TTA, aplica land mask y muestra el mapa Folium con las detecciones. |
| `2_inference.py` | Inferencia manual. Acepta un archivo `.SAFE.zip` (producto L1C de Copernicus Browser) o un `.npy` preprocesado. Divide automáticamente la región en patches 224×224 y muestra los resultados por patch. |
| `sentinel_pipeline.py` | Descarga imágenes de Sentinel Hub Process API. EvalScript en DN → ρtoa, orden MADOS (B, G, R, NIR), normalización ×5 alineada con `base_dataset._normalize`. |
| `map_viewer.py` | Construye el mapa Folium con las detecciones superpuestas como capas RGBA georreferenciadas. |
| `model_loader.py` | Carga el modelo desde `weights.pth` con `st.cache_resource` para que solo se cargue una vez por sesión. |

---

## Instalación

```bash
git clone https://github.com/tu-usuario/Sargazo_Detection_and_Monitoring_Project.git
cd Sargazo_Detection_and_Monitoring_Project
```

---

## Cómo ejecutar

### 1. Preprocesar el dataset MADOS

Descarga el dataset MADOS desde [zenodo.org/record/7229756](https://zenodo.org/record/7229756) y configura las rutas en `core/config/paths.py`.

```bash
# Dataset estándar (4 bandas)
python datasets/preprocessors/mados/mados_preprocessor.py

# Dataset con bandas SWIR (6 bandas) (Opcional)
python datasets/preprocessors/mados/mados_preprocessor_swir.py

# Dataset completo 11 bandas (Opcional)
python datasets/preprocessors/mados/mados_preprocessor_11bands.py
```

### 2. Entrenar

```bash
# Entrenamiento con configuración por defecto (Swin-Tiny + Attention U-Net + FocalDice)
python train.py

# Con argumentos personalizados
python train.py --model swin_transformer_attention --epochs 50 --lr 5e-5 --batch-size 8
```

Los checkpoints se guardan en `experiments/runs/<nombre_experimento>/`.

### 3. Evaluar sobre el dataset MADOS

```bash
# Evaluación sobre val (por defecto)
python inference.py

# Evaluación final sobre test
python inference.py --split test

# Con opciones adicionales
python inference.py --split test --n 10 --umbral 0.95 --modelo experiments/runs/mi_experimento/
```

### 4. Aplicación web NRT

La Aplicación web es sumamente una demostración de concepto y no está optimizada para producción. Para usarla, ejecuta:

```bash
streamlit run web/streamlit_app/app.py
```

---

## Dataset MADOS

[MADOS (Marine Debris and Oil Spill Dataset)](https://zenodo.org/record/7229756) es un dataset de segmentación semántica marina con **16 clases** anotadas manualmente sobre imágenes Sentinel-2 procesadas con ACOLITE.

| ID | Clase |
|---|---|
| 0 | Non-annotated |
| 1 | Marine Debris |
| **2** | **Dense Sargassum** ← objetivo principal |
| **3** | **Sparse Floating Algae** ← objetivo principal |
| 4 | Natural Organic Material |
| 5 | Ship |
| 6 | Oil Spill |
| 7 | Marine Water |
| 8 | Sediment-Laden Water |
| 9 | Foam |
| 10 | Turbid Water |
| 11 | Shallow Water |
| 12 | Waves & Wakes |
| 13 | Oil Platform |
| 14 | Kelp |
| 15 | Coastline |

---

## Referencia

Si usas este trabajo, por favor cita:

```
Lorenzo Lorenzo, J. (2026). Detección y Monitorización de Sargazo mediante
Imágenes Satelitales Sentinel-2 y Aprendizaje Profundo.
TFG, Grado en Ciencia e Ingeniería de Datos, ULPGC.
```

**Dataset MADOS:**
```
Kikaki, K., Kakogeorgiou, I., Mikeli, P., Raitsos, D. E., & Karantzalos, K. (2022).
MADOS: Spotting Marine Debris, Oil Spills and Sargassum in the Ocean.
Zenodo. https://doi.org/10.5281/zenodo.7229756
```

---

## Licencia

Este proyecto es un Trabajo de Fin de Grado académico. Los pesos pre-entrenados de Swin Transformer pertenecen a Microsoft y están sujetos a su [licencia MIT](https://github.com/microsoft/Swin-Transformer/blob/main/LICENSE).