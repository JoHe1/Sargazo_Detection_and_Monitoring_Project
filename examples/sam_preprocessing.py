import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import SamModel, SamProcessor
from PIL import Image

# ==========================================
# 1. CARGAR Y PREPARAR LA IMAGEN PARA SAM
# ==========================================
print("1. Cargando y procesando la imagen TIFF...")
imagen = tiff.imread("examples/recorte_sargazo_limpio.tiff")

# Normalizamos y extraemos canales para el Falso Color (NIR-R-G)
imagen_norm = np.clip(imagen, 0, 1)
canal_R = imagen_norm[:, :, 0]
canal_G = imagen_norm[:, :, 1]
canal_NIR = imagen_norm[:, :, 3]

falso_color = np.stack([canal_NIR, canal_R, canal_G], axis=-1)
falso_color_8bit = (falso_color * 255).astype(np.uint8)

# ¡NUEVO!: Hugging Face espera que la imagen sea un objeto PIL (Python Imaging Library)
raw_image = Image.fromarray(falso_color_8bit)

# ==========================================
# 2. CARGAR SAM DESDE HUGGING FACE
# ==========================================
print("2. Descargando/Cargando modelo SAM desde Hugging Face...")
device = "cuda" if torch.cuda.is_available() else "cpu" 

# Descarga automática de pesos y configuración
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# ==========================================
# 3. EL PROMPT: GUIAR A SAM AL SARGAZO (MÚLTIPLES PUNTOS)
# ==========================================
# 1. PUNTOS POSITIVOS (Sargazo = 1)
# Mantén aquí las coordenadas de las 6 estrellas que pusiste

puntos_positivos = [
    [110, 74.8],  # Tu punto central original (la balsa densa)
    [79.8, 92.1], 
    [48.5, 104.6],   
    [7.4, 108.5],    
    [148.9, 60.8],
    [166.7, 20.9]   
]

# Hugging Face espera la estructura: [Batch [Puntos_Imagen [[x1,y1], [x2,y2], ...]]]
etiquetas_positivas = [1] * len(puntos_positivos)

# 2. PUNTOS NEGATIVOS (Fondo/Ruido = 0)
# ¡LA SOLUCIÓN!: Añade 2 o 3 coordenadas que caigan justo en las manchas negras que cogió por error
puntos_negativos = [
    [74.2, 43.7],  # EJEMPLO: Coordenada de la mancha oscura de la izquierda
    [33.4, 37.1]   # EJEMPLO: Coordenada de la otra mancha oscura
]
etiquetas_negativas = [0] * len(puntos_negativos) # Todo esto es fondo (0)

# Juntamos las listas para dárselas a SAM
todos_los_puntos = puntos_positivos + puntos_negativos
todas_las_etiquetas = etiquetas_positivas + etiquetas_negativas

# Formato de Hugging Face
input_points = [[todos_los_puntos]]
input_labels = [[todas_las_etiquetas]] # Le pasamos también las etiquetas 1 y 0

print(f"3. Ejecutando inferencia con {len(puntos_positivos)} puntos de sargazo y {len(puntos_negativos)} de fondo...")

# Preparamos las entradas incluyendo el parámetro input_labels
inputs = processor(
    raw_image, 
    input_points=input_points, 
    input_labels=input_labels, 
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model(**inputs)

# ==========================================
# 4. POST-PROCESADO DE MÁSCARAS
# ==========================================
# SAM devuelve las máscaras en resolución baja, el processor las escala al tamaño original
masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), 
    inputs["original_sizes"].cpu(), 
    inputs["reshaped_input_sizes"].cpu()
)

# SAM siempre genera 3 máscaras posibles (diferentes niveles de detalle).
# Extraemos las puntuaciones (scores) de confianza para elegir la mejor.
scores = outputs.iou_scores.cpu().numpy()[0][0]
best_mask_idx = np.argmax(scores) # Buscamos el índice de la nota más alta

# Extraemos la mejor máscara como un array booleano (True/False)
mascara_final = masks[0][0][best_mask_idx].numpy()

# ==========================================
# 5. VISUALIZAR EL RESULTADO
# ==========================================
print("4. Generando visualización...")
fig, ejes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: Original + Prompts
ejes[0].imshow(falso_color_8bit)
# Dibujar estrellas cyan (Positivos)
for px, py in puntos_positivos:
    ejes[0].scatter(px, py, color='cyan', marker='*', s=100, edgecolor='white', linewidth=1)
# Dibujar cruces rojas (Negativos)
for nx, ny in puntos_negativos:
    ejes[0].scatter(nx, ny, color='red', marker='X', s=80, edgecolor='white', linewidth=1)
ejes[0].set_title(f"1. Prompts (Sargazo: *, Fondo: X)", fontsize=12)
ejes[0].axis('off')

# Panel 2: Ground Truth
ejes[1].imshow(mascara_final, cmap='gray')
ejes[1].set_title(f"2. Ground Truth (Confianza: {scores[best_mask_idx]:.2f})", fontsize=12)
ejes[1].axis('off')

# Panel 3: Overlay (Superposición)
ejes[2].imshow(falso_color_8bit)
color_mascara = np.zeros((224, 224, 4))
color_mascara[mascara_final] = [0, 1, 0, 0.5] # Verde transparente
ejes[2].imshow(color_mascara)
ejes[2].set_title("3. Overlay (Verificación)", fontsize=12)
ejes[2].axis('off')

plt.tight_layout()
plt.show()

# ==========================================
# 6. GUARDAR PARA ENTRENAMIENTO
# ==========================================
mascara_guardar = (mascara_final * 1).astype(np.uint8)
tiff.imwrite("mascara_sargazo_gt.tiff", mascara_guardar)
print("¡Completado! Máscara guardada como 'mascara_sargazo_gt.tiff'")