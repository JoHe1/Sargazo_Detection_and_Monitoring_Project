import tifffile as tiff
import numpy as np
import cv2
import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import glob
import os

# ==========================================
# 1. INICIALIZAR SAM
# ==========================================
print("Cargando modelo SAM (esto puede tardar unos segundos)...")
device = "cuda" if torch.cuda.is_available() else "cpu" 
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

lista_imagenes = sorted(glob.glob("dataset_generator/dataset/2_masked/*.tiff"))
if not lista_imagenes:
    print("No hay imágenes en la carpeta dataset_generator/dataset/2_masked/")
    exit()

indice_actual = 0
WINDOW_W, WINDOW_H = 1000, 800 
NOMBRE_VENTANA = "Generador de Dataset HIBRIDO (SAM + Pincel)"

# Variables de la imagen
raw_image = None
cv2_falso_color, cv2_color_real, cv2_ndvi = None, None, None
img_w, img_h = 128, 128

# Sistema de Máscaras
puntos, etiquetas = [], []
mascara_sam = np.zeros((128, 128), dtype=bool)
mascara_pincel_pos = np.zeros((128, 128), dtype=bool)
mascara_pincel_neg = np.zeros((128, 128), dtype=bool)
mascara_actual = np.zeros((128, 128), dtype=bool)

# ¡NUEVO! Sistema de Deshacer (Historial)
historial = []

# Variables de Interfaz y Estado
vista_actual = 0 
modo_pincel = False
is_painting_pos = False
is_painting_neg = False
radio_pincel = 0 

# Variables de navegación
scale = 6.0 
dx, dy = 0.0, 0.0
is_dragging = False
last_mouse_x, last_mouse_y = 0, 0
current_mouse_x, current_mouse_y = 0, 0

def guardar_estado():
    """Guarda una copia exacta de todas las máscaras y puntos antes de hacer un cambio"""
    global historial
    estado = (
        list(puntos), 
        list(etiquetas), 
        mascara_sam.copy(), 
        mascara_pincel_pos.copy(), 
        mascara_pincel_neg.copy()
    )
    historial.append(estado)
    # Guardamos solo los últimos 30 pasos para no saturar la memoria RAM
    if len(historial) > 30:
        historial.pop(0)

def cargar_imagen_actual():
    global raw_image, cv2_falso_color, cv2_color_real, cv2_ndvi, img_w, img_h
    global puntos, etiquetas, mascara_sam, mascara_pincel_pos, mascara_pincel_neg, historial
    global dx, dy, scale
    
    if indice_actual >= len(lista_imagenes):
        print("\n¡DATASET COMPLETADO! Has etiquetado todas las imágenes.")
        exit()
        
    ruta = lista_imagenes[indice_actual]
    print(f"\n[{indice_actual+1}/{len(lista_imagenes)}] Etiquetando: {os.path.basename(ruta)}")
    
    imagen = tiff.imread(ruta)
    img_h, img_w = imagen.shape[0], imagen.shape[1]
    
    R = np.clip(imagen[:, :, 0], 0, 1)
    G = np.clip(imagen[:, :, 1], 0, 1)
    B = np.clip(imagen[:, :, 2], 0, 1)
    NIR = np.clip(imagen[:, :, 3], 0, 1)
    NDVI = imagen[:, :, 4]
    
    falso_color = np.stack([NIR, R, G], axis=-1)
    color_real = np.stack([R, G, B], axis=-1)
    
    ndvi_norm = np.clip((NDVI + 1.0) / 2.0, 0, 1)
    ndvi_gris = (ndvi_norm * 255).astype(np.uint8)
    cv2_ndvi = cv2.applyColorMap(ndvi_gris, cv2.COLORMAP_JET)
    
    raw_image = Image.fromarray((falso_color * 255).astype(np.uint8))
    cv2_falso_color = cv2.cvtColor((falso_color * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2_color_real = cv2.cvtColor((color_real * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    puntos, etiquetas, historial = [], [], []
    mascara_sam = np.zeros((img_h, img_w), dtype=bool)
    mascara_pincel_pos = np.zeros((img_h, img_w), dtype=bool)
    mascara_pincel_neg = np.zeros((img_h, img_w), dtype=bool)
    
    dx = (WINDOW_W - (img_w * 6.0)) / 2
    dy = (WINDOW_H - (img_h * 6.0)) / 2
    scale = 6.0 
    
    actualizar_mascara_final()
    actualizar_pantalla()

def actualizar_mascara_final():
    global mascara_actual
    mascara_actual = (mascara_sam | mascara_pincel_pos) & ~mascara_pincel_neg

def ejecutar_sam():
    global mascara_sam
    if len(puntos) == 0:
        mascara_sam = np.zeros((img_h, img_w), dtype=bool)
        actualizar_mascara_final()
        return
        
    input_points = [[puntos]]
    input_labels = [[etiquetas]]
    inputs = processor(raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    best_mask_idx = np.argmax(outputs.iou_scores.cpu().numpy()[0][0])
    mascara_sam = masks[0][0][best_mask_idx].numpy().astype(bool)
    actualizar_mascara_final()

def aplicar_pincel(x, y, positivo):
    global mascara_pincel_pos, mascara_pincel_neg
    if 0 <= x < img_w and 0 <= y < img_h:
        plantilla = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.circle(plantilla, (x, y), radio_pincel, 1, thickness=-1)
        zona_pintada = (plantilla == 1)
        
        if positivo:
            mascara_pincel_pos = mascara_pincel_pos | zona_pintada
            mascara_pincel_neg = mascara_pincel_neg & ~zona_pintada 
        else:
            mascara_pincel_neg = mascara_pincel_neg | zona_pintada
            mascara_pincel_pos = mascara_pincel_pos & ~zona_pintada 
            
        actualizar_mascara_final()
        actualizar_pantalla()

def actualizar_pantalla():
    global cv2_falso_color, cv2_color_real, cv2_ndvi, mascara_actual, puntos, etiquetas
    global scale, dx, dy
    
    if vista_actual == 0: pantalla_base = cv2_falso_color.copy(); texto_vista = "VISTA: Falso Color (NIR)"
    elif vista_actual == 1: pantalla_base = cv2_color_real.copy(); texto_vista = "VISTA: Color Real (RGB)"
    else: pantalla_base = cv2_ndvi.copy(); texto_vista = "VISTA: Mapa de Calor NDVI"
    
    if mascara_actual is not None and np.any(mascara_actual):
        color_mask = np.zeros_like(pantalla_base)
        color_mask[mascara_actual] = [0, 255, 0] 
        cv2.addWeighted(color_mask, 0.5, pantalla_base, 1.0, 0, pantalla_base)
    
    M = np.float32([[scale, 0, dx], [0, scale, dy]])
    pantalla_final = cv2.warpAffine(pantalla_base, M, (WINDOW_W, WINDOW_H))
    
    for pt, label in zip(puntos, etiquetas):
        px = int(pt[0] * scale + dx)
        py = int(pt[1] * scale + dy)
        if label == 1: cv2.drawMarker(pantalla_final, (px, py), (255, 255, 0), cv2.MARKER_STAR, markerSize=8, thickness=1)
        else: cv2.drawMarker(pantalla_final, (px, py), (0, 0, 255), cv2.MARKER_CROSS, markerSize=8, thickness=1)
    
    cv2.putText(pantalla_final, f"Img {indice_actual+1}/{len(lista_imagenes)} - {texto_vista}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    modo_texto = "MODO: PINCEL MANUAL (Grosor: {})".format(radio_pincel) if modo_pincel else "MODO: NAVEGAR y SAM"
    color_modo = (0, 165, 255) if modo_pincel else (200, 255, 200) 
    
    cv2.putText(pantalla_final, modo_texto, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_modo, 2)
    
    if modo_pincel:
        cv2.putText(pantalla_final, "Clic Izd: Pintar | Clic Der: Borrar | [+ / -] Grosor", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        cv2.putText(pantalla_final, "[1] SAM + | [2] SAM - | Mover: Arrastrar imagen", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    # He añadido la Z en la leyenda
    cv2.putText(pantalla_final, "[V] Vista | [P] Pincel | [Z] Deshacer | [C] Limpiar | [S] Guardar", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.imshow(NOMBRE_VENTANA, pantalla_final)

def eventos_raton(event, x, y, flags, param):
    global scale, dx, dy, is_dragging, last_mouse_x, last_mouse_y, current_mouse_x, current_mouse_y
    global is_painting_pos, is_painting_neg
    
    current_mouse_x, current_mouse_y = x, y
    img_x = int((x - dx) / scale)
    img_y = int((y - dy) / scale)
    
    if modo_pincel:
        if event == cv2.EVENT_LBUTTONDOWN:
            guardar_estado() # ¡Guardamos estado justo antes de empezar el trazo!
            is_painting_pos = True; aplicar_pincel(img_x, img_y, True)
        elif event == cv2.EVENT_RBUTTONDOWN:
            guardar_estado() # ¡Guardamos estado justo antes de empezar a borrar!
            is_painting_neg = True; aplicar_pincel(img_x, img_y, False)
        elif event == cv2.EVENT_MOUSEMOVE:
            if is_painting_pos: aplicar_pincel(img_x, img_y, True)
            elif is_painting_neg: aplicar_pincel(img_x, img_y, False)
        elif event == cv2.EVENT_LBUTTONUP: is_painting_pos = False
        elif event == cv2.EVENT_RBUTTONUP: is_painting_neg = False
    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            is_dragging = True; last_mouse_x, last_mouse_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if is_dragging: 
                dx += (x - last_mouse_x); dy += (y - last_mouse_y)
                last_mouse_x, last_mouse_y = x, y
                actualizar_pantalla()
        elif event == cv2.EVENT_LBUTTONUP: is_dragging = False

    if event == cv2.EVENT_MOUSEWHEEL:
        zoom_factor = 1.1 if flags > 0 else 0.9
        dx = x - (x - dx) * zoom_factor
        dy = y - (y - dy) * zoom_factor
        scale *= zoom_factor
        actualizar_pantalla()

# ==========================================
# 4. BUCLE PRINCIPAL
# ==========================================
cv2.namedWindow(NOMBRE_VENTANA, cv2.WINDOW_NORMAL)
cv2.resizeWindow(NOMBRE_VENTANA, WINDOW_W, WINDOW_H)
cv2.setMouseCallback(NOMBRE_VENTANA, eventos_raton)

os.makedirs("dataset_generator/dataset/3_ground_truth", exist_ok=True)

cargar_imagen_actual() 

while True:
    tecla = cv2.waitKey(1) & 0xFF
    
    if tecla == ord('1') or tecla == ord('2'):
        if not modo_pincel: 
            img_x = int((current_mouse_x - dx) / scale)
            img_y = int((current_mouse_y - dy) / scale)
            if 0 <= img_x < img_w and 0 <= img_y < img_h:
                guardar_estado() # ¡Guardamos estado justo antes de añadir un punto SAM!
                puntos.append([img_x, img_y])
                etiquetas.append(1 if tecla == ord('1') else 0)
                ejecutar_sam()
                actualizar_pantalla()
                
    elif tecla == ord('z'): # ¡LÓGICA DEL DESHACER (UNDO)!
        if len(historial) > 0:
            estado_anterior = historial.pop()
            puntos = list(estado_anterior[0])
            etiquetas = list(estado_anterior[1])
            mascara_sam = estado_anterior[2].copy()
            mascara_pincel_pos = estado_anterior[3].copy()
            mascara_pincel_neg = estado_anterior[4].copy()
            actualizar_mascara_final()
            actualizar_pantalla()
            
    elif tecla == ord('v'): 
        vista_actual = (vista_actual + 1) % 3
        actualizar_pantalla()
        
    elif tecla == ord('p'): 
        modo_pincel = not modo_pincel
        is_dragging, is_painting_pos, is_painting_neg = False, False, False
        actualizar_pantalla()
        
    elif tecla == ord('+'): 
        radio_pincel += 1; actualizar_pantalla()
        
    elif tecla == ord('-'): 
        radio_pincel = max(0, radio_pincel - 1); actualizar_pantalla()
        
    elif tecla == ord('s'): 
        nombre_base = os.path.basename(lista_imagenes[indice_actual])
        nombre_mask = nombre_base.replace(".tiff", "_mask.tiff")
        ruta_guardado = f"dataset_generator/dataset/3_ground_truth/{nombre_mask}"
        
        mascara_guardar = (mascara_actual * 1).astype(np.uint8)
        tiff.imwrite(ruta_guardado, mascara_guardar)
        print(f"Guardado: {ruta_guardado}")
        
        indice_actual += 1
        cargar_imagen_actual()
        
    elif tecla == ord('c'): 
        guardar_estado() # También guardamos el estado antes de limpiar por si limpias sin querer
        puntos, etiquetas = [], []
        mascara_sam = np.zeros((img_h, img_w), dtype=bool)
        mascara_pincel_pos = np.zeros((img_h, img_w), dtype=bool)
        mascara_pincel_neg = np.zeros((img_h, img_w), dtype=bool)
        actualizar_mascara_final()
        actualizar_pantalla()
        
    elif tecla == ord('q'): 
        print("Saliendo del etiquetador...")
        break

cv2.destroyAllWindows()