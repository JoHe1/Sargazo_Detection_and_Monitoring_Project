import tifffile as tiff
import numpy as np
import cv2
import torch
from transformers import SamModel, SamProcessor
from PIL import Image

# ==========================================
# 1. CARGAR Y PREPARAR LAS IMÁGENES
# ==========================================
print("1. Cargando imagen TIFF...")
ruta_imagen = "examples/recorte_sargazo_enmascarado.tiff" 
imagen = tiff.imread(ruta_imagen)

img_h, img_w = imagen.shape[0], imagen.shape[1]

# Normalizamos
imagen_norm = np.clip(imagen, 0, 1)
canal_R = imagen_norm[:, :, 0]
canal_G = imagen_norm[:, :, 1]
canal_B = imagen_norm[:, :, 2]
canal_NIR = imagen_norm[:, :, 3]

# Vistas
falso_color = np.stack([canal_NIR, canal_R, canal_G], axis=-1)
falso_color_8bit = (falso_color * 255).astype(np.uint8)
raw_image = Image.fromarray(falso_color_8bit) 

color_real = np.stack([canal_R, canal_G, canal_B], axis=-1)
color_real_8bit = (color_real * 255).astype(np.uint8)

cv2_falso_color = cv2.cvtColor(falso_color_8bit, cv2.COLOR_RGB2BGR)
cv2_color_real = cv2.cvtColor(color_real_8bit, cv2.COLOR_RGB2BGR)

# ==========================================
# 2. CARGAR SAM
# ==========================================
print("2. Cargando modelo SAM...")
device = "cuda" if torch.cuda.is_available() else "cpu" 
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# ==========================================
# 3. VARIABLES GLOBALES (ESTADO Y NAVEGACIÓN)
# ==========================================
puntos = []
etiquetas = []
mascara_actual = None
mostrar_falso_color = True

# Variables para Zoom y Pan (Desplazamiento)
scale = 1.0
# Si la imagen es pequeña (224x224), empezamos con un zoom x3 para verla grande
if img_w < 500: scale = 3.0 
dx, dy = 0.0, 0.0
is_dragging = False
last_mouse_x, last_mouse_y = 0, 0
current_mouse_x, current_mouse_y = 0, 0

# Tamaño fijo de la ventana de la herramienta
WINDOW_W, WINDOW_H = 1000, 800 
NOMBRE_VENTANA = "Etiquetador Profesional SAM"

def ejecutar_sam():
    global puntos, etiquetas, mascara_actual
    if len(puntos) == 0:
        mascara_actual = None
        return

    input_points = [[puntos]]
    input_labels = [[etiquetas]]
    
    inputs = processor(raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    
    scores = outputs.iou_scores.cpu().numpy()[0][0]
    best_mask_idx = np.argmax(scores)
    mascara_actual = masks[0][0][best_mask_idx].numpy()

def actualizar_pantalla():
    global cv2_falso_color, cv2_color_real, mostrar_falso_color, mascara_actual, puntos, etiquetas
    global scale, dx, dy
    
    # 1. Dibujamos todo sobre la imagen original
    pantalla_base = cv2_falso_color.copy() if mostrar_falso_color else cv2_color_real.copy()
    
    if mascara_actual is not None:
        color_mask = np.zeros_like(pantalla_base)
        color_mask[mascara_actual] = [0, 255, 0] 
        cv2.addWeighted(color_mask, 0.5, pantalla_base, 1.0, 0, pantalla_base)
        
    for pt, label in zip(puntos, etiquetas):
        if label == 1:
            cv2.drawMarker(pantalla_base, tuple(pt), (255, 255, 0), cv2.MARKER_STAR, markerSize=10, thickness=1)
        else:
            cv2.drawMarker(pantalla_base, tuple(pt), (0, 0, 255), cv2.MARKER_CROSS, markerSize=10, thickness=1)
            
    # 2. APLICAR TRANSFORMACIÓN (ZOOM Y PAN)
    # Creamos la matriz de transformación: [[escala, 0, desplazamiento_X], [0, escala, desplazamiento_Y]]
    M = np.float32([[scale, 0, dx], [0, scale, dy]])
    pantalla_final = cv2.warpAffine(pantalla_base, M, (WINDOW_W, WINDOW_H))
    
    # 3. Dibujar la interfaz (Texto)
    texto_modo = "VISTA: Falso Color (NIR)" if mostrar_falso_color else "VISTA: Color Real (RGB)"
    cv2.putText(pantalla_final, texto_modo, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(pantalla_final, "[1] + Sargazo | [2] - Fondo", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
            
    cv2.imshow(NOMBRE_VENTANA, pantalla_final)

def eventos_raton(event, x, y, flags, param):
    global scale, dx, dy, is_dragging, last_mouse_x, last_mouse_y, current_mouse_x, current_mouse_y
    
    current_mouse_x, current_mouse_y = x, y
    
    # Arrastrar la imagen (Pan)
    if event == cv2.EVENT_LBUTTONDOWN:
        is_dragging = True
        last_mouse_x, last_mouse_y = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_dragging:
            dx += (x - last_mouse_x)
            dy += (y - last_mouse_y)
            last_mouse_x, last_mouse_y = x, y
            actualizar_pantalla()
            
    elif event == cv2.EVENT_LBUTTONUP:
        is_dragging = False
        
    # Zoom con la rueda del ratón
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            zoom_factor = 1.1 # Zoom In
        else:
            zoom_factor = 0.9 # Zoom Out
            
        # Matemáticas para hacer el zoom centrado en la posición del ratón
        dx = x - (x - dx) * zoom_factor
        dy = y - (y - dy) * zoom_factor
        scale *= zoom_factor
        actualizar_pantalla()

# ==========================================
# 4. BUCLE PRINCIPAL
# ==========================================
print("\n--- INSTRUCCIONES AVANZADAS ---")
print("🖱️  Rueda Ratón: Zoom In / Zoom Out")
print("🖱️  Clic Izquierdo (Mantenido): Arrastrar imagen")
print("⌨️  Presiona '1': Añadir punto Positivo (Sargazo) en el cursor")
print("⌨️  Presiona '2': Añadir punto Negativo (Fondo) en el cursor")
print("⌨️  Presiona 'V': Cambiar vista (RGB/NIR)")
print("⌨️  Presiona 'C': Limpiar | 'S': Guardar | 'Q': Salir")

cv2.namedWindow(NOMBRE_VENTANA, cv2.WINDOW_NORMAL)
cv2.resizeWindow(NOMBRE_VENTANA, WINDOW_W, WINDOW_H)
cv2.setMouseCallback(NOMBRE_VENTANA, eventos_raton)
actualizar_pantalla()

while True:
    tecla = cv2.waitKey(1) & 0xFF
    
    # PUNTOS VÍA TECLADO
    if tecla == ord('1') or tecla == ord('2'):
        # Convertimos las coordenadas de la pantalla (con zoom) a coordenadas de la imagen original
        img_x = int((current_mouse_x - dx) / scale)
        img_y = int((current_mouse_y - dy) / scale)
        
        # Comprobamos que el ratón esté dentro de la imagen
        if 0 <= img_x < img_w and 0 <= img_y < img_h:
            puntos.append([img_x, img_y])
            etiquetas.append(1 if tecla == ord('1') else 0)
            ejecutar_sam()
            actualizar_pantalla()
    
    elif tecla == ord('v'): # Cambiar Vista
        mostrar_falso_color = not mostrar_falso_color
        actualizar_pantalla()
        
    elif tecla == ord('s'): # Guardar
        if mascara_actual is not None:
            mascara_guardar = (mascara_actual * 1).astype(np.uint8)
            tiff.imwrite("examples/mascara_sargazo_gt.tiff", mascara_guardar)
            print("\n¡Éxito! Ground Truth guardado.")
        else:
            print("\nError: No hay máscara.")
        break
        
    elif tecla == ord('c'): # Limpiar
        puntos, etiquetas, mascara_actual = [], [], None
        actualizar_pantalla()
        
    elif tecla == ord('q'): # Salir
        break

cv2.destroyAllWindows()