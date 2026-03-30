import tifffile as tiff
import matplotlib.pyplot as plt
import numpy as np

# 1. Cargamos la imagen TIFF
ruta_imagen = "examples/recorte_sargazo_limpio.tiff"
imagen = tiff.imread(ruta_imagen)

print(f"Dimensiones de la imagen cargada: {imagen.shape}")
# Debería imprimir algo como (224, 224, 4). 
# Los canales son: 0=Rojo, 1=Verde, 2=Azul, 3=NIR

# 2. Normalización de los datos
# Las pantallas solo saben dibujar valores entre 0 y 1 (o 0 y 255).
# Como nuestros datos científicos pueden tener picos de brillo, 
# "recortamos" (clip) los valores para que se queden estrictamente entre 0.0 y 1.0
imagen_norm = np.clip(imagen, 0, 1)

# 3. Separar las bandas
# Recordamos el evalscript: [Rojo, Verde, Azul, NIR]
canal_R = imagen_norm[:, :, 0]
canal_G = imagen_norm[:, :, 1]
canal_B = imagen_norm[:, :, 2]
canal_NIR = imagen_norm[:, :, 3]

# 4. Crear las composiciones
# A) Color Verdadero (RGB) - Como lo vería el ojo humano
imagen_rgb = np.stack([canal_R, canal_G, canal_B], axis=-1)

# B) Falso Color (NIR-Rojo-Verde) - El estándar para detectar biomasa
# Asignamos el Infrarrojo al Rojo de la pantalla, el Rojo al Verde, y el Verde al Azul.
imagen_falso_color = np.stack([canal_NIR, canal_R, canal_G], axis=-1)

# 5. Dibujar las imágenes con Matplotlib
fig, ejes = plt.subplots(1, 2, figsize=(12, 6))

# Subplot 1: Color Real
ejes[0].imshow(imagen_rgb)
ejes[0].set_title("Color Real (RGB)\nSargazo apenas visible", fontsize=14)
ejes[0].axis('off')

# Subplot 2: Falso Color
ejes[1].imshow(imagen_falso_color)
ejes[1].set_title("Falso Color (NIR-R-G)\n¡El sargazo brilla en rojo!", fontsize=14)
ejes[1].axis('off')

# Mostrar la figura
plt.tight_layout()
plt.show()