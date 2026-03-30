import rasterio
from rasterio.mask import mask
import geopandas as gpd
import pandas as pd # ¡NUEVO: Importamos pandas para unir los mapas!
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# CONFIGURACIÓN
# ============================================
# 1. Rutas
ruta_imagen_cruda = "examples/recorte_sargazo_limpio.tiff"
ruta_mapa_principal = "examples/land_mask/ne_10m_land.shp" 
ruta_mapa_islas = "examples/land_mask/ne_10m_minor_islands.shp" # ¡NUEVO: Ruta de las islas menores!
ruta_salida = "examples/recorte_sargazo_enmascarado.tiff"

# ============================================
# PROCESAMIENTO
# ============================================
print("1. Cargando y uniendo los mapas vectoriales (Tierra e Islas Menores)...")
# Cargamos ambos archivos
mapa_principal = gpd.read_file(ruta_mapa_principal)
mapa_islas = gpd.read_file(ruta_mapa_islas)

# Unimos los dos GeoDataFrames en uno solo
mundo = pd.concat([mapa_principal, mapa_islas], ignore_index=True)
print(f"   Mapas unidos. Total de polígonos: {len(mundo)}")

print("2. Procesando la imagen y aplicando Land Masking...")
with rasterio.open(ruta_imagen_cruda) as src:
    # Aseguramos que el mapa unificado tenga el mismo sistema de coordenadas que la imagen
    mundo = mundo.to_crs(src.crs)
    
    # --- Leemos la imagen original antes de tocarla ---
    print("   Leyendo datos originales...")
    img_original = src.read()
    
    try:
        print("   Aplicando máscara geométrica unificada...")
        # Enmascaramos (tierra = ceros absolutos). invert=True mantiene el océano.
        img_enmascarada, out_transform = mask(src, mundo.geometry, invert=True)
        print("   ¡Máscara de tierra aplicada con éxito!")
    except ValueError:
        print("   Aviso: Esta imagen es 100% océano, no se ha recortado nada.")
        img_enmascarada = img_original
        out_transform = src.transform

    # 3. GUARDAR EL NUEVO TIFF ENMASCARADO
    print(f"3. Guardando el resultado en {ruta_salida}...")
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": img_enmascarada.shape[1],
        "width": img_enmascarada.shape[2],
        "transform": out_transform
    })
    
    with rasterio.open(ruta_salida, "w", **out_meta) as dest:
        dest.write(img_enmascarada)
        
print("   ¡Proceso completado!")

# ============================================
# VISUALIZACIÓN COMPARATIVA (ANTES vs DESPUÉS)
# ============================================
print("4. Generando visualización comparativa...")

def preparar_falso_color(raster_data):
    # Asumimos que el orden de canales es [0:R, 1:G, 2:B, 3:NIR]
    # Normalizamos entre 0 y 1 para visualización
    nir = np.clip(raster_data[3, :, :], 0, 1)
    r = np.clip(raster_data[0, :, :], 0, 1)
    g = np.clip(raster_data[1, :, :], 0, 1)
    return np.stack([nir, r, g], axis=-1)

# Preparamos ambas imágenes
viz_antes = preparar_falso_color(img_original)
viz_despues = preparar_falso_color(img_enmascarada)

# Creamos la figura con 2 paneles (1 fila, 2 columnas)
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Panel 1: Antes
axes[0].imshow(viz_antes)
axes[0].set_title("ANTES: Imagen Original\n(La costa y las islas son visibles)", fontsize=12)
axes[0].axis('off')

# Panel 2: Después
axes[1].imshow(viz_despues)
axes[1].set_title("DESPUÉS: Land Masking Aplicado\n(Tierra e islas menores censuradas a negro)", fontsize=12)
axes[1].axis('off')

plt.tight_layout()
print("   Mostrando gráfico. ¡Listo para la captura!")
plt.show()