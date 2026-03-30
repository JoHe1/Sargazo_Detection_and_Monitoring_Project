import requests
import json

# 1. Credenciales de Copernicus Data Space Ecosystem
# (Tendrás que registrarte gratis en dataspace.copernicus.eu para obtenerlas)
CLIENT_ID = "sh-fee1f5d8-063e-46d8-b2d0-959eab301524"
CLIENT_SECRET = "8Gfz02kS5qYPCxLtIZF6pju8QUh7MIvO"

def get_token(client_id, client_secret):
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }
    response = requests.post(url, data=data)
    return response.json()["access_token"]

# 2. El "EvalScript" (Para usar SCL en L2A)
evalscript = """
function setup() {
    return {
        input: ["B02", "B03", "B04", "B08", "SCL", "dataMask"],
        output: { bands: 4, sampleType: "FLOAT32" }
    };
}

function evaluatePixel(sample) {
    // Clases SCL a eliminar: 0: No Data, 3: Sombras, 8: Nubes medias, 9: Nubes altas, 10: Cirrus
    let badPixels = [0, 3, 8, 9, 10];
    
    if (badPixels.includes(sample.SCL) || sample.dataMask === 0) {
        return [0, 0, 0, 0]; 
    }
    
    return [sample.B04 * 2.5, sample.B03 * 2.5, sample.B02 * 2.5, sample.B08 * 2.5];
}
"""

# 3. Configuración de la petición (Zona, Fechas y Formato adaptados)
request_payload = {
    "input": {
        "bounds": {
            # Bounding Box calculado alrededor del centro lat: 18.5215, lng: -67.62
            # Formato: [min_lng, min_lat, max_lng, max_lat]
            "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
            "bbox": [-67.6700, 18.4715, -67.5700, 18.5715] 
        },
        "data": [
            {
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        # Fechas actualizadas a Julio de 2023
                        "from": "2023-07-01T00:00:00Z",
                        "to": "2023-07-31T23:59:59Z"
                    },
                    # Tolerancia de nubosidad aumentada al 30%
                    "maxCloudCoverage": 30 
                }
            }
        ]
    },
    "evalscript": evalscript,
    "output": {
        "width": 224,  # Tamaño para el Swin Transformer
        "height": 224,
        "responses": [
            {
                "identifier": "default",
                "format": {"type": "image/tiff"}
            }
        ]
    }
}

# 4. Hacemos la petición a la API
def download_patch():
    print("Obteniendo token...")
    token = get_token(CLIENT_ID, CLIENT_SECRET)
    
    print("Enviando petición de procesamiento a la nube...")
    url = "https://sh.dataspace.copernicus.eu/api/v1/process"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "image/tiff"
    }
    
    response = requests.post(url, headers=headers, json=request_payload)
    
    if response.status_code == 200:
        # Se guarda en la ruta que tenías configurada
        with open("examples/recorte_sargazo_limpio.tiff", "wb") as f:
            f.write(response.content)
        print("¡Éxito! Imagen guardada como 'recorte_sargazo_limpio.tiff'")
    else:
        print(f"Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    download_patch()