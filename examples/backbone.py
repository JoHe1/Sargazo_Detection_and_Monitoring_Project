import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ==========================================
# 1. FUNCIÓN DE PÉRDIDA: DICE LOSS
# ==========================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        # 'smooth' evita divisiones por cero si la imagen es todo agua
        self.smooth = smooth

    def forward(self, preds, targets):
        # Aplicamos sigmoide para que las predicciones estén entre 0 y 1
        preds = torch.sigmoid(preds)
        
        # Aplanamos los tensores a 1D para el cálculo matemático
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        # TP (Intersección)
        intersection = (preds * targets).sum()
        
        # Cálculo del coeficiente Dice
        dice_coeff = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        
        # Como queremos minimizar el error, devolvemos 1 - Dice
        return 1 - dice_coeff

# ==========================================
# 2. MODELO COMPLETO: ENCODER + DECODER
# ==========================================
class SargassumSegmenter(nn.Module):
    def __init__(self):
        super(SargassumSegmenter, self).__init__()
        
        # ENCODER: Nuestro Swin Transformer (4 canales)
        # features_only=True es clave: le dice al modelo que no queremos clasificar,
        # sino extraer los mapas de características jerárquicos para segmentar.
        self.encoder = timm.create_model(
            'swin_tiny_patch4_window7_224', 
            pretrained=True, 
            in_chans=4,
            features_only=True
        )
        
        # DECODER: Un bloque sencillo de convolución y upsampling (interpolación)
        # El último mapa de características de la versión 'tiny' tiene 768 canales.
        # Lo reducimos paso a paso hasta llegar a 1 solo canal (máscara binaria: sargazo o no).
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1) # Salida: 1 canal (Sargazo)
        )

    def forward(self, x):
        # 1. Pasamos la imagen por el Encoder
        features = self.encoder(x)
        deep_features = features[-1] 
        
        # 2. ¡LA MAGIA DE LAS DIMENSIONES!
        # Convertimos de [Batch, Alto, Ancho, Canales] a [Batch, Canales, Alto, Ancho]
        # O sea, pasamos de [2, 7, 7, 768] a [2, 768, 7, 7]
        if deep_features.dim() == 4 and deep_features.shape[-1] == 768:
            deep_features = deep_features.permute(0, 3, 1, 2)
        elif deep_features.dim() == 3: # Por si en el futuro usas otra versión de la librería
            B, L, C = deep_features.shape
            H = W = int(L**0.5)
            deep_features = deep_features.view(B, H, W, C).permute(0, 3, 1, 2)
            
        # 3. Pasamos las características por el Decoder
        mask_logits = self.decoder(deep_features)
        
        # 4. Restauramos la resolución original dinámicamente
        out_mask = F.interpolate(mask_logits, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        
        return out_mask

# ==========================================
# 3. PRUEBA DEL CÓDIGO
# ==========================================
if __name__ == "__main__":
    # Simulamos un lote de 2 imágenes Sentinel-2 (4 canales, 224x224)
    dummy_images = torch.randn(2, 4, 224, 224)
    # Simulamos el Ground Truth (máscaras de SAM, 1 canal, 224x224)
    dummy_masks = torch.randint(0, 2, (2, 1, 224, 224)).float()
    
    # Inicializamos el modelo y la pérdida
    modelo = SargassumSegmenter()
    criterio = DiceLoss()
    
    # Hacemos una pasada hacia adelante (Inferencia)
    predicciones = modelo(dummy_images)
    
    # Calculamos el error
    error = criterio(predicciones, dummy_masks)
    
    print(f"Dimensión de entrada: {dummy_images.shape}")
    print(f"Dimensión de salida de la máscara: {predicciones.shape}")
    print(f"Dice Loss calculado: {error.item():.4f}")
    print("¡Estructura base completada con éxito!")