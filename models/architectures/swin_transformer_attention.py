import torch
import torch.nn as nn
from transformers import SwinModel
from core.interfaces.base_model import BaseModel

class AttentionGate(nn.Module):
    """
    Puerta de Atención: Refina las características de la skip connection
    utilizando la señal de gating del decodificador.
    """
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g (int): Canales de la señal de gating (decodificador, más profunda).
            F_l (int): Canales de la skip connection (codificador, más superficial).
            F_int (int): Canales intermedios para la transformación.
        """
        super(AttentionGate, self).__init__()
        # Transformación para la señal de gating (g)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # Transformación para la skip connection (x)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # Transformación final para el mapa de coeficientes
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid() # Genera coeficientes entre 0.0 y 1.0
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: Señal de gating del decodificador [B, F_g, H', W'].
            x: Skip connection del codificador [B, F_l, H, W].
        """
        g1 = self.W_g(g) # Transformar gating
        x1 = self.W_x(x) # Transformar skip
        
        # Combinar y aplicar no linealidad para resaltar características
        psi = self.relu(g1 + x1) 
        # Generar mapa de coeficientes de atención
        psi = self.psi(psi)
        
        # Multiplicar la skip connection original por el mapa de coeficientes
        # (Atenúa el ruido y resalta el sargazo)
        return x * psi

class AttDecoderBlock(nn.Module):
    """
    Bloque de Decodificación con Atención (AttDecoderBlock).
    Sustituye al DecoderBlock original.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(AttDecoderBlock, self).__init__()
        # Puerta de Atención: El decodificador (in_channels) actúa como gate
        # sobre la skip connection (skip_channels).
        self.attn = AttentionGate(F_g=in_channels, F_l=skip_channels, F_int=skip_channels // 2)
        
        # Upsampling y Convoluciones Dobles
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Después de concatenar, el número de canales es:
        # canales_gating (out_channels del upsample) + canales_skip (atenuados)
        # Como AttUNet concatenamos in_channels (upsampled) + skip_channels.
        # En tu arquitectura Swin Tiny, in_channels = (768, 384, 192, 96) y out = in // 2.
        # skip = (384, 192, 96, None).
        
        # Para mantener el flujo de canales original, ajustamos el DoubleConv
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        # 1. Aplicar la Puerta de Atención a la Skip Connection
        skip = self.attn(g=x, x=skip)
        
        # 2. Upsample de las características del decodificador
        x = self.up(x)
        
        # 3. Concatenar y Convolución Doble
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class SwinAttSegmenter(BaseModel):
    """
    Segmentador Swin Transformer con Decodificador de Atención (AttUNet).
    """
    def __init__(self, num_classes=16, patch_size=4, window_size=7):
        super(SwinAttSegmenter, self).__init__()
        # 1. Codificador Swin (Intacto, Transfer Learning)
        self.backbone = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        
        # Adaptar para 4 canales si es necesario (como ya hacías)
        old_conv = self.backbone.embeddings.patch_embeddings.projection
        new_conv = nn.Conv2d(4, old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding)
        new_conv.weight.data[:, :3, :, :] = old_conv.weight.data
        new_conv.weight.data[:, 3:4, :, :] = old_conv.weight.data.mean(dim=1, keepdim=True)
        self.backbone.embeddings.patch_embeddings.projection = new_conv

        # 2. Decodificador de Atención
        # Stage 3 bottleneck (7x7, 768ch) -> Up4 -> 14x14 (384ch). Skip2: 14x14 (384ch)
        self.up4 = AttDecoderBlock(in_channels=768, skip_channels=384, out_channels=384)
        
        # 14x14 (384ch) -> Up3 -> 28x28 (192ch). Skip1: 28x28 (192ch)
        self.up3 = AttDecoderBlock(in_channels=384, skip_channels=192, out_channels=192)
        
        # 28x28 (192ch) -> Up2 -> 56x56 (96ch). Skip0: 56x56 (96ch)
        self.up2 = AttDecoderBlock(in_channels=192, skip_channels=96, out_channels=96)
        
        # 56x56 (96ch) -> Up1 -> 112x112 (96ch). No skip.
        # El Up1 es el final para recuperar 112x112 antes del final head.
        # No tiene skip connection. Se usa DoubleConv normal.
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        # 3. Final Head (Classification)
        self.head = nn.Conv2d(96, num_classes, kernel_size=1)

    def forward(self, x):
        outputs = self.backbone(x, output_hidden_states=True)
        # Extraer estados intermedios del codificador Swin
        s0 = outputs.hidden_states[1].permute(0, 3, 1, 2) # Stage 1 (56x56)
        s1 = outputs.hidden_states[3].permute(0, 3, 1, 2) # Stage 2 (28x28)
        s2 = outputs.hidden_states[5].permute(0, 3, 1, 2) # Stage 3 (14x14)
        s3 = outputs.hidden_states[7].permute(0, 3, 1, 2) # Bottleneck (7x7)

        # Decodificador de Atención con Skips
        x = self.up4(s3, s2) # -> 14x14
        x = self.up3(x, s1)  # -> 28x28
        x = self.up2(x, s0)  # -> 56x56
        
        # Último Upsample sin skip
        x = self.up1(x)       # -> 112x112
        
        # Clasificación final
        return self.head(x)   # -> 112x112, num_classes