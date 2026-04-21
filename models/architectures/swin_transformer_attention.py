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
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid() 
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g) 
        x1 = self.W_x(x) 
        psi = self.relu(g1 + x1) 
        psi = self.psi(psi)
        return x * psi

class AttDecoderBlock(nn.Module):
    """
    Bloque de Decodificación con Atención (AttDecoderBlock).
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(AttDecoderBlock, self).__init__()
        self.attn = AttentionGate(F_g=in_channels, F_l=skip_channels, F_int=skip_channels // 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        skip = self.attn(g=x, x=skip)
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class SwinAttSegmenter(BaseModel):
    """
    Segmentador Swin Transformer con Decodificador de Atención (AttUNet).
    """
    def __init__(self, num_classes=16, patch_size=4, window_size=7):
        super(SwinAttSegmenter, self).__init__()
        # 1. Codificador Swin
        self.backbone = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        
        # Adaptar para 4 canales
        old_conv = self.backbone.embeddings.patch_embeddings.projection
        new_conv = nn.Conv2d(4, old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding)
        new_conv.weight.data[:, :3, :, :] = old_conv.weight.data
        new_conv.weight.data[:, 3:4, :, :] = old_conv.weight.data.mean(dim=1, keepdim=True)
        self.backbone.embeddings.patch_embeddings.projection = new_conv

        # 2. Decodificador de Atención
        self.up4 = AttDecoderBlock(in_channels=768, skip_channels=384, out_channels=384)
        self.up3 = AttDecoderBlock(in_channels=384, skip_channels=192, out_channels=192)
        self.up2 = AttDecoderBlock(in_channels=192, skip_channels=96, out_channels=96)
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        # 3. Final Head
        self.head = nn.Conv2d(96, num_classes, kernel_size=1)

    def reshape_hidden(self, tensor: torch.Tensor, h: int, w: int) -> torch.Tensor:
            """
            Convierte la secuencia de parches 1D del Swin a un mapa 2D espacial.
            Usa .contiguous() y .view() para evitar problemas de memoria en GPU.
            tensor shape in:  (Batch, Sequence, Channels)
            tensor shape out: (Batch, Channels, Height, Width)
            """
            # Transponer de (B, L, C) a (B, C, L)
            tensor = tensor.transpose(1, 2)
            # Hacerlo contiguo en memoria y darle forma 2D
            return tensor.contiguous().view(-1, tensor.size(1), h, w)

    def forward(self, x):
        # 1. Pasar por el codificador Swin
        outputs = self.backbone(x, output_hidden_states=True)

        # 2. Extraer features de cada etapa usando la misma lógica que tu modelo original
        # s0: (B,  96, 56, 56)
        # s1: (B, 192, 28, 28)
        # s2: (B, 384, 14, 14)
        # s3: (B, 768,  7,  7)
        s0 = self.reshape_hidden(outputs.hidden_states[0], 56, 56) 
        s1 = self.reshape_hidden(outputs.hidden_states[1], 28, 28) 
        s2 = self.reshape_hidden(outputs.hidden_states[2], 14, 14) 
        s3 = self.reshape_hidden(outputs.last_hidden_state, 7, 7)

        # 3. Pasar por el decodificador de Atención
        x = self.up4(s3, s2)  # Sube de 7x7 a 14x14
        x = self.up3(x, s1)   # Sube de 14x14 a 28x28
        x = self.up2(x, s0)   # Sube de 28x28 a 56x56
        
        # 4. Upsample final y cabecera de clasificación
        x = self.up1(x)       # Sube a 112x112
        return self.head(x)   # Clasifica y sube internamente a 224x224 (según tu self.head)

    def configure_optimizers(self, config):
        """
        Configura el optimizador extrayendo el lr del objeto config que pasa train.py
        """
        import torch.optim as optim
        # Usamos el config.lr que tú defines al lanzar el comando
        return optim.AdamW(self.parameters(), lr=config.lr, weight_decay=1e-4)