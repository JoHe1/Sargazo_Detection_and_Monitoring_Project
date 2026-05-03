import torch
import torch.nn as nn
from transformers import SwinModel
from core.interfaces.base_model import BaseModel

class AttentionGate(nn.Module):
    """
    Puerta de Atención: Filtra la información de la skip connection.
    Ayuda a eliminar el 'efecto halo' al silenciar píxeles de agua.
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
    Bloque de Decodificación con Puerta de Atención incorporada.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(AttDecoderBlock, self).__init__()
        # La señal de gating es 'in_channels' (lo que viene de abajo)
        # La skip connection es 'skip_channels' (lo que viene del encoder)
        self.attn = AttentionGate(F_g=in_channels, F_l=skip_channels, F_int=skip_channels // 2)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.15),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
            # 1. PRIMERO escalamos la imagen profunda (gating) hacia arriba
            # Así pasamos de 7x7 a 14x14 para que encaje con la skip connection
            x = self.up(x)
            
            # 2. AHORA aplicamos la atención (ambos tensores ya miden lo mismo)
            skip = self.attn(g=x, x=skip)
            
            # 3. Concatenamos y aplicamos convoluciones
            x = torch.cat([x, skip], dim=1)
            x = self.conv(x)
            return x

class SwinAttSegmenter(BaseModel):
    """
    ARQUITECTURA FINAL: Swin Transformer + Attention U-Net
    """
    def __init__(self, num_classes=16, **kwargs):
        super(SwinAttSegmenter, self).__init__()
        # 1. Codificador Swin (Cargando pesos de Microsoft)
        self.backbone = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        
        # Adaptar para 4 canales (RGB + NIR)
        old_conv = self.backbone.embeddings.patch_embeddings.projection
        new_conv = nn.Conv2d(4, old_conv.out_channels, kernel_size=old_conv.kernel_size, 
                             stride=old_conv.stride, padding=old_conv.padding)
        new_conv.weight.data[:, :3, :, :] = old_conv.weight.data
        new_conv.weight.data[:, 3:4, :, :] = old_conv.weight.data.mean(dim=1, keepdim=True)
        self.backbone.embeddings.patch_embeddings.projection = new_conv

        # 2. Decodificadores con Atención (U-Net)
        # s3 (7x7, 768ch) + s2 (14x14, 384ch) -> 256ch
        self.up4 = AttDecoderBlock(in_channels=768, skip_channels=384, out_channels=256)
        # x (14x14, 256ch) + s1 (28x28, 192ch) -> 128ch
        self.up3 = AttDecoderBlock(in_channels=256, skip_channels=192, out_channels=128)
        # x (28x28, 128ch) + s0 (56x56, 96ch) -> 64ch
        self.up2 = AttDecoderBlock(in_channels=128, skip_channels=96, out_channels=64)
        
        # 3. Capas finales para llegar a 224x224
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # 112x112
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.10),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # 224x224
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def reshape_hidden(self, tensor: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """ Función auxiliar para doblar parches en mapas 2D (Usada en tu base_model) """
        tensor = tensor.transpose(1, 2) # (B, C, L)
        return tensor.contiguous().view(-1, tensor.size(1), h, w)

    def forward(self, x):
        # 1. Paso por el Encoder
        outputs = self.backbone(x, output_hidden_states=True)

        # 2. Extracción de capas (Lógica idéntica a tu swin_transformer.py base)
        s0 = self.reshape_hidden(outputs.hidden_states[0], 56, 56) 
        s1 = self.reshape_hidden(outputs.hidden_states[1], 28, 28) 
        s2 = self.reshape_hidden(outputs.hidden_states[2], 14, 14) 
        s3 = self.reshape_hidden(outputs.last_hidden_state, 7, 7)

        # 3. Paso por el Decoder con Atención
        x = self.up4(s3, s2)  # 14x14
        x = self.up3(x, s1)   # 28x28
        x = self.up2(x, s0)   # 56x56
        
        # 4. Reconstrucción final y clasificación
        x = self.up1(x)       # 224x224
        return self.head(x)

    def configure_optimizers(self, config):
        """ Configura el optimizador AdamW usando el objeto config de train.py """
        import torch.optim as optim
        wd = getattr(config, 'weight_decay', 1e-4)
        return optim.AdamW(self.parameters(), lr=config.lr, weight_decay=wd)