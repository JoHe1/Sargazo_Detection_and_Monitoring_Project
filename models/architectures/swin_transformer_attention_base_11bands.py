"""
models/architectures/swin_transformer_attention_base_11bands.py
----------------------------------------------------------------
Arquitectura: Swin-Base + Attention U-Net + Stage-1 Features + 11 bandas

Mejoras sobre SwinAttSegmenter (Swin-Tiny, 4 canales):

1. Swin-Base en lugar de Swin-Tiny
   - microsoft/swin-base-patch4-window7-224
   - Dimensiones encoder: [128, 256, 512, 1024] vs [96, 192, 384, 768]
   - ~87M parámetros vs ~28M — mayor capacidad de representación

2. Stage-1 High-Resolution Features (MariNeXt, Kikaki et al. 2024)
   - El paper MADOS demuestra que recuperar las features de Stage-1
     (resolución H/4 × W/4 = 56×56) mejora +5.7% F1 sobre SegNeXt base.
   - En Swin-Tiny estas features se ignoraban. Aquí se usan como
     skip connection adicional en el decoder para reconstrucción fina.
   - Permite predicciones a H/4 × W/4 antes del upsample final,
     capturando objetos pequeños como sargazo disperso en filamentos.

3. 11 canales de entrada Sentinel-2
   - R, G, B: copian pesos ImageNet pre-entrenados
   - Canales 4-10 (B1, B5-B12): inicializados con media de pesos RGB

Dimensiones del encoder Swin-Base:
    Stage 0 (s0): (B, 128, 56, 56)   ← Stage-1 HR features
    Stage 1 (s1): (B, 256, 28, 28)
    Stage 2 (s2): (B, 512, 14, 14)
    Stage 3 (s3): (B, 1024, 7, 7)    ← last_hidden_state

Decoder con Attention Gates:
    up4: s3(1024) + s2(512)  → 512ch  @14×14
    up3: 512      + s1(256)  → 256ch  @28×28
    up2: 256      + s0(128)  → 128ch  @56×56   ← usa Stage-1 HR features
    up1: 128      → 64ch     @112×112
    up0: 64       → 32ch     @224×224
    head: 32      → num_classes

Archivos independientes — no modifica los archivos originales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SwinModel
from core.interfaces.base_model import BaseModel


# ══════════════════════════════════════════════════════════════════════
# Bloques reutilizables
# ══════════════════════════════════════════════════════════════════════

class AttentionGate(nn.Module):
    """
    Puerta de Atención — filtra la skip connection con la señal gating.
    Suprime píxeles de agua (fondo) y refuerza sargazo.
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        psi = self.relu(self.W_g(g) + self.W_x(x))
        return x * self.psi(psi)


class AttDecoderBlock(nn.Module):
    """
    Bloque decoder: upsample + attention gate + concat + conv×2.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.attn = AttentionGate(
            F_g=in_channels, F_l=skip_channels, F_int=skip_channels // 2
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x    = self.up(x)
        skip = self.attn(g=x, x=skip)
        return self.conv(torch.cat([x, skip], dim=1))


# ══════════════════════════════════════════════════════════════════════
# Modelo principal
# ══════════════════════════════════════════════════════════════════════

class SwinBaseAtt11Bands(BaseModel):
    """
    Swin-Base + Attention U-Net + Stage-1 HR Features + 11 bandas S2.

    Corresponde a la arquitectura más cercana a MariNeXt pero con
    backbone Swin-Base y decoder con Attention Gates en lugar de
    la concatenación simple de SegNeXt.
    """

    def __init__(self, num_classes: int = 16, num_input_channels: int = 11, **kwargs):
        super().__init__()
        self.num_classes        = num_classes
        self.num_input_channels = num_input_channels

        # ── Encoder: Swin-Base ────────────────────────────────────────
        # Dimensiones ocultas: [128, 256, 512, 1024]
        self.backbone = SwinModel.from_pretrained(
            "microsoft/swin-base-patch4-window7-224"
        )

        # Adaptar primera capa conv para 11 canales
        old_conv = self.backbone.embeddings.patch_embeddings.projection
        new_conv = nn.Conv2d(
            num_input_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            # Canales R(0), G(1), B(2): copiar pesos ImageNet pre-entrenados
            new_conv.weight.data[:, :3, :, :] = old_conv.weight.data
            # Canales adicionales (3-10): inicializar con media de pesos RGB
            mean_rgb = old_conv.weight.data.mean(dim=1, keepdim=True)
            for c in range(3, num_input_channels):
                new_conv.weight.data[:, c:c+1, :, :] = mean_rgb
            if old_conv.bias is not None:
                new_conv.bias.data = old_conv.bias.data.clone()
        self.backbone.embeddings.patch_embeddings.projection = new_conv

        # ── Decoder: Attention U-Net con Stage-1 features ────────────
        # Swin-Base hidden dims: s0=128, s1=256, s2=512, s3=1024
        #
        # up4: s3(1024) + s2(512)  → 512  @14×14
        self.up4 = AttDecoderBlock(in_channels=1024, skip_channels=512, out_channels=512)
        # up3: 512      + s1(256)  → 256  @28×28
        self.up3 = AttDecoderBlock(in_channels=512,  skip_channels=256, out_channels=256)
        # up2: 256      + s0(128)  → 128  @56×56  ← Stage-1 HR features
        self.up2 = AttDecoderBlock(in_channels=256,  skip_channels=128, out_channels=128)

        # Capas finales: 56×56 → 224×224
        # up1: 128 → 64  @112×112
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # up0: 64 → 32  @224×224
        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Cabeza de clasificación final
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    # ------------------------------------------------------------------
    # Auxiliar: reshape hidden states de Swin a (B, C, H, W)
    # ------------------------------------------------------------------

    @staticmethod
    def _reshape(tensor: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """(B, L, C) → (B, C, H, W)"""
        return tensor.transpose(1, 2).contiguous().view(-1, tensor.size(2), h, w)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Encoder ──────────────────────────────────────────────────
        outputs = self.backbone(x, output_hidden_states=True)

        # hidden_states[0] = Stage-1 output = s0 (56×56, 128ch)
        # hidden_states[1] = Stage-2 output = s1 (28×28, 256ch)
        # hidden_states[2] = Stage-3 output = s2 (14×14, 512ch)
        # last_hidden_state = Stage-4 output = s3 (7×7,  1024ch)
        s0 = self._reshape(outputs.hidden_states[0], 56, 56)   # (B, 128, 56, 56)
        s1 = self._reshape(outputs.hidden_states[1], 28, 28)   # (B, 256, 28, 28)
        s2 = self._reshape(outputs.hidden_states[2], 14, 14)   # (B, 512, 14, 14)
        s3 = self._reshape(outputs.last_hidden_state,  7,  7)  # (B, 1024, 7, 7)

        # ── Decoder con Stage-1 features ─────────────────────────────
        x = self.up4(s3, s2)   # (B, 512, 14, 14)
        x = self.up3(x,  s1)   # (B, 256, 28, 28)
        x = self.up2(x,  s0)   # (B, 128, 56, 56) ← Stage-1 HR features aquí

        # ── Reconstrucción final ──────────────────────────────────────
        x = self.up1(x)        # (B, 64,  112, 112)
        x = self.up0(x)        # (B, 32,  224, 224)
        return self.head(x)    # (B, num_classes, 224, 224)

    # ------------------------------------------------------------------
    # Optimizador con learning rates diferenciados
    # ------------------------------------------------------------------

    def configure_optimizers(self, config):
        """
        Backbone (Swin-Base pre-entrenado) recibe LR reducido (×0.1)
        para no destruir los pesos ImageNet en las primeras épocas.
        Decoder y head reciben el LR completo.
        """
        import torch.optim as optim
        wd = getattr(config, 'weight_decay', 1e-4)

        backbone_params = list(self.backbone.parameters())
        decoder_params  = (
            list(self.up4.parameters()) +
            list(self.up3.parameters()) +
            list(self.up2.parameters()) +
            list(self.up1.parameters()) +
            list(self.up0.parameters()) +
            list(self.head.parameters())
        )

        return optim.AdamW([
            {"params": backbone_params, "lr": config.lr * 0.1},
            {"params": decoder_params,  "lr": config.lr},
        ], weight_decay=wd)
