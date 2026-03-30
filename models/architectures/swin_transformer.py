"""
models/architectures/swin_transformer.py
------------------------------------------
Arquitectura SwinSegmenter: Swin Transformer backbone + decoder UNet-like.

Esta es la ÚNICA definición de SwinSegmenter en todo el proyecto.
Tanto train.py como inference.py importan desde aquí.

Arquitectura:
    Backbone : microsoft/swin-tiny-patch4-window7-224 (HuggingFace)
               Adaptado de 3 canales (RGB) a 4 (RGB + NIR)
               usando el truco de inicializar el canal NIR con la
               media de los pesos RGB pre-entrenados de ImageNet.

    Decoder  : UNet-like con skip connections desde las 4 etapas del Swin.
               Recupera la resolución espacial mediante ConvTranspose2d.

    Salida   : (B, NUM_CLASSES, 224, 224) — logits por píxel

Flujo de resoluciones:
    Entrada:  (B,  4, 224, 224)
    Etapa 0:  (B,  96,  56,  56)  ← skip s0
    Etapa 1:  (B, 192,  28,  28)  ← skip s1
    Etapa 2:  (B, 384,  14,  14)  ← skip s2
    Bottleneck:(B, 768,   7,   7)  ← s3
    dec1:     (B, 256,  14,  14)  (s3 + s2)
    dec2:     (B, 128,  28,  28)  (dec1 + s1)
    dec3:     (B,  64,  56,  56)  (dec2 + s0)
    dec4:     (B,  32, 112, 112)
    head:     (B,  16, 224, 224)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import SwinModel

from core.interfaces.base_model import BaseModel


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def reshape_hidden(seq: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    Convierte la salida secuencial del Swin a formato espacial.

    El Swin devuelve (B, L, C) donde L = H*W aplanado.
    PyTorch y los bloques conv esperan (B, C, H, W).

    Args:
        seq:      tensor (B, L, C)
        target_h: altura espacial de destino
        target_w: anchura espacial de destino

    Returns:
        tensor (B, C, target_h, target_w)
    """
    B, L, C = seq.shape
    return seq.transpose(1, 2).reshape(B, C, target_h, target_w)


# ══════════════════════════════════════════════════════════════════════
# BLOQUE DECODER
# ══════════════════════════════════════════════════════════════════════

class DecoderBlock(nn.Module):
    """
    Bloque de decodificación UNet-like.

    Pasos:
        1. Upsample x2 con ConvTranspose2d
        2. Concatenar skip connection de la etapa Swin correspondiente
        3. Dos Conv 3×3 para fusionar y refinar las features
    """

    def __init__(
        self,
        in_channels:   int,
        skip_channels: int,
        out_channels:  int,
    ) -> None:
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=4, stride=2, padding=1,
        )
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(
                out_channels + skip_channels, out_channels,
                kernel_size=3, padding=1, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv_fuse(x)


# ══════════════════════════════════════════════════════════════════════
# MODELO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

class SwinSegmenter(BaseModel):
    """
    Swin Transformer Tiny con decoder UNet para segmentación semántica.

    Hereda de BaseModel para obtener save(), load(), count_parameters()
    y get_info() de forma gratuita.

    Diferencia clave respecto al Swin original:
        La capa de proyección de patches acepta 4 canales (RGB + NIR)
        en lugar de los 3 originales (RGB). El canal NIR se inicializa
        con la media de los pesos de los 3 canales RGB pre-entrenados
        para aprovechar el transfer learning de ImageNet.

    Uso:
        model = SwinSegmenter(num_classes=16)
        logits = model(x)  # x: (B, 4, 224, 224)
                           # logits: (B, 16, 224, 224)
    """

    BACKBONE_NAME = "microsoft/swin-tiny-patch4-window7-224"

    def __init__(self, num_classes: int = 16) -> None:
        """
        Args:
            num_classes: número de clases de segmentación (16 para MADOS)
        """
        super().__init__()
        self.num_classes = num_classes

        # ── Backbone ──────────────────────────────────────────────────
        self.backbone = SwinModel.from_pretrained(
            self.BACKBONE_NAME,
            output_hidden_states=True,
        )
        self._adapt_input_channels()

        # ── Decoder ───────────────────────────────────────────────────
        # dec1: bottleneck (7×7, 768ch) → 14×14,  skip etapa 2 (384ch)
        self.dec1 = DecoderBlock(in_channels=768, skip_channels=384, out_channels=256)
        # dec2: 14×14 → 28×28,  skip etapa 1 (192ch)
        self.dec2 = DecoderBlock(in_channels=256, skip_channels=192, out_channels=128)
        # dec3: 28×28 → 56×56,  skip etapa 0 (96ch)
        self.dec3 = DecoderBlock(in_channels=128, skip_channels=96,  out_channels=64)
        # dec4: 56×56 → 112×112 (sin skip — el backbone ya no tiene features aquí)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # head: 112×112 → 224×224 → logits por clase
        self.head = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    # ------------------------------------------------------------------
    # Implementación de métodos abstractos de BaseModel
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante.

        Args:
            x: tensor (B, 4, 224, 224) — canales R, G, B, NIR normalizados

        Returns:
            logits (B, num_classes, 224, 224)
        """
        outputs = self.backbone(x, output_hidden_states=True)

        # Extraer features de cada etapa del Swin y convertir a formato espacial
        # hidden_states[0] = patch embeddings: (B, 3136,  96) = 56×56
        # hidden_states[1] = tras etapa 0:     (B,  784, 192) = 28×28
        # hidden_states[2] = tras etapa 1:     (B,  196, 384) = 14×14
        # last_hidden_state= tras etapa 3:     (B,   49, 768) =  7×7
        s0 = reshape_hidden(outputs.hidden_states[0], 56, 56)   # (B,  96, 56, 56)
        s1 = reshape_hidden(outputs.hidden_states[1], 28, 28)   # (B, 192, 28, 28)
        s2 = reshape_hidden(outputs.hidden_states[2], 14, 14)   # (B, 384, 14, 14)
        s3 = reshape_hidden(outputs.last_hidden_state,  7,  7)  # (B, 768,  7,  7)

        x = self.dec1(s3, s2)  # (B, 256, 14, 14)
        x = self.dec2(x,  s1)  # (B, 128, 28, 28)
        x = self.dec3(x,  s0)  # (B,  64, 56, 56)
        x = self.dec4(x)       # (B,  32, 112,112)
        return self.head(x)    # (B,  num_classes, 224, 224)

    def configure_optimizers(self, config) -> torch.optim.Optimizer:
        """
        Configura el optimizador según ExperimentConfig.

        Soporta: adamw, adam, sgd.
        Por defecto AdamW, que es el más adecuado para transformers.
        """
        name = getattr(config, "optimizer_name", "adamw").lower()
        lr   = getattr(config, "lr", 5e-5)
        wd   = getattr(config, "weight_decay", 1e-4)

        if name == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        elif name == "adam":
            return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        elif name == "sgd":
            return torch.optim.SGD(self.parameters(), lr=lr,
                                   momentum=0.9, weight_decay=wd)
        else:
            raise ValueError(f"Optimizador no reconocido: '{name}'. "
                             f"Opciones: adamw, adam, sgd")

    # ------------------------------------------------------------------
    # Interno
    # ------------------------------------------------------------------

    def _adapt_input_channels(self) -> None:
        """
        Adapta la capa de proyección de patches de 3 canales a 4.

        Inicialización del canal NIR:
            Se usa la media de los pesos de los 3 canales RGB pre-entrenados.
            Esto maximiza el aprovechamiento del transfer learning porque
            el NIR de Sentinel-2 se comporta de forma similar al canal R
            en escenas de vegetación — usar la media es más conservador
            que copiar un canal específico.
        """
        old_conv = self.backbone.embeddings.patch_embeddings.projection
        new_conv = nn.Conv2d(
            4, old_conv.out_channels,
            old_conv.kernel_size,
            old_conv.stride,
            old_conv.padding,
        )

        with torch.no_grad():
            # Copiar pesos RGB existentes
            new_conv.weight[:, :3, :, :] = old_conv.weight
            # Inicializar canal NIR con la media de los 3 canales RGB
            new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias = nn.Parameter(old_conv.bias.clone())

        self.backbone.embeddings.patch_embeddings.projection = new_conv
        self.backbone.config.num_channels = 4