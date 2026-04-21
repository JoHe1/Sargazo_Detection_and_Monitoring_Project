"""
models/architectures/segformer.py
-----------------------------------
Arquitectura SegFormerSegmenter: SegFormer backbone + decoder MLP.

Esta arquitectura usa SegformerForSemanticSegmentation de HuggingFace,
adaptada para aceptar 4 canales (RGB + NIR) en lugar de los 3 originales.

Arquitectura:
    Backbone : nvidia/mit-b2 (Mix Transformer, HuggingFace)
               Adaptado de 3 canales (RGB) a 4 (RGB + NIR)
               usando el mismo truco que SwinSegmenter:
               inicializar el canal NIR con la media de los pesos RGB.

    Decoder  : MLP ligero propio de SegFormer — fusiona features de
               las 4 etapas del encoder directamente mediante proyección
               lineal + upsampling bilinear. Evita el upsampling
               progresivo del UNet que genera halos en los bordes.

    Salida   : (B, NUM_CLASSES, 224, 224) — logits por píxel

Ventajas frente a Swin+UNet:
    - El decoder MLP no hace upsampling progresivo → menos halo
    - Diseñado nativamente para segmentación semántica
    - Mix Transformer captura contexto local y global eficientemente
    - Más ligero que Swin-Tiny en parámetros totales

Flujo de resoluciones (mit-b2):
    Entrada  : (B,  4, 224, 224)
    Etapa 0  : (B,  64,  56,  56)
    Etapa 1  : (B, 128,  28,  28)
    Etapa 2  : (B, 320,  14,  14)
    Etapa 3  : (B, 512,   7,   7)
    Decoder  : proyección MLP → (B, 256, 56, 56) → upsample → (B, 16, 224, 224)

Uso:
    model = SegFormerSegmenter(num_classes=16)
    logits = model(x)  # x: (B, 4, 224, 224)
                       # logits: (B, 16, 224, 224)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerModel

from core.interfaces.base_model import BaseModel


# ══════════════════════════════════════════════════════════════════════
# DECODER MLP
# ══════════════════════════════════════════════════════════════════════

class SegFormerDecoder(nn.Module):
    """
    Decoder MLP propio de SegFormer.

    Pasos:
        1. Proyección lineal de cada etapa al mismo número de canales
        2. Upsample bilinear de todas las etapas a la resolución de etapa 0
        3. Concatenación y fusión con Conv 1×1
        4. Upsample bilinear final a resolución de entrada (224×224)
        5. Clasificación con Conv 1×1

    La clave es que NO usa ConvTranspose2d progresivo — el upsample
    bilinear directo desde cada escala reduce el efecto halo.
    """

    def __init__(
        self,
        in_channels:  list[int],   # canales de cada etapa del encoder
        decoder_dim:  int = 256,   # dimensión interna del decoder
        num_classes:  int = 16,
        dropout:      float = 0.1,
    ) -> None:
        super().__init__()

        # Proyección lineal de cada etapa → decoder_dim
        self.linear_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, decoder_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(decoder_dim),
                nn.ReLU(inplace=True),
            )
            for c in in_channels
        ])

        # Fusión de todas las etapas concatenadas
        self.fusion = nn.Sequential(
            nn.Conv2d(decoder_dim * len(in_channels), decoder_dim,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        # Cabeza de clasificación
        self.classifier = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(
        self,
        features: list[torch.Tensor],   # [s0, s1, s2, s3] de tamaños distintos
        target_size: tuple[int, int],   # (H, W) de salida — 224×224
    ) -> torch.Tensor:

        # Tamaño de referencia = etapa 0 (mayor resolución)
        h0, w0 = features[0].shape[2], features[0].shape[3]

        # Proyectar y upsamplear todas las etapas a la resolución de etapa 0
        upsampled = []
        for feat, proj in zip(features, self.linear_projs):
            x = proj(feat)
            if x.shape[2] != h0 or x.shape[3] != w0:
                x = F.interpolate(x, size=(h0, w0),
                                  mode="bilinear", align_corners=False)
            upsampled.append(x)

        # Fusionar
        x = self.fusion(torch.cat(upsampled, dim=1))

        # Clasificar
        x = self.classifier(x)

        # Upsample final a tamaño de entrada
        x = F.interpolate(x, size=target_size,
                          mode="bilinear", align_corners=False)
        return x


# ══════════════════════════════════════════════════════════════════════
# MODELO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

class SegFormerSegmenter(BaseModel):
    """
    SegFormer (mit-b2) con decoder MLP para segmentación semántica.

    Hereda de BaseModel para obtener save(), load(), count_parameters()
    y get_info() de forma gratuita — igual que SwinSegmenter.

    Diferencia clave respecto al SegFormer original:
        La primera capa de patch embedding acepta 4 canales (RGB + NIR)
        en lugar de los 3 originales (RGB). El canal NIR se inicializa
        con la media de los pesos de los 3 canales RGB pre-entrenados.

    Uso:
        model = SegFormerSegmenter(num_classes=16)
        logits = model(x)  # x: (B, 4, 224, 224)
                           # logits: (B, 16, 224, 224)
    """

    BACKBONE_NAME = "nvidia/mit-b2"

    # Canales de salida de cada etapa de mit-b2
    ENCODER_CHANNELS = [64, 128, 320, 512]

    def __init__(self, num_classes: int = 16) -> None:
        """
        Args:
            num_classes: número de clases de segmentación (16 para MADOS)
        """
        super().__init__()
        self.num_classes  = num_classes
        self.target_size  = (224, 224)

        # ── Backbone ──────────────────────────────────────────────────
        self.backbone = SegformerModel.from_pretrained(
            self.BACKBONE_NAME,
            output_hidden_states=True,
            ignore_mismatched_sizes=True,
        )
        self._adapt_input_channels()

        # ── Decoder MLP ───────────────────────────────────────────────
        self.decoder = SegFormerDecoder(
            in_channels = self.ENCODER_CHANNELS,
            decoder_dim = 256,
            num_classes = num_classes,
            dropout     = 0.1,
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
        outputs = self.backbone(
            pixel_values=x,
            output_hidden_states=True,
        )

        # hidden_states: lista de 4 tensores (B, L, C) — uno por etapa
        # Los convertimos a formato espacial (B, C, H, W)
        hidden_states = outputs.hidden_states  # tuple de 4 elementos

        features = []
        spatial_sizes = [
            (56, 56),   # etapa 0: 224/4
            (28, 28),   # etapa 1: 224/8
            (14, 14),   # etapa 2: 224/16
            (7,  7),    # etapa 3: 224/32
        ]

        for i, (hs, (h, w)) in enumerate(zip(hidden_states, spatial_sizes)):
            # hs puede ser (B, L, C) o ya (B, C, H, W) según versión HF
            if hs.dim() == 3:
                B, L, C = hs.shape
                feat = hs.transpose(1, 2).reshape(B, C, h, w)
            else:
                feat = hs
            features.append(feat)

        return self.decoder(features, self.target_size)

    def configure_optimizers(self, config) -> torch.optim.Optimizer:
        """
        Configura el optimizador según ExperimentConfig.

        Usa learning rates diferenciados:
            - Backbone (encoder): lr / 10  — ya pre-entrenado
            - Decoder:            lr        — entrenado desde cero
        """
        lr = getattr(config, "lr", 5e-5)
        wd = getattr(config, "weight_decay", 1e-4)

        backbone_params = list(self.backbone.parameters())
        decoder_params  = list(self.decoder.parameters())

        name = getattr(config, "optimizer_name", "adamw").lower()

        param_groups = [
            {"params": backbone_params, "lr": lr / 10},
            {"params": decoder_params,  "lr": lr},
        ]

        if name == "adamw":
            return torch.optim.AdamW(param_groups, weight_decay=wd)
        elif name == "adam":
            return torch.optim.Adam(param_groups, weight_decay=wd)
        elif name == "sgd":
            return torch.optim.SGD(param_groups, momentum=0.9, weight_decay=wd)
        else:
            raise ValueError(f"Optimizador no reconocido: '{name}'. "
                             f"Opciones: adamw, adam, sgd")

    # ------------------------------------------------------------------
    # Interno
    # ------------------------------------------------------------------

    def _adapt_input_channels(self) -> None:
        """
        Adapta la primera capa de patch embedding de 3 canales a 4.

        SegFormer usa overlapping patch embeddings en cada etapa.
        Solo la primera etapa recibe la imagen directamente — las demás
        reciben features ya procesadas, así que solo hay que adaptar
        la primera capa de proyección.

        Inicialización del canal NIR:
            Media de los pesos RGB pre-entrenados, igual que en
            SwinSegmenter — conservador y aprovecha el transfer learning.
        """
        # La primera capa de patch embedding está en encoder.patch_embeddings[0]
        first_pe = self.backbone.encoder.patch_embeddings[0]
        old_proj = first_pe.proj  # Conv2d(3, 64, kernel=7, stride=4, padding=3)

        new_proj = nn.Conv2d(
            4,
            old_proj.out_channels,
            kernel_size  = old_proj.kernel_size,
            stride       = old_proj.stride,
            padding      = old_proj.padding,
            bias         = old_proj.bias is not None,
        )

        with torch.no_grad():
            new_proj.weight[:, :3, :, :] = old_proj.weight
            new_proj.weight[:, 3:4, :, :] = old_proj.weight.mean(dim=1, keepdim=True)
            if old_proj.bias is not None:
                new_proj.bias = nn.Parameter(old_proj.bias.clone())

        first_pe.proj = new_proj
