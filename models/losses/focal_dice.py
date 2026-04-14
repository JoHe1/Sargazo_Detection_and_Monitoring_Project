"""
models/losses/focal_dice.py
-----------------------------
Función de pérdida combinada Focal Loss + Dice.

Motivación:
    Con ~0.002% de píxeles de sargazo en el dataset, CrossEntropy
    ponderada no es suficiente: los píxeles de agua "fáciles" siguen
    dominando el gradiente incluso con weight=1, porque son ~82M
    píxeles contra unos pocos miles de sargazo.

    Focal Loss (Lin et al., 2017, "Focal Loss for Dense Object Detection")
    introduce un factor modulador (1 - p_t)^gamma que:
      - Reduce el peso de ejemplos bien clasificados (p_t alto).
      - Mantiene el peso de ejemplos difíciles (p_t bajo).
    Es compatible con class weights: se multiplican, no se sustituyen.

    El término Dice se mantiene igual que en CrossEntropyDiceLoss
    porque aporta una señal a nivel de región (no solo pixel a pixel).

Parámetros clave:
    - gamma:  factor de focalización (2.0 es el valor estándar del paper).
    - alpha:  los class weights actúan como alpha por clase.
    - ignore_index=0: igual que antes, Non-annotated se ignora.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalDiceLoss(nn.Module):
    """
    Loss = FocalLoss(pesos, gamma, ignore_index=0) + Dice(clases_críticas)

    Pesos por clase idénticos a CrossEntropyDiceLoss para permitir
    comparación directa entre ambas losses en el TFG.
    """

    CLASS_WEIGHTS = {
        0:  0.5,    # Non-annotated (ignorada por ignore_index, pero se define)
        1:  10.0,   # Marine Debris
        2:  100.0,  # Dense Sargassum
        3:  100.0,  # Sparse Floating Algae
        5:  10.0,   # Ship
        6:  10.0,   # Oil Spill
        7:  1.0,    # Marine Water
        10: 1.0,    # Turbid Water
    }

    DICE_CLASSES = [1, 2, 3, 5, 6]

    def __init__(
        self,
        num_classes: int = 16,
        gamma: float = 2.0,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.gamma       = gamma
        self.ignore_index = 0

        pesos = torch.ones(num_classes, device=device)
        for clase, peso in self.CLASS_WEIGHTS.items():
            if clase < num_classes:
                pesos[clase] = peso

        # Guardamos los pesos como buffer para que se muevan con .to(device)
        self.register_buffer("class_weights", pesos)

    def focal_loss(
        self,
        inputs:  torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Focal Loss multiclase con class weights y ignore_index.

        FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)

        donde p_t es la probabilidad predicha para la clase verdadera.
        """
        # log-softmax numéricamente estable: shape (B, C, H, W)
        log_probs = F.log_softmax(inputs, dim=1)

        # Máscara de píxeles válidos (distintos de ignore_index)
        valid_mask = (targets != self.ignore_index)

        # Clamp de targets para que gather no falle con índice 0 ignorado
        # (los píxeles ignorados se enmascaran después, no importa su valor)
        targets_clamped = targets.clamp(min=0)

        # Recogemos log p_t para la clase verdadera de cada píxel
        # log_probs: (B, C, H, W)  →  (B, 1, H, W)  →  (B, H, W)
        log_pt = log_probs.gather(1, targets_clamped.unsqueeze(1)).squeeze(1)
        pt     = log_pt.exp()

        # alpha_t: peso de la clase verdadera en cada píxel
        alpha_t = self.class_weights[targets_clamped]

        # Focal term
        focal_term = (1.0 - pt) ** self.gamma
        loss_per_pixel = -alpha_t * focal_term * log_pt

        # Aplicamos la máscara de píxeles válidos
        loss_per_pixel = loss_per_pixel * valid_mask.float()

        # Normalización: media sobre píxeles válidos (no sobre todos)
        num_valid = valid_mask.sum().clamp(min=1)
        return loss_per_pixel.sum() / num_valid

    def forward(
        self,
        inputs:  torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # ── Término Focal (reemplaza al CE) ─────────────────────────
        fl_loss = self.focal_loss(inputs, targets)

        # ── Término Dice (idéntico a CrossEntropyDiceLoss) ──────────
        mascara_anotada = (targets != 0).unsqueeze(1).float()
        probs           = torch.softmax(inputs, dim=1)
        targets_ohe     = F.one_hot(targets, num_classes=self.num_classes)
        targets_ohe     = targets_ohe.permute(0, 3, 1, 2).float()

        smooth    = 1.0
        dice_loss = 0.0

        for c in self.DICE_CLASSES:
            pred_c   = (probs[:, c:c+1, :, :] * mascara_anotada).contiguous().view(-1)
            target_c = (targets_ohe[:, c:c+1, :, :] * mascara_anotada).contiguous().view(-1)
            intersection = (pred_c * target_c).sum()
            dice_score   = (2.0 * intersection + smooth) / (
                pred_c.sum() + target_c.sum() + smooth
            )
            dice_loss += 1.0 - dice_score

        dice_loss = dice_loss / len(self.DICE_CLASSES)
        return fl_loss + dice_loss

    def to(self, device):
        self.class_weights = self.class_weights.to(device)
        return super().to(device)