"""
models/losses/cross_entropy_dice.py
-------------------------------------
Función de pérdida combinada CrossEntropy + Dice.

Cambios respecto a v1:
    - Pesos de sargazo subidos de 20 a 50 (clases 2 y 3).
      Con solo 0.002% de píxeles de sargazo, el peso 20 era insuficiente
      para que la señal compitiera con los 82M píxeles de fondo.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyDiceLoss(nn.Module):
    """
    Loss = CrossEntropy(pesos, ignore_index=0) + Dice(solo sargazo)

    Cambios v3: pesos sargazo 10 → 7, Marine Water 1 → 2,
    Dice solo en clases 2 y 3, smooth 1.0 → 0.1.
    """

    CLASS_WEIGHTS = {
            0:   0.0,   # Non-annotated — ignorado completamente
            1:   5.0,   # Marine Debris
            2:   7.0,   # Dense Sargassum — bajado de 10 a 7 para reducir sobredetección
            3:   7.0,   # Sparse Floating Algae — bajado de 10 a 7
            5:   5.0,   # Ship
            6:   5.0,   # Oil Spill
            7:   2.0,   # Marine Water — subido de 1 a 2 para mejor discriminación agua/sargazo
            10:  1.0,   # Turbid Water
    }

    # Solo sargazo en Dice — concentra el gradiente en las clases de interés
    # sin diluirlo entre Marine Debris, Ship y Oil Spill
    DICE_CLASSES = [2, 3]

    def __init__(self, num_classes: int = 16, device: str = "cpu") -> None:
        super().__init__()
        self.num_classes = num_classes

        pesos = torch.ones(num_classes, device=device)
        for clase, peso in self.CLASS_WEIGHTS.items():
            if clase < num_classes:
                pesos[clase] = peso

        self.ce = nn.CrossEntropyLoss(weight=pesos, ignore_index=0)

    def forward(
        self,
        inputs:  torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        ce_loss = self.ce(inputs, targets)

        mascara_anotada = (targets != 0).unsqueeze(1).float()
        probs           = torch.softmax(inputs, dim=1)
        targets_ohe     = F.one_hot(targets, num_classes=self.num_classes)
        targets_ohe     = targets_ohe.permute(0, 3, 1, 2).float()

        smooth    = 0.1   # Bajado de 1.0 para Dice más estricto en bordes
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
        return ce_loss + dice_loss

    def to(self, device):
        if self.ce.weight is not None:
            self.ce.weight = self.ce.weight.to(device)
        return super().to(device)