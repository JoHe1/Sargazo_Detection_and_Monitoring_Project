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
    Loss = CrossEntropy(pesos, ignore_index=0) + Dice(clases_críticas)

    Cambio principal: pesos de sargazo 20 → 50.
    """

    CLASS_WEIGHTS = {
        0:  0.5,   # Non-annotated
        1:  10.0,  # Marine Debris
        2:  100.0,  # Dense Sargassum       ← subido de 50 a 100
        3:  100.0,  # Sparse Floating Algae ← subido de 50 a 100
        5:  10.0,  # Ship
        6:  10.0,  # Oil Spill
        7:  1.0,   # Marine Water
        10: 1.0,   # Turbid Water
    }

    DICE_CLASSES = [1, 2, 3, 5, 6]

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
        return ce_loss + dice_loss

    def to(self, device):
        if self.ce.weight is not None:
            self.ce.weight = self.ce.weight.to(device)
        return super().to(device)