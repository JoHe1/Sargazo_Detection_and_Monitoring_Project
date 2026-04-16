"""
models/losses/cross_entropy_dice_tversky.py
---------------------------------------------
Función de pérdida combinada CrossEntropy + Dice + Tversky.

Respecto a cross_entropy_dice.py:
    - Se añade Tversky Loss para las clases de sargazo (2 y 3).
    - Tversky con alpha=0.7, beta=0.3 penaliza los Falsos Positivos
      más que los Falsos Negativos, obligando al modelo a ceñirse
      a los bordes del sargazo real y reducir el halo de sobredetección.

Fórmula Tversky:
    TverskyScore = TP / (TP + alpha*FP + beta*FN)
    TverskyLoss  = 1 - TverskyScore

    alpha=0.7 → penaliza FP (halo)
    beta=0.3  → penaliza menos FN (no queremos perder Recall)

Loss final:
    Loss = CrossEntropy + Dice + Tversky
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyDiceTverskyLoss(nn.Module):
    """
    Loss = CrossEntropy(pesos, ignore_index=0)
         + Dice(clases_críticas)
         + Tversky(clases_sargazo, alpha=0.7, beta=0.3)

    CrossEntropy: penaliza el desbalance global de clases.
    Dice:         favorece la superposición de máscaras.
    Tversky:      penaliza específicamente el halo (FP) en sargazo.
    """

    CLASS_WEIGHTS = {
        0:   0.5,   # Non-annotated
        1:  10.0,   # Marine Debris
        2: 100.0,   # Dense Sargassum
        3: 100.0,   # Sparse Floating Algae
        5:  10.0,   # Ship
        6:  10.0,   # Oil Spill
        7:   1.0,   # Marine Water
        10:  1.0,   # Turbid Water
    }

    DICE_CLASSES    = [1, 2, 3, 5, 6]   # igual que antes
    TVERSKY_CLASSES = [2, 3]             # solo sargazo

    def __init__(
        self,
        num_classes: int   = 16,
        device:      str   = "cpu",
        alpha:       float = 0.7,   # peso FP — sube para reducir halo
        beta:        float = 0.3,   # peso FN — bajo para no perder Recall
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.alpha       = alpha
        self.beta        = beta

        pesos = torch.ones(num_classes, device=device)
        for clase, peso in self.CLASS_WEIGHTS.items():
            if clase < num_classes:
                pesos[clase] = peso

        self.ce = nn.CrossEntropyLoss(weight=pesos, ignore_index=0)

    def forward(
        self,
        inputs:  torch.Tensor,   # (B, C, H, W) logits
        targets: torch.Tensor,   # (B, H, W)    clases
    ) -> torch.Tensor:

        # ── CrossEntropy ──────────────────────────────────────────────
        ce_loss = self.ce(inputs, targets)

        mascara_anotada = (targets != 0).unsqueeze(1).float()
        probs           = torch.softmax(inputs, dim=1)
        targets_ohe     = F.one_hot(targets, num_classes=self.num_classes)
        targets_ohe     = targets_ohe.permute(0, 3, 1, 2).float()

        smooth = 1.0

        # ── Dice ──────────────────────────────────────────────────────
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

        # ── Tversky (solo sargazo) ────────────────────────────────────
        tversky_loss = 0.0
        for c in self.TVERSKY_CLASSES:
            pred_c   = (probs[:, c:c+1, :, :] * mascara_anotada).contiguous().view(-1)
            target_c = (targets_ohe[:, c:c+1, :, :] * mascara_anotada).contiguous().view(-1)

            tp = (pred_c * target_c).sum()
            fp = (pred_c * (1.0 - target_c)).sum()   # halo — penalizado con alpha
            fn = ((1.0 - pred_c) * target_c).sum()   # sargazo perdido — penalizado con beta

            tversky_score = (tp + smooth) / (
                tp + self.alpha * fp + self.beta * fn + smooth
            )
            tversky_loss += 1.0 - tversky_score
        tversky_loss = tversky_loss / len(self.TVERSKY_CLASSES)

        return ce_loss + dice_loss + tversky_loss

    def to(self, device):
        if self.ce.weight is not None:
            self.ce.weight = self.ce.weight.to(device)
        return super().to(device)
