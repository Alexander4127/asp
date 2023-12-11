import torch
import torch.nn as nn
from typing import Optional, Sequence


class RawNet2Loss(nn.Module):
    def __init__(self, weight: Optional[Sequence] = None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=torch.tensor(weight))

    def forward(self, pred, target, **kwargs):
        return self.loss(pred, target)
