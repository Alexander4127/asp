from typing import List

import numpy as np
from asp.base.base_metric import BaseMetric
from .utils import compute_eer


class EERMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, targets: np.ndarray, probs: np.ndarray):
        """
        Calculate EER (Equal Error Rate)
        1 - bonafide audio
        0 - spoof audio
        :param targets: (B,) tensor of targets
        :param probs: (B,) tensor of target == 1 probs
        :return: EER
        """
        return compute_eer(probs[targets], probs[~targets])
