from typing import List
import torch


class Collator(object):
    def __call__(self, batch: List[dict]):
        length_wave = torch.tensor([len(el["wave"]) for el in batch])

        waves = torch.zeros([len(batch), max(length_wave)])
        for idx, (length, el) in enumerate(zip(length_wave, batch)):
            waves[idx, :length] = torch.tensor(el["wave"])

        return {
            "wave": waves,
            "length_wave": length_wave,
            "target": torch.tensor([el["target"] for el in batch])
        }
