import json
import numpy as np
from pathlib import Path
import random
import soundfile as sf
import time
from tqdm import tqdm
from typing import Optional

from asp.utils import ROOT_PATH
from asp.logger import logger


class ASVSpoof2019Dataset(object):
    def __init__(self, part: str, cut_audio: Optional[int] = None, data_dir=None, limit=None):
        self._data_dir = Path(data_dir) if data_dir is not None else ROOT_PATH / "data"
        self._index_dir = ROOT_PATH / "asp" / "datasets"
        self.index = self._get_or_load_index(part)
        self.cut_audio = cut_audio
        if limit is not None:
            random.seed(42)
            random.shuffle(self.index)
            self.index = self.index[:limit]

    def _get_or_load_index(self, part: str):
        if Path(self._index_dir / f"{part}.json").exists():
            with open(self._index_dir / f"{part}.json") as file:
                return json.loads(file.read())
        part_q = part + ".trn" if part == "train" else part + ".trl"
        protocol_file = self._data_dir / "LA" / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{part_q}.txt"
        flac_dir = self._data_dir / "LA" / f"ASVspoof2019_LA_{part}" / "flac"
        start = time.perf_counter()
        index = []
        with open(protocol_file) as file:
            protocols = sorted([line.strip() for line in file.readlines()])
        for line in tqdm(sorted(protocols), desc=part):
            _, f_name, _, _, tp = line.split()
            assert tp in ["bonafide", "spoof"]
            file_path = flac_dir / f"{f_name}.flac"
            assert file_path.exists(), f"{file_path}"
            index.append({"flac_file": str(file_path), "target": int(tp == "bonafide")})

        logger.info(f"Cost {time.perf_counter() - start:.2f}s to load {part} data.")

        with open(self._index_dir / f"{part}.json", "w") as file:
            print(json.dumps(index), file=file)

        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        d = self.index[item]
        wav, _ = sf.read(d["flac_file"])
        assert len(wav.shape) == 1, f'{wav.shape}'
        if self.cut_audio is not None:
            start_pos = np.random.randint(0, max(0, len(wav) - self.cut_audio) + 1)
            d["wave"] = wav[start_pos:start_pos + self.cut_audio]
        else:
            d["wave"] = wav
        return d
