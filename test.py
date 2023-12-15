import argparse
import multiprocessing
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from asp.logger import WanDBWriter
import asp.model as module_model
from asp.utils import ROOT_PATH
from asp.utils.parse_config import ConfigParser
from train import SEED

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"
torch.manual_seed(SEED)

def main(args, config):
    logger = config.get_logger("test")
    writer: WanDBWriter = WanDBWriter(config, logger) if args.log_wandb else None

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    # read audio filenames and texts
    audio_names = [filename for filename in sorted(Path(args.input_dir).iterdir())]

    sr = config["preprocessing"]["sr"]
    classes = ["spoof", "bonafide"]
    table = pd.DataFrame(columns=["Filename", "Audio", "Logits", "Probs", "Prediction"])
    with torch.no_grad():
        for idx, audio_name in tqdm(enumerate(audio_names), total=len(audio_names), desc="Infer"):
            name = audio_name.stem
            audio, sr_audio = sf.read(os.path.join('', audio_name))
            if sr_audio != sr:
                audio = torchaudio.functional.resample(torch.from_numpy(audio), sr_audio, sr).numpy()
            audio = torch.tensor(audio).to(torch.float32).unsqueeze(0).to(device)
            logit = model(audio).squeeze()
            probs = torch.softmax(logit, dim=-1)
            table.loc[len(table)] = [
                name,
                writer.wandb.Audio(audio.squeeze().cpu().numpy(), sample_rate=sr) if writer is not None else None,
                str(logit.cpu().tolist()),
                str(probs.cpu().tolist()),
                classes[torch.argmax(probs).item()]
            ]

    if writer is not None:
        writer.set_step(step=0, mode="test")
        writer.add_table("results", table)

    table.drop(columns=["Audio"]).to_csv(args.output)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default=str(DEFAULT_CHECKPOINT_PATH.parent / "test"),
        help="Path to file with audios to test"
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.csv",
        type=str,
        help="File to write result verdicts",
    )
    args.add_argument(
        "-l",
        "--log-wandb",
        default=False,
        type=bool,
        help="Save results in wandb or not (wand params are in config file)"
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    main(args, config)
