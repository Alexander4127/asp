# Anti-Spoofing Project

## Report

The pipeline details, experiments, ablations, and results are presented in the [wandb report](https://wandb.ai/practice-cifar/asp_project/reports/Anti-Spoofing-Report--Vmlldzo2MjcyOTQ3).

## Installation guide

To get started install the requirements
```shell
pip install -r ./requirements.txt
```

Then download train data ([ASVSpoof2019](https://datashare.ed.ac.uk/handle/10283/3336) dataset)
```shell
sudo apt install axel
bash loader.sh
```

## Model training

This project implements [RawNet2](https://arxiv.org/pdf/2011.01108.pdf) model for speech synthesis.

To train model from scratch run
```shell
python3 train.py -c asp/configs/train.json
```

For fine-tuning pretrained model from checkpoint, `--resume` parameter is applied.
For example, continuing training model with `train.json` config organized as follows
```shell
python3 train.py -c nv/configs/train.json -r saved/models/final/<run_id>/<any_checkpoint>.pth
```

## Inference stage

Checkpoint should be located in `default_test_model` directory. Pretrained model can be downloaded by running python code
```python3
import gdown
gdown.download("https://drive.google.com/uc?id=1HBDp__V5AGJacbhnaoQiC4g0wxizEFWa", "default_test_model/checkpoint.pth")
```

Model evaluation is executed by command
```shell
python3 test.py \
   -i default_test_model/test \
   -r default_test_model/checkpoint.pth \
   -o output.csv \
   -l False
```

- `-i` (`--input-dir`) provide the path to directory with input audio files. The model classify them to bonafide/spoof.
- `-r` (`--resume`) provide the path to model checkpoint. Note that config file is expected to be in the same dir with name `config.json`.
- `-o` (`--output`) specify output `.csv` file path. The dataframe with results will be saved there. Each row contains
  1. `Filename` of audio.
  2. `Logits` predicted by model (the first one corresponds to the `spoof` target).
  3. `Probs` calculated by `softmax` of logits.
  4. `Prediction` specifying the forecasted type (`spoof` for fake, `bonafide` for real).
- `-l` (`--log-wandb`) determine log results to wandb project or not. If `True`, authorization in command line is needed. Name of project can be changed in the config file.

Running with default parameters
```shell
python3 test.py
```

## Credits

The code of model is based on an [asr-template project](https://github.com/WrathOfGrapes/asr_project_template) 
and [notebook](https://github.com/XuMuK1/dla2023/blob/2023/week10/antispoofing_seminar.ipynb) with SincNet implementation.
