from itertools import chain
import torch
import torch.nn as nn

from asp.base.base_model import BaseModel
from asp.logger.logger import logger
from .sinc_layer import SincConv
from .res_layer import ResLayer


class RawNet2Model(BaseModel):
    def __init__(self,
                 sinc_params,
                 res1_channels: int,
                 res2_channels: int,
                 gru_hidden: int = 1024,
                 gru_layers: int = 3,
                 n_res2: int = 4,
                 batch_norm_gru: bool = True):
        super().__init__()
        self.sinc_layer = SincConv(**sinc_params)
        out_channels = sinc_params["out_channels"]
        self.pool1 = nn.MaxPool1d(kernel_size=3)
        self.batch_norm1 = nn.Sequential(
            nn.BatchNorm1d(num_features=out_channels),
            nn.LeakyReLU()
        )

        self.res_blocks1 = nn.ModuleList([
            ResLayer(num_channels=out_channels, conv_channels=res1_channels, first_layer=True),
            ResLayer(num_channels=res1_channels, conv_channels=res1_channels)
        ])
        out_channels = res1_channels
        self.res_blocks2 = nn.ModuleList(
            [ResLayer(num_channels=out_channels, conv_channels=res2_channels)] +
            [ResLayer(num_channels=res2_channels, conv_channels=res2_channels) for _ in range(n_res2 - 1)]
        )
        out_channels = res2_channels

        if batch_norm_gru:
            self.batch_norm2 = nn.Sequential(
                nn.BatchNorm1d(num_features=out_channels),
                nn.LeakyReLU()
            )
        else:
            self.batch_norm2 = nn.Sequential()

        self.gru = nn.GRU(input_size=out_channels, hidden_size=gru_hidden, num_layers=gru_layers, batch_first=True)

        self.head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(in_features=gru_hidden, out_features=2)
        )

    def forward(self, x):
        assert len(x.shape) == 2
        out = x.unsqueeze(1)

        out = self.sinc_layer(out)
        out = self.pool1(out)
        out = self.batch_norm1(out)


        for res_layer in chain(self.res_blocks1, self.res_blocks2):
            out = res_layer(out)

        out = self.batch_norm2(out).permute(0, 2, 1)   # (B, H, T) -> (B, T, H)
        out, _ = self.gru(out)

        out = out[:, -1, :]  # (B, H)
        out = self.head(out)

        return out
