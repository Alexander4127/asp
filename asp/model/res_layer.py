import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, num_channels: int, conv_channels: int, first_layer: bool = False):
        super().__init__()
        if first_layer:
            self.batch_norm1 = nn.Sequential()
        else:
            self.batch_norm1 = nn.Sequential(
                nn.BatchNorm1d(num_features=num_channels),
                nn.LeakyReLU()
            )

        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=conv_channels, kernel_size=3, padding=1)
        self.batch_norm2 = nn.Sequential(
            nn.BatchNorm1d(num_features=conv_channels),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, padding=1)

        self.need_downsample = num_channels != conv_channels
        if num_channels != conv_channels:
            self.downsample = nn.Conv1d(in_channels=num_channels, out_channels=conv_channels, kernel_size=1)

        self.pool = nn.MaxPool1d(kernel_size=3)

    def forward(self, x):
        out = self.batch_norm1(x)
        out = self.conv1(out)
        out = self.batch_norm2(out)
        out = self.conv2(out)

        resid = x if not self.need_downsample else self.downsample(x)
        return self.pool(out + resid)


class ResLayer(nn.Module):
    def __init__(self, num_channels: int, conv_channels: int, first_layer: bool = False):
        super().__init__()
        self.block = ResBlock(num_channels, conv_channels, first_layer)
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc_attention = nn.Linear(in_features=conv_channels, out_features=conv_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.shape[0]
        xx = self.block(x)
        xy = self.avg_pool(xx).view(batch_size, -1)
        xy = self.fc_attention(xy)
        xy = self.sigmoid(xy).unsqueeze(2)
        return xx * xy + xy
