import torch
from torch import nn

from project.config.cuda_config import current_device
from project.nn.conv import LocationAwareConv2d
from project.nn.convgru import ConvGRU


class LatentBlock(nn.Module):
    def __init__(
        self, hidden_dim: int, input_dim: int, location_aware: bool = False,
    ):
        super().__init__()
        # self.lstm_dims = [128, 128, 64]
        self.lstm_dims = [64]
        if location_aware:
            self.conv1x1 = LocationAwareConv2d(
                in_channels=hidden_dim,
                out_channels=64,
                kernel_size=1,
                w=input_dim,
                h=input_dim,
                gradient=None,
            )
        else:
            self.conv1x1 = nn.Conv2d(
                in_channels=hidden_dim, out_channels=64, kernel_size=1
            )
        self.convgru1 = ConvGRU(
            input_size=(input_dim, input_dim),
            input_dim=hidden_dim,
            hidden_dim=self.lstm_dims,
            num_layers=len(self.lstm_dims),
            kernel_size=(3, 3),
            n_step_ahead=3,
        )
        self.convgru2 = ConvGRU(
            input_size=(input_dim, input_dim),
            input_dim=hidden_dim,
            hidden_dim=self.lstm_dims,
            num_layers=len(self.lstm_dims),
            kernel_size=(5, 5),
            n_step_ahead=3,
        )
        self.convgru3 = ConvGRU(
            input_size=(input_dim, input_dim),
            input_dim=hidden_dim,
            hidden_dim=self.lstm_dims,
            num_layers=len(self.lstm_dims),
            kernel_size=(7, 7),
            n_step_ahead=3,
        )

    def forward(self, x, seq_len=3, n_ahead=3, test=False):
        # print("LAT-BLK", x.shape)
        x0 = self.conv1x1(x.view(-1, *x.shape[-3:]))

        _x = x.view(-1, seq_len, *x.shape[-3:])
        b, t, c, w, h = _x.shape
        pad = torch.zeros((b, n_ahead, c, w, h), device=current_device)
        _x = torch.cat((_x, pad), dim=1)

        x1, hidden_state_1 = self.convgru1.forward(_x)
        x2, hidden_state_2 = self.convgru2.forward(_x)
        x3, hidden_state_3 = self.convgru3.forward(_x)

        if test:
            return hidden_state_1[-1], hidden_state_2[-1], hidden_state_3[-1]

        x1 = x1[:, -n_ahead:, :, :, :]
        x2 = x2[:, -n_ahead:, :, :, :]
        x3 = x3[:, -n_ahead:, :, :, :]

        return (
            x0,
            x1.reshape(-1, *x1.shape[-3:]),
            x2.reshape(-1, *x2.shape[-3:]),
            x3.reshape(-1, *x3.shape[-3:]),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, batch_norm: bool = True):
        super().__init__()
        block = []
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(in_channels))
        block.extend(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=1024,
                    kernel_size=(3, 3),
                    padding=1,
                ),
                nn.PixelShuffle(2),
                nn.Conv2d(
                    in_channels=256, out_channels=64, kernel_size=(3, 3), padding=1
                ),
            ]
        )
        self.model = nn.Sequential(*block)

    def forward(self, x):
        # print("DEC-BLK", x.shape)
        x = self.model(x)
        return x
