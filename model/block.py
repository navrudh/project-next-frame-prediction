from torch import nn

from project.nn.conv import LocationAwareConv2d
from project.nn.convgru import ConvGRU


class LatentBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        input_dim: int,
        location_aware: bool = False,
        batch_size: int = 1,
    ):
        super().__init__()
        self.batch_size = batch_size
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
        )
        self.convgru2 = ConvGRU(
            input_size=(input_dim, input_dim),
            input_dim=hidden_dim,
            hidden_dim=self.lstm_dims,
            num_layers=len(self.lstm_dims),
            kernel_size=(5, 5),
        )
        self.convgru3 = ConvGRU(
            input_size=(input_dim, input_dim),
            input_dim=hidden_dim,
            hidden_dim=self.lstm_dims,
            num_layers=len(self.lstm_dims),
            kernel_size=(7, 7),
        )

    def forward(self, x):
        # print("LAT-BLK", x.shape)
        x0 = self.conv1x1(x)

        _x = x.view(self.batch_size, -1, *x.shape[1:])
        hidden = self.convgru1.get_init_states(self.batch_size)
        x1, _ = self.convgru1(_x, hidden)
        hidden = self.convgru2.get_init_states(self.batch_size)
        x2, _ = self.convgru2(_x, hidden)
        hidden = self.convgru3.get_init_states(self.batch_size)
        x3, _ = self.convgru3(_x, hidden)

        return x0, x1, x2, x3


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, batch_norm: bool = True):
        super().__init__()
        block = []
        block.append(nn.ReLU(inplace=True))
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
