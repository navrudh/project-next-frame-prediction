from torch import nn

from nn.conv import LocationAwareConv2d
from nn.convgru import ConvGRUCell


class LatentBlock(nn.Module):
    def __init__(
            self,
            input_dim: int,
            input_sz: int,
            location_aware: bool = False,
            output_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.input_sz = input_sz
        self.output_dim = output_dim

        if location_aware:
            self.conv1x1 = LocationAwareConv2d(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=1,
                w=input_sz,
                h=input_sz,
                gradient=None,
            )
        else:
            self.conv1x1 = nn.Conv2d(
                in_channels=input_dim, out_channels=output_dim, kernel_size=1
            )

        self.convgru1 = ConvGRUCell(
            input_size=(input_sz, input_sz),
            input_dim=input_dim,
            hidden_dim=self.output_dim,
            kernel_size=(3, 3),
        )
        self.convgru2 = ConvGRUCell(
            input_size=(input_sz, input_sz),
            input_dim=input_dim,
            hidden_dim=self.output_dim,
            kernel_size=(5, 5),
        )
        self.convgru3 = ConvGRUCell(
            input_size=(input_sz, input_sz),
            input_dim=input_dim,
            hidden_dim=self.output_dim,
            kernel_size=(7, 7),
        )

    def forward(self, x, hidden):
        b, c, w, h = x.shape
        x0 = self.conv1x1(x)

        if hidden is None:
            hidden_state_1 = self.convgru1.init_hidden(b)
            hidden_state_2 = self.convgru2.init_hidden(b)
            hidden_state_3 = self.convgru3.init_hidden(b)
        else:
            hidden_state_1, hidden_state_2, hidden_state_3 = hidden

        x1 = self.convgru1.forward(x, h_prev=hidden_state_1)
        x2 = self.convgru2.forward(x, h_prev=hidden_state_2)
        x3 = self.convgru3.forward(x, h_prev=hidden_state_3)

        return (
            (x0, x1, x2, x3,),
            (x1, x2, x3),
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
        x = self.model(x)
        return x
