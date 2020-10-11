import torch
from torch import nn

from nn.conv import LocationAwareConv2d
from nn.convgru import ConvGRU
from nn.convgru2 import ConvGRU as ConvGRU2


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
        self.lstm_dims = [output_dim]

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

        self.convgru1 = ConvGRU(
            input_size=(input_sz, input_sz),
            input_dim=input_dim,
            hidden_dim=self.lstm_dims,
            num_layers=len(self.lstm_dims),
            kernel_size=(3, 3),
        )
        self.convgru2 = ConvGRU(
            input_size=(input_sz, input_sz),
            input_dim=input_dim,
            hidden_dim=self.lstm_dims,
            num_layers=len(self.lstm_dims),
            kernel_size=(5, 5),
        )
        self.convgru3 = ConvGRU(
            input_size=(input_sz, input_sz),
            input_dim=input_dim,
            hidden_dim=self.lstm_dims,
            num_layers=len(self.lstm_dims),
            kernel_size=(7, 7),
        )

    def forward(self, x, hidden):
        b, t, c, w, h = x.shape
        x0 = self.conv1x1(x.clone().view(-1, c, w, h))

        _x = x

        if hidden is None:
            hidden_state_1 = self.convgru1.get_init_states(b)
            hidden_state_2 = self.convgru2.get_init_states(b)
            hidden_state_3 = self.convgru3.get_init_states(b)
        else:
            hidden_state_1, hidden_state_2, hidden_state_3 = hidden

        x1, hidden_state_1 = self.convgru1.forward(_x.clone(), hidden_state_1)
        x2, hidden_state_2 = self.convgru2.forward(_x.clone(), hidden_state_2)
        x3, hidden_state_3 = self.convgru3.forward(_x.clone(), hidden_state_3)

        return (
            (
                x0,
                x1.reshape(-1, *x1.shape[-3:]),
                x2.reshape(-1, *x2.shape[-3:]),
                x3.reshape(-1, *x3.shape[-3:]),
            ),
            (hidden_state_1, hidden_state_2, hidden_state_3),
        )


class ConvolutionalRecurrentBlocks(nn.Module):
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

        self.convgru = ConvGRU2(
            input_size=input_dim,
            hidden_sizes=output_dim,
            kernel_sizes=[3, 5, 7],
            n_layers=3,
        )

    def forward(self, x, hidden):
        print("crb.forward")
        b, t, c, w, h = x.shape
        x0 = self.conv1x1(x.view(-1, *x.shape[-3:]))

        layer_outs = []
        inputs = torch.unbind(x, dim=1)
        for idx, inp in enumerate(inputs):
            hidden = self.convgru.forward(inp, hidden)
            layer_outs.append(hidden)

            print(len(hidden))
            print([o.shape for o in hidden])

        return (
            (x0, hidden[-1]),
            hidden,
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, batch_norm: bool = True):
        super().__init__()
        block = []
        block.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
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
