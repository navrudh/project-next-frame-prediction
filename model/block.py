from torch import nn

from project.nn.conv import LocationAwareConv2d
from project.nn.convgru import ConvGRU


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

        convgru_input_dim = input_dim
        if self.output_dim != self.input_dim:
            self.fix_filter_depth_conv = nn.Conv2d(
                in_channels=input_dim, out_channels=output_dim, kernel_size=1
            )
            convgru_input_dim = output_dim

        self.convgru1 = ConvGRU(
            input_size=(input_sz, input_sz),
            input_dim=convgru_input_dim,
            hidden_dim=self.lstm_dims,
            num_layers=len(self.lstm_dims),
            kernel_size=(3, 3),
            n_step_ahead=3,
        )
        self.convgru2 = ConvGRU(
            input_size=(input_sz, input_sz),
            input_dim=convgru_input_dim,
            hidden_dim=self.lstm_dims,
            num_layers=len(self.lstm_dims),
            kernel_size=(5, 5),
            n_step_ahead=3,
        )
        self.convgru3 = ConvGRU(
            input_size=(input_sz, input_sz),
            input_dim=convgru_input_dim,
            hidden_dim=self.lstm_dims,
            num_layers=len(self.lstm_dims),
            kernel_size=(7, 7),
            n_step_ahead=3,
        )

    def forward(self, x, test=False):
        b, t, c, w, h = x.shape
        x0 = self.conv1x1(x.view(-1, *x.shape[-3:]))

        _x = x
        if self.output_dim != self.input_dim:
            _x = self.fix_filter_depth_conv(_x.view(-1, *_x.shape[-3:]))
            _x = _x.view(b, t, *_x.shape[-3:])

        x1, hidden_state_1 = self.convgru1.forward(_x)
        x2, hidden_state_2 = self.convgru2.forward(_x)
        x3, hidden_state_3 = self.convgru3.forward(_x)

        if test:
            return hidden_state_1[-1], hidden_state_2[-1], hidden_state_3[-1]

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
        x = self.model(x)
        return x
