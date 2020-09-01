import torch
from torch import device as torch_device, cuda as torch_cuda

device = torch_device("cuda:0" if torch_cuda.is_available() else "cpu")


class LocationAwareConv2d(torch.nn.Conv2d):
    def __init__(
        self,
        gradient,
        w,
        h,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.locationBias = torch.nn.Parameter(torch.zeros(w, h, 3, device=device))
        self.locationEncode = torch.autograd.Variable(
            torch.ones(w, h, 3, device=device)
        )
        if gradient:
            for i in range(w):
                self.locationEncode[i, :, 1] = self.locationEncode[:, i, 0] = i / float(
                    w - 1
                )

    def forward(self, inputs):
        b = self.locationBias * self.locationEncode
        return super().forward(inputs) + b[:, :, 0] + b[:, :, 1] + b[:, :, 2]
