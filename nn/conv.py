import torch


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
        self.locationBias = torch.nn.Parameter(torch.zeros(w, h, 3))
        self.locationEncode = torch.autograd.Variable(torch.ones(w, h, 3))
        if gradient:
            for i in range(w):
                self.locationEncode[i, :, 1] = self.locationEncode[:, i, 0] = i / float(
                    w - 1
                )

    def forward(self, inputs):
        if self.locationBias.device != inputs.device:
            self.locationBias = self.locationBias.to(inputs.get_device())
        if self.locationEncode.device != inputs.device:
            self.locationEncode = self.locationEncode.to(inputs.get_device())
        b = self.locationBias * self.locationEncode
        return super().forward(inputs) + b[:, :, 0] + b[:, :, 1] + b[:, :, 2]
