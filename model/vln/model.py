import torch

from nn.conv import LocationAwareConv2d
import math


class VLNDecoderBlock(torch.nn.Module):
    def __init__(self, w, h, in_channels, kernel):
        super().__init__()
        self.BN = torch.nn.ModuleList(
            [torch.nn.BatchNorm2d(in_channels), torch.nn.BatchNorm2d(in_channels)]
        )
        self.deConv0 = torch.nn.ConvTranspose2d(
            in_channels, in_channels, kernel, stride=2, padding=1
        )
        # torch.nn.init.xavier_uniform_(self.deConv0.weight)
        self.LReLU = torch.nn.LeakyReLU(0.2)
        # print("Omitted Padding =", 1)
        self.conv0 = LocationAwareConv2d(
            w, h, in_channels, in_channels, kernel, stride=1, padding=0
        )
        # torch.nn.init.xavier_uniform_(self.conv0.weight)

    def forward(self, inputs):
        resDeConv0 = self.deConv0(
            self.BN[0](inputs),
            output_size=[
                inputs.shape[0],
                inputs.shape[1],
                inputs.shape[2] * 2,
                inputs.shape[3] * 2,
            ],
        )
        return (
            self.conv0(self.LReLU(self.conv0(self.BN[1](self.LReLU(resDeConv0)))))
            + resDeConv0
        )


class VLNEncoderBlock(torch.nn.Module):
    def __init__(self, dilation, w, h, in_channels, out_channels, kernel):
        super().__init__()
        self.BN = torch.nn.ModuleList(
            [torch.nn.BatchNorm2d(in_channels), torch.nn.BatchNorm2d(out_channels)]
        )
        self.LReLU = torch.nn.LeakyReLU(0.2)
        self.ReLU = torch.nn.ReLU()
        print("Omitted Padding =", (kernel + (kernel - 1) * (dilation - 1)) // 2)
        self.conv0 = LocationAwareConv2d(
            False,
            False,
            w,
            h,
            in_channels,
            out_channels,
            kernel,
            padding=0,
            dilation=dilation,
        )
        torch.nn.init.xavier_uniform_(self.conv0.weight)
        print("Omitted Padding =", (kernel + (kernel - 1) * (2 * dilation - 1)) // 2)
        self.conv1 = LocationAwareConv2d(
            False,
            False,
            w,
            h,
            out_channels,
            out_channels,
            kernel,
            padding=int(2 * dilation) - 1,
            dilation=int(2 * dilation),
        )
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv3 = torch.nn.Conv2d(
            in_channels, out_channels, 1, padding=0, dilation=dilation
        )
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.pooling = torch.nn.MaxPool2d([2, 2])

        # bn1 = torch.nn.BatchNorm2d(in_channels)
        # lrelu1 = torch.nn.LeakyReLU(0.2)
        # dilated_conv1 = torch.nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=dilation)
        # bn2 = torch.nn.BatchNorm2d(in_channels)
        # lrelu2 = lrelu1
        # dilated_conv2 = torch.nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=dilation)
        # strided_conv = torch.nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=dilation)
        # relu = torch.nn.ReLU()

    def forward(self, inputs):
        nIn = self.BN[0](inputs.contiguous())
        res = self.BN[1](self.conv0(self.LReLU(nIn)))
        return self.ReLU(self.pooling(self.conv1(self.LReLU(res)) + self.conv3(inputs)))


class VLN(torch.nn.Module):
    def __init__(self, w, h, channels, kernel):
        super().__init__()
        self.channels = channels
        self.encoders = torch.nn.ModuleList()
        for i in range(len(channels)):
            self.encoders.append(
                VLNEncoderBlock(
                    w=int(w // (math.pow(2, i))),
                    h=int(h // (math.pow(2, i))),
                    dilation=int(math.pow(2, i)),
                    in_channels=(3 if AI else 1) if i < 1 else channels[i - 1],
                    out_channels=channels[i],
                    kernel=kernel,
                )
            )

        self.decoders = torch.nn.ModuleList()
        for i in range(len(channels)):
            self.decoders.append(
                VLNDecoderBlock(
                    w=int(w // (math.pow(2, i))),
                    h=int(h // (math.pow(2, i))),
                    in_channels=channels[i],
                    kernel=kernel,
                )
            )

        self.convLSTMs = torch.nn.ModuleList()
        for i in range(len(channels)):
            self.convLSTMs.append(
                ConvLSTMCell(
                    w=int(w // (math.pow(2, i + 1))),
                    h=int(h // (math.pow(2, i + 1))),
                    input_size=channels[i],
                    hidden_size=channels[i],
                    kernel=kernel,
                )
            )

        self.convCompressors = torch.nn.ModuleList()
        for i in range(len(channels) * 2):
            if i % 2 == 0:
                in_channels = (0 if i < 1 else channels[(i // 2) - 1]) + channels[
                    i // 2
                ]
            else:
                in_channels = channels[i // 2] * 2
            self.convCompressors.append(
                torch.nn.Conv2d(in_channels, 1 if i < 1 else channels[(i - 1) // 2], 1)
            )
            torch.nn.init.xavier_uniform_(self.convCompressors[-1].weight)
        self.LReLu = torch.nn.LeakyReLU(0.2)

        self.locationEncode = torch.zeros(1, 2, w, h)
        for i in range(w):
            self.locationEncode[0, 1, :, i] = self.locationEncode[
                0, 0, i, :
            ] = i / float(w - 1)

    def forward(self, inputs):
        if self.locationEncode.device != inputs.device:
            self.locationEncode = self.locationEncode.to(inputs.get_device())

        hiddLSTM = [[None] for i in range(len(self.channels))]
        cellLSTM = [[None] for i in range(len(self.channels))]
        encoderActivation = [[None] for i in range(len(self.channels))]
        decoderActivation = [[None] for i in range(len(self.channels))]
        # getting ready to iterate through timesteps
        inputs = inputs.permute(1, 0, 4, 3, 2)
        res = [inputs[0]]
        # iterate through Timesteps except last
        for index, batch in enumerate(inputs[:-1]):
            for i in range(len(self.channels)):
                # ENC.0 INPUT => if index < seed frames; then use orig images else use prev prediction
                #
                # ENC.!0 INPUT => previous encoder activation
                #
                inputM = (
                    (batch if index < seedNumber else occludedDS(res[-1], val=0, dim=4))
                    if i < 1
                    else encoderActivation[i - 1][-1]
                )
                if AI and i < 1:
                    inputM = torch.cat(
                        (inputM, self.locationEncode.repeat(inputM.shape[0], 1, 1, 1)),
                        dim=1,
                    )
                # for each channel/block we encoder.fwd
                encoderActivation[i].append(self.encoders[i](inputM))
                # for each channel/block we conv-lstm.fwd
                h, c = self.convLSTMs[i](
                    encoderActivation[i][-1], hiddLSTM[i][-1], cellLSTM[i][-1]
                )
                # save hidden state
                hiddLSTM[i].append(h)
                cellLSTM[i].append(c)

            conI = (len(self.channels) * 2) - 1
            for i in range(len(self.channels) - 1, -1, -1):
                if i == len(self.channels) - 1:
                    #
                    # skip connection: last lstm and enc activation is used
                    tmp = torch.cat((encoderActivation[i][-1], hiddLSTM[i][-1]), 1)
                    # some down conv
                    tmp = self.convCompressors[conI](tmp)
                    conI = conI - 1
                    # feed into corresponding layer
                    decoderActivation[i].append(self.decoders[i](tmp))
                else:

                    tmp = torch.cat((decoderActivation[i + 1][-1], hiddLSTM[i][-1]), 1)
                    tmp = self.convCompressors[conI](tmp)
                    conI = conI - 1
                    tmp = self.LReLu(tmp)
                    tmp = torch.cat((encoderActivation[i][-1], tmp), 1)
                    tmp = self.convCompressors[conI](tmp)
                    conI = conI - 1
                    decoderActivation[i].append(self.decoders[i](tmp))

            # save each prediction
            res.append(torch.sigmoid(self.convCompressors[0](decoderActivation[0][-1])))

        # order [batch, timestep, ...]
        res = torch.stack(res).permute(1, 0, 4, 3, 2)
        return res, encoderActivation, hiddLSTM, cellLSTM, decoderActivation

    def forward(self, inputs):
        if self.locationEncode.device != inputs.device:
            self.locationEncode = self.locationEncode.to(inputs.get_device())

        hiddLSTM = [[None] for i in range(len(self.channels))]
        cellLSTM = [[None] for i in range(len(self.channels))]
        encoderActivation = [[None] for i in range(len(self.channels))]
        decoderActivation = [[None] for i in range(len(self.channels))]
        inputs = inputs.permute(1, 0, 4, 3, 2)
        res = [inputs[0]]
        for index, batch in enumerate(inputs[:-1]):
            for i in range(len(self.channels)):
                inputM = (
                    (batch if index < seedNumber else occludedDS(res[-1], val=0, dim=4))
                    if i < 1
                    else encoderActivation[i - 1][-1]
                )
                if AI and i < 1:
                    inputM = torch.cat(
                        (inputM, self.locationEncode.repeat(inputM.shape[0], 1, 1, 1)),
                        dim=1,
                    )
                encoderActivation[i].append(self.encoders[i](inputM))
                h, c = self.convLSTMs[i](
                    encoderActivation[i][-1], hiddLSTM[i][-1], cellLSTM[i][-1]
                )
                hiddLSTM[i].append(h)
                cellLSTM[i].append(c)

            conI = (len(self.channels) * 2) - 1
            for i in range(len(self.channels) - 1, -1, -1):
                if i == len(self.channels) - 1:
                    tmp = torch.cat((encoderActivation[i][-1], hiddLSTM[i][-1]), 1)
                    tmp = self.convCompressors[conI](tmp)
                    conI = conI - 1
                    decoderActivation[i].append(self.decoders[i](tmp))
                else:

                    tmp = torch.cat((decoderActivation[i + 1][-1], hiddLSTM[i][-1]), 1)
                    tmp = self.convCompressors[conI](tmp)
                    conI = conI - 1
                    tmp = self.LReLu(tmp)
                    tmp = torch.cat((encoderActivation[i][-1], tmp), 1)
                    tmp = self.convCompressors[conI](tmp)
                    conI = conI - 1
                    decoderActivation[i].append(self.decoders[i](tmp))

            res.append(torch.sigmoid(self.convCompressors[0](decoderActivation[0][-1])))

        res = torch.stack(res).permute(1, 0, 4, 3, 2)
        return res, encoderActivation, hiddLSTM, cellLSTM, decoderActivation
