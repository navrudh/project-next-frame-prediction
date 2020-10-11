from typing import List

import torch
import torchvision
from torch import nn

from model.block import LatentBlock, DecoderBlock


class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        resnet18 = torchvision.models.resnet18(pretrained=True)
        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4

    def forward(self, x):
        x = self.conv1(x)
        x0 = x
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x1 = x
        x = self.layer2(x)
        x2 = x
        x = self.layer3(x)
        x3 = x
        x = self.layer4(x)
        x4 = x

        return x0, x1, x2, x3, x4


class SelfSupervisedVideoPredictionModel(nn.Module):
    def __init__(self, latent_block_dims: List[int]):
        super().__init__()
        hidden_dims = [64, 64, 128, 256]
        self.encoder = Resnet18()
        self.latent_block_0 = LatentBlock(
            input_dim=hidden_dims[0], input_sz=latent_block_dims[0],
        )
        self.latent_block_1 = LatentBlock(
            input_dim=hidden_dims[1], input_sz=latent_block_dims[1],
        )
        self.latent_block_2 = LatentBlock(
            input_dim=hidden_dims[2],
            input_sz=latent_block_dims[2],
            location_aware=True,
        )
        self.latent_block_3 = LatentBlock(
            input_dim=hidden_dims[3],
            input_sz=latent_block_dims[3],
            location_aware=True,
        )
        self.decoder_block_1 = DecoderBlock(in_channels=512, batch_norm=False)
        self.decoder_block_2 = DecoderBlock(in_channels=320)
        self.decoder_block_3 = DecoderBlock(in_channels=320)
        self.decoder_block_4 = DecoderBlock(in_channels=320)
        self.decoder_block_5 = nn.Conv2d(in_channels=320, out_channels=3, kernel_size=1)

        self.latent_blocks = (
            self.latent_block_0,
            self.latent_block_1,
            self.latent_block_2,
            self.latent_block_3,
        )
        self.decoder_blocks = (
            self.decoder_block_1,
            self.decoder_block_2,
            self.decoder_block_3,
            self.decoder_block_4,
            self.decoder_block_5,
        )

    def forward(self, x, hidden=None, pooling_out_size=(1, 1)):
        # print("model.forward")
        b, t, c, w, h = x.shape

        # 1. ENCODER
        encoder_outputs = self.encoder(x.reshape(-1, c, w, h))

        # 2. INTERMEDIATE
        decoder_inputs = []
        if hidden is None:
            hidden = [None] * 4
        for idx, block in enumerate(self.latent_blocks):
            inp = encoder_outputs[idx]
            outputs, hidden[idx] = block.forward(
                inp.view(b, t, *inp.shape[-3:]), hidden=hidden[idx]
            )
            decoder_inputs.append(tuple(outputs))

        # if test:
        #     return torch.cat(
        #         tuple(
        #             nn.functional.adaptive_avg_pool2d(
        #                 torch.cat(lb, dim=1), pooling_out_size
        #             )
        #             for lb in decoder_inputs
        #         ),
        #         dim=1,
        #     )

        decoder_inputs.append(encoder_outputs[4])

        # 3. DECODER
        output = None
        for dec_inp, dec_block in zip(reversed(decoder_inputs), self.decoder_blocks):
            if output is not None:
                output = torch.cat(dec_inp + (output,), dim=1)
            else:
                output = dec_inp
            output = dec_block(output)

        return torch.sigmoid_(output), hidden

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
