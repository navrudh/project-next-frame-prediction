from typing import List

import torch
import torchvision
from torch import nn

from project.model.block import LatentBlock, DecoderBlock


class SelfSupervisedVideoPredictionModel(nn.Module):
    def __init__(self, latent_block_dims: List[int]):
        super().__init__()
        hidden_dims = [64, 64, 128, 256]
        self.enc2lateral_hook_layers = ["conv1", "layer1", "layer2", "layer3", "layer4"]
        self.encoder = torchvision.models.resnet18(pretrained=True)
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

        self.lateral_inputs = {}
        self._register_hooks()

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

    def forward(self, x, test=False, pooling_out_size=(1, 1)):
        b, t, c, w, h = x.shape

        self.encoder(x.reshape(-1, c, w, h))

        decoder_inputs = []
        for idx, block in enumerate(self.latent_blocks):
            inp = self.lateral_inputs[self.enc2lateral_hook_layers[idx]]
            decoder_inputs.append(
                tuple(block.forward(inp.view(b, t, *inp.shape[-3:]), test=test))
            )

        if test:
            return torch.cat(
                tuple(
                    nn.functional.adaptive_avg_pool2d(
                        torch.cat(lb, dim=1), pooling_out_size
                    )
                    for lb in decoder_inputs
                ),
                dim=1,
            )

        decoder_inputs.append(self.lateral_inputs[self.enc2lateral_hook_layers[4]])

        output = None
        for dec_inp, dec_block in zip(reversed(decoder_inputs), self.decoder_blocks):
            if output is not None:
                output = torch.cat(dec_inp + (output,), dim=1)
            else:
                output = dec_inp
            output = dec_block(output)

        return output

    def _register_hooks(self):
        for n, m in self.encoder.named_modules():
            # print(n)
            if n in self.enc2lateral_hook_layers:
                m.register_forward_hook(self.get_module_output(n))
                print(f"registered hook\t({n})")

    def get_module_output(self, name):
        def hook(model, input, output):
            self.lateral_inputs[name] = output

        return hook

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
