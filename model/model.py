from typing import List

import torch
import torch.nn as nn
import torchvision

from project.model.block import LatentBlock, DecoderBlock


class SelfSupervisedVideoPredictionModel(nn.Module):
    def __init__(
        self, hidden_dims: List[int], latent_block_dims: List[int], batch_size: int = 32
    ):
        super().__init__()
        self.batch_size = batch_size
        self.enc2lateral_hook_layers = ["conv1", "layer1", "layer2", "layer3", "layer4"]
        self.encoder = torchvision.models.resnet18(pretrained=True)
        self.latent_block_0 = LatentBlock(
            hidden_dim=hidden_dims[0],
            input_dim=latent_block_dims[0],
            batch_size=batch_size,
        )
        self.latent_block_1 = LatentBlock(
            hidden_dim=hidden_dims[1],
            input_dim=latent_block_dims[1],
            batch_size=batch_size,
        )
        self.latent_block_2 = LatentBlock(
            hidden_dim=hidden_dims[2],
            input_dim=latent_block_dims[2],
            location_aware=True,
            batch_size=batch_size,
        )
        self.latent_block_3 = LatentBlock(
            hidden_dim=hidden_dims[3],
            input_dim=latent_block_dims[3],
            location_aware=True,
            batch_size=batch_size,
        )
        self.decoder_block_1 = DecoderBlock(in_channels=512, batch_norm=False)
        self.decoder_block_2 = DecoderBlock(in_channels=320)
        self.decoder_block_3 = DecoderBlock(in_channels=320)
        self.decoder_block_4 = DecoderBlock(in_channels=320)
        self.decoder_block_5 = nn.Conv2d(in_channels=320, out_channels=3, kernel_size=1)

        self.lateral_inputs = {}
        self._register_hooks()

    def parameters(self, recurse: bool = True):
        return (
            list(self.latent_block_0.parameters())
            + list(self.latent_block_1.parameters())
            + list(self.latent_block_2.parameters())
            + list(self.latent_block_3.parameters())
            + list(self.decoder_block_1.parameters())
            + list(self.decoder_block_2.parameters())
            + list(self.decoder_block_3.parameters())
            + list(self.decoder_block_4.parameters())
            + list(self.decoder_block_5.parameters())
        )

    def forward(self, x):
        self.encoder(x)
        lb0 = self.latent_block_0(self.lateral_inputs[self.enc2lateral_hook_layers[0]])
        lb1 = self.latent_block_1(self.lateral_inputs[self.enc2lateral_hook_layers[1]])
        lb2 = self.latent_block_2(self.lateral_inputs[self.enc2lateral_hook_layers[2]])
        lb3 = self.latent_block_3(self.lateral_inputs[self.enc2lateral_hook_layers[3]])

        lb0 = tuple(
            it.view(self.batch_size, -1, *it.shape[-3:])[:, -3:, :, :, :].reshape(
                -1, *it.shape[-3:]
            )
            for it in lb0
        )
        lb1 = tuple(
            it.view(self.batch_size, -1, *it.shape[-3:])[:, -3:, :, :, :].reshape(
                -1, *it.shape[-3:]
            )
            for it in lb1
        )
        lb2 = tuple(
            it.view(self.batch_size, -1, *it.shape[-3:])[:, -3:, :, :, :].reshape(
                -1, *it.shape[-3:]
            )
            for it in lb2
        )
        lb3 = tuple(
            it.view(self.batch_size, -1, *it.shape[-3:])[:, -3:, :, :, :].reshape(
                -1, *it.shape[-3:]
            )
            for it in lb3
        )

        x = self.lateral_inputs[self.enc2lateral_hook_layers[4]]
        # print("DEC-BLK-1", x.shape)
        x = self.decoder_block_1(x)
        x = x.view(self.batch_size, -1, *x.shape[-3:])[:, -3:, :, :, :].reshape(
            -1, *x.shape[-3:]
        )
        # print("DEC-BLK-2: CAT ", x.shape, [it.shape for it in lb3])
        x = torch.cat(lb3 + (x,), dim=1)
        x = self.decoder_block_2(x)
        # print("DEC-BLK-3: CAT ", x.shape, [it.shape for it in lb2])
        x = torch.cat(lb2 + (x,), dim=1)
        x = self.decoder_block_3(x)
        # print("DEC-BLK-4: CAT ", x.shape, [it.shape for it in lb1])
        x = torch.cat(lb1 + (x,), dim=1)
        x = self.decoder_block_4(x)
        # print("DEC-BLK-5: CAT ", x.shape, [it.shape for it in lb0])
        x = torch.cat(lb0 + (x,), dim=1)
        x = self.decoder_block_5(x)
        # print("DEC-BLK: OUT", x.shape)
        return x

    def _register_hooks(self):
        for n, m in self.encoder.named_modules():
            # print(n)
            if n in self.enc2lateral_hook_layers:
                m.register_forward_hook(self.get_module_output(n))
                print(f"registered hook\t({n})")

    def get_module_output(self, name):
        def hook(model, input, output):
            self.lateral_inputs[name] = output.detach_()

        return hook
