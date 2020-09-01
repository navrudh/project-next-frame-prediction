import torch
from pytorch_lightning.metrics import SSIM
from torch import nn, device as torch_device, cuda as torch_cuda

from project.model.model import SelfSupervisedVideoPredictionModel

device = torch_device("cuda:0" if torch_cuda.is_available() else "cpu")

IMG_DIM = 224
block_inp_dims = [IMG_DIM // v for v in (2, 4, 8, 16)]

model = SelfSupervisedVideoPredictionModel(
    hidden_dims=[64, 64, 128, 256], latent_block_dims=block_inp_dims, batch_size=2
)
model = model.cuda()

inp = torch.randn((2, 5, 3, 224, 224)).to(device)
_inp = inp.view(-1, 3, 224, 224)
x = model(_inp)
print("TEST pred", x.shape)
loss = SSIM(data_range=1.0)
l = loss(
    nn.functional.interpolate(
        inp[:, -3:, :, :, :].reshape(-1, 3, 224, 224), size=(112, 112)
    ),
    x,
)
print(l)
exit()
