import torch
from pytorch_lightning.metrics import SSIM
from torch import device as torch_device, cuda as torch_cuda

from project.model.model import SelfSupervisedVideoPredictionModel

device = torch_device("cuda:0" if torch_cuda.is_available() else "cpu")

IMG_DIM = 224
block_inp_dims = [IMG_DIM // v for v in (2, 4, 8, 16)]

model = SelfSupervisedVideoPredictionModel(
    hidden_dims=[64, 64, 128, 256], latent_block_dims=block_inp_dims
)
model = model.cuda()

inp = torch.randn((1, 5, 3, IMG_DIM, IMG_DIM)).to(device)
_inp = inp.view(-1, 3, IMG_DIM, IMG_DIM)

## TRAIN
x = model(_inp)
print(
    "TRAIN pred:",
    x.shape,
    ", expected:",
    [inp.shape[0] * 3, 3, IMG_DIM // 2, IMG_DIM // 2],
)
loss = SSIM(data_range=1.0)
l = loss(
    torch.nn.functional.interpolate(
        inp[:, -3:, :, :, :].reshape(-1, 3, 224, 224), size=(112, 112)
    ),
    x,
)
print(l)
# exit()

## TEST
out_size = (1, 1)
hidden_states = model.forward(_inp, test=True, pooling_out_size=out_size)
print("TEST hidden:", hidden_states.shape, ", expected out sz:", out_size)
# hidden_states = torch.unbind(hidden_states)
# for hs in hidden_states:
#     print('HIDDEN', hs.shape)
