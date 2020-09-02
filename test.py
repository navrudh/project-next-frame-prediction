import torch
from torch import device as torch_device, cuda as torch_cuda

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

## TRAIN
# x = model(_inp)
# print("TEST pred", x.shape)
# loss = SSIM(data_range=1.0)
# l = loss(
#     nn.functional.interpolate(
#         inp[:, -3:, :, :, :].reshape(-1, 3, 224, 224), size=(112, 112)
#     ),
#     x,
# )
# print(l)
# exit()

## TEST
hidden_states = model.forward(_inp, test=True)
print("HIDDEN", hidden_states.shape)
# hidden_states = torch.unbind(hidden_states)
# for hs in hidden_states:
#     print('HIDDEN', hs.shape)
