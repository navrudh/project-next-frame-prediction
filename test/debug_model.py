import torch
from pytorch_lightning.metrics import SSIM

from model.model import SelfSupervisedVideoPredictionModel

IMG_DIM = 224
block_inp_dims = [IMG_DIM // v for v in (2, 4, 8, 16)]

model = SelfSupervisedVideoPredictionModel(latent_block_dims=block_inp_dims)

b = 2
t = 3
inp = torch.randn((b, t, 3, IMG_DIM, IMG_DIM))
_inp = inp.view(-1, 3, IMG_DIM, IMG_DIM)

## TRAIN
pred, hidden = model.forward(inp, hidden=None)
# print(type(hidden))
# print(len(hidden))
# print(type(hidden[0]))
# print(len(hidden[0]))
# print(type(hidden[0][0]))
# print(len(hidden[0][0]))
# print(type(hidden[0][1]))
# print(len(hidden[0][1]))
# print(type(hidden[0][0][0]))
# print(hidden[0][0][0].shape)


print(
    "TRAIN pred:", pred.shape, ", expected:", [b * t, 3, IMG_DIM // 2, IMG_DIM // 2],
)
loss = SSIM(data_range=1.0)
l = loss(
    torch.nn.functional.interpolate(
        inp[:, -1:, :, :, :].reshape(-1, 3, 224, 224), size=(112, 112)
    ),
    pred[t - 1 :: t, :, :, :],
)
print(l)
exit()

## TEST
out_size = (1, 1)
hidden_states = model.forward(_inp, test=True, pooling_out_size=out_size)
print("TEST hidden:", hidden_states.shape, ", expected out sz:", out_size)
# hidden_states = torch.unbind(hidden_states)
# for hs in hidden_states:
#     print('HIDDEN', hs.shape)
