import torch
from pytorch_lightning.metrics import SSIM

from model.model import SelfSupervisedVideoPredictionModel

IMG_DIM = 64
b = 2
t = 6
ch = 3
block_inp_dims = [IMG_DIM // v for v in (2, 4, 8, 16)]

model = SelfSupervisedVideoPredictionModel(
    latent_block_dims=block_inp_dims, seed_frames=3
)
# model = model.cuda()


inp = torch.randn((b, t, ch, IMG_DIM, IMG_DIM))  # .to(device)
_inp = inp.view(-1, ch, IMG_DIM, IMG_DIM)

## TRAIN
pred, hidden = model.forward(inp, hidden=None)
# print(hidden)
print(
    "TRAIN pred:", pred.shape, ", expected:", [b, t, ch, IMG_DIM, IMG_DIM],
)
loss = SSIM(data_range=1.0)
n_cmp = t-model.seed_frames
l = loss(
    inp[:, -n_cmp:, :, :, :].reshape(-1, ch, IMG_DIM, IMG_DIM),
    pred[:, -n_cmp:, :, :, :].reshape(-1, ch, IMG_DIM, IMG_DIM),
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
