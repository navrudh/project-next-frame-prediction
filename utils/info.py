from pytorch_lightning import seed_everything
from torch import device as torch_device, cuda as torch_cuda


def print_device():
    device = torch_device("cuda:0" if torch_cuda.is_available() else "cpu")
    print("Device:", device)
    if torch_cuda.is_available():
        print("Device Name:", torch_cuda.get_device_name(0))


def seed(seed):
    seed_everything(seed)
    print("Set Random Seed:", seed)
