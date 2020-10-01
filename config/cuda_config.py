import torch

current_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
