# import torch
import torch.nn as nn
# import socket

# from . import util
# import layer

# Model: conv 3x3,5 -> relu -> conv 3x3,10 -> relu -> flatten -> fc 10
# Shape: 1x10x10 -> 5x8x8  -> 10x6x6 -> 360 -> 10

ishape = (1, 10, 10)
Poc1Model = nn.Sequential(
    nn.Conv2d(1, 5, 3),
    nn.ReLU(),
    nn.Conv2d(5, 10, 3),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(360, 10),
    # nn.Softmax()
)
