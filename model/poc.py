# import torch
import torch.nn as nn
from torch_extension.shortcut import ShortCut

# Model 1:
# Shape: 1x10x10 -> 5x8x8  -> 10x6x6 -> 360 -> 10

Poc1Inshape = (1, 10, 10)
Poc1Model = nn.Sequential(
    nn.Conv2d(1, 5, 3),
    nn.ReLU(),
    nn.Conv2d(5, 10, 3),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(360, 10),
    # nn.Softmax()
)

# Model 2:
# Shape: 1x32x32 -conv-> 5x30x30 -max-> 5x10x10 -conv-> 10x8x8 -max-> 10x2x2 -> 40 -> 10

Poc2Inshape = (1, 32, 32)
Poc2Model = nn.Sequential(
    nn.Conv2d(1, 5, 3),
    nn.ReLU(),
    nn.MaxPool2d(3, 3),
    nn.Conv2d(5, 10, 3),
    nn.ReLU(),
    nn.MaxPool2d(3, 3),
    nn.Flatten(),
    nn.Linear(40, 10),
)

# Model 3:
# Shape: 20 -fc-> 10 -relu-> 10 -shortcut(-1,-3)-> 10 -> 5

Poc3Inshape = (20)
Poc3Model = nn.Sequential(
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    ShortCut(-3),
    nn.Linear(10, 5),
)
