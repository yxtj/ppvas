# import torch
import torch.nn as nn
from torch_extension.shortcut import ShortCut

map = {}

# Model 0 linear:
# Shape: 6 -> 2

Poc0Inshape_l = (6,)
Poc0Model_l = nn.Sequential(
    nn.Linear(6, 2),
)
map["0-linear"] = (Poc0Inshape_l, Poc0Model_l)

# Model 0 pool:
# Shape: 1x6x6 -> 1x2x2 -> 4 -> 2

Poc0Inshape_p = (1, 6, 6)
Poc0Model_m = nn.Sequential(
    nn.MaxPool2d(3, 3),
    nn.Flatten(),
    nn.Linear(4, 2),
)
Poc0Model_a = nn.Sequential(
    nn.AvgPool2d(3, 3),
    nn.Flatten(),
    nn.Linear(4, 2),
)
map["0-max"] = (Poc0Inshape_p, Poc0Model_m)
map["0-avg"] = (Poc0Inshape_p, Poc0Model_a)

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
map["1"] = (Poc1Inshape, Poc1Model)

# Model 2:
# Shape: 1x10x10 -> 5x8x8 -> 5x2x2 -> 20 -> 10

Poc2Inshape = (1, 10, 10)
Poc2Model = nn.Sequential(
    nn.Conv2d(1, 5, 3),
    nn.ReLU(),
    nn.MaxPool2d(3, 3),
    nn.Flatten(),
    nn.Linear(20, 10),
)
map["2"] = (Poc2Inshape, Poc2Model)

Poc2Model_avg = nn.Sequential(
    nn.Conv2d(1, 5, 3),
    nn.ReLU(),
    nn.AvgPool2d(3, 3),
    nn.Flatten(),
    nn.Linear(20, 10),
)
map["2-avg"] = (Poc2Inshape, Poc2Model_avg)

# Model 3:
# Shape: 1x32x32 -conv-> 5x30x30 -max-> 5x10x10 -conv-> 10x8x8 -max-> 10x2x2 -> 40 -> 10

Poc3Inshape = (1, 32, 32)
Poc3Model = nn.Sequential(
    nn.Conv2d(1, 5, 3),
    nn.ReLU(),
    nn.MaxPool2d(3, 3),
    nn.Conv2d(5, 10, 3),
    nn.ReLU(),
    nn.MaxPool2d(3, 3),
    nn.Flatten(),
    nn.Linear(40, 10),
)
map["3"] = (Poc2Inshape, Poc2Model)

# Model 4:
# Shape: 10 -relu-> 10 -shortcut(-1,-3)-> 10 -> 5

Poc4Inshape = (10)
Poc4Model = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    ShortCut(-3),
    nn.Linear(10, 5),
)
map["4"] = (Poc3Inshape, Poc3Model)
