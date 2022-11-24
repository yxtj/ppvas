import torch
import torch.nn as nn

def flatten_model(model:nn.Module) -> list:
    l = list(model.children())
    if len(l) == 0:
        return model
    return [flatten_model(m) for m in l]


def compute_shape(model:nn.Module, inshape:tuple) -> list[tuple]:
    t = torch.zeros(inshape)
    shapes = [inshape]
    for i, lyr in enumerate(model):
        t = lyr(t)
        shapes.append(t.shape)
    return shapes

