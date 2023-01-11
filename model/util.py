import torch
import torch.nn as nn
import layer
from torch_extension.shortcut import ShortCut


def compute_shape(model, inshape):
    t = torch.zeros(inshape)
    shapes = [inshape]
    for i, lyr in enumerate(model):
        t = lyr(t)
        shapes.append(t.shape)
    return shapes

    
def make_client_model(socket, model, inshape):
    shapes = compute_shape(model, inshape)
    layers = []
    for i, lyr in enumerate(model):
        if isinstance(lyr, nn.Conv2d):
            layers.append(layer.conv.Conv2DClient(socket, shapes[i], shapes[i+1]))
        elif isinstance(lyr, nn.Linear):
            if i == len(model) - 1:
                layers.append(layer.last_fc.LastFcClient(socket, shapes[i], shapes[i+1]))
            else:
                layers.append(layer.fc.FcClient(socket, shapes[i], shapes[i+1]))
        elif isinstance(lyr, nn.ReLU):
            layers.append(layer.relu.ReLUClient(socket, shapes[i], shapes[i+1]))
        elif isinstance(lyr, nn.MaxPool2d):
            layers.append(layer.maxpool.MaxPoolClient(socket, shapes[i], shapes[i+1]))
            layers[-1].setup(lyr)
        elif isinstance(lyr, nn.Flatten):
            layers.append(layer.flatten.FlattenClient(socket, shapes[i], shapes[i+1]))
        elif isinstance(lyr, ShortCut):
            offset = lyr.otherlayer # a negative integer
            rother = layers[offset].r
            layers.append(layer.shortcut.ShortCutClient(socket, shapes[i], shapes[i+1]))
            layers[-1].setup(offset, rother)
        elif isinstance(lyr, nn.Softmax):
            assert i == len(model) - 1
            layers.append(layer.softmax.SoftmaxClient(socket, shapes[i], shapes[i+1]))
        else:
            raise Exception("Unknown layer type: " + str(lyr))
    return layers


def make_server_model(socket, model, inshape):
    shapes = compute_shape(model, inshape)
    layers = []
    to_buffer = []
    mlast = 1
    for i, lyr in enumerate(model):
        if isinstance(lyr, nn.Conv2d):
            layers.append(layer.conv.Conv2DServer(socket, shapes[i], shapes[i+1], lyr, mlast))
        elif isinstance(lyr, nn.Linear):
            if i == len(model) - 1:
                layers.append(layer.last_fc.LastFcServer(socket, shapes[i], shapes[i+1], lyr, mlast))
            else:
                layers.append(layer.fc.FcServer(socket, shapes[i], shapes[i+1], lyr, mlast))
        elif isinstance(lyr, nn.ReLU):
            layers.append(layer.relu.ReLUServer(socket, shapes[i], shapes[i+1], lyr, mlast))
        elif isinstance(lyr, nn.MaxPool2d):
            layers.append(layer.maxpool.MaxPoolServer(socket, shapes[i], shapes[i+1], lyr, mlast))
        elif isinstance(lyr, nn.Flatten):
            layers.append(layer.flatten.FlattenServer(socket, shapes[i], shapes[i+1], lyr, mlast))
        elif isinstance(lyr, ShortCut):
            offset = lyr.otherlayer # a negative integer
            mother = layers[offset].m
            layers.append(layer.shortcut.ShortCutServer(socket, shapes[i], shapes[i+1], lyr, mlast))
            layers[-1].setup(mother)
            to_buffer.append(i + offset)
        elif isinstance(lyr, nn.Softmax):
            layers.append(layer.softmax.SoftmaxServer(socket, shapes[i], shapes[i+1], lyr, mlast))
        else:
            raise Exception("Unknown layer type: " + str(lyr))
        mlast = layers[-1].m
    return layers, to_buffer
