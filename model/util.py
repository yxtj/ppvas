import torch
import torch.nn as nn
import layer
from torch_extension.shortcut import ShortCut


def compute_shape(model, inshape):
    if len(inshape) == 3:
        inshape = (1, *inshape)
    t = torch.zeros(inshape)
    shapes = [inshape]
    for i, lyr in enumerate(model):
        t = lyr(t)
        shapes.append(tuple(t.shape))
    return shapes

    
def make_client_model(socket, model, inshape, he):
    shapes = compute_shape(model, inshape)
    layers = []
    shortcuts = {}
    for i, lyr in enumerate(model):
        if isinstance(lyr, nn.Conv2d):
            if i == len(model) - 1:
                layers.append(layer.last_conv.LastConvClient(socket, shapes[i], shapes[i+1], he))
            else:
                layers.append(layer.conv.ConvClient(socket, shapes[i], shapes[i+1], he))
        elif isinstance(lyr, nn.Linear):
            if i == len(model) - 1:
                layers.append(layer.last_fc.LastFcClient(socket, shapes[i], shapes[i+1], he))
            else:
                layers.append(layer.fc.FcClient(socket, shapes[i], shapes[i+1], he))
        elif isinstance(lyr, nn.ReLU):
            layers.append(layer.relu.ReLUClient(socket, shapes[i], shapes[i+1], he))
        elif isinstance(lyr, nn.MaxPool2d):
            layers.append(layer.maxpool.MaxPoolClient(socket, shapes[i], shapes[i+1], he, lyr))
        elif isinstance(lyr, nn.AvgPool2d):
            layers.append(layer.avgpool.AvgPoolClient(socket, shapes[i], shapes[i+1], he, lyr))
        elif isinstance(lyr, nn.Flatten):
            layers.append(layer.flatten.FlattenClient(socket, shapes[i], shapes[i+1], he))
        elif isinstance(lyr, ShortCut):
            layers.append(layer.shortcut.ShortCutClient(socket, shapes[i], shapes[i+1], he))
            shortcuts[i] = i + lyr.otherlayer # lyr.otherlayer is a negative index
        elif isinstance(lyr, nn.Softmax):
            assert i == len(model) - 1
            layers.append(layer.softmax.SoftmaxClient(socket, shapes[i], shapes[i+1], he))
        else:
            raise Exception("Unknown layer type: " + str(lyr))
    return layers, shortcuts


def make_server_model(socket, model, inshape):
    shapes = compute_shape(model, inshape)
    layers = []
    shortcuts = {}
    for i, lyr in enumerate(model):
        if isinstance(lyr, nn.Conv2d):
            if i == len(model) - 1:
                layers.append(layer.last_conv.LastConvServer(socket, shapes[i], shapes[i+1], lyr))
            else:
                layers.append(layer.conv.ConvServer(socket, shapes[i], shapes[i+1], lyr))
        elif isinstance(lyr, nn.Linear):
            if i == len(model) - 1:
                layers.append(layer.last_fc.LastFcServer(socket, shapes[i], shapes[i+1], lyr))
            else:
                layers.append(layer.fc.FcServer(socket, shapes[i], shapes[i+1], lyr))
        elif isinstance(lyr, nn.ReLU):
            layers.append(layer.relu.ReLUServer(socket, shapes[i], shapes[i+1], lyr))
        elif isinstance(lyr, nn.MaxPool2d):
            layers.append(layer.maxpool.MaxPoolServer(socket, shapes[i], shapes[i+1], lyr))
        elif isinstance(lyr, nn.AvgPool2d):
            layers.append(layer.avgpool.AvgPoolServer(socket, shapes[i], shapes[i+1], lyr))
        elif isinstance(lyr, nn.Flatten):
            layers.append(layer.flatten.FlattenServer(socket, shapes[i], shapes[i+1], lyr))
        elif isinstance(lyr, ShortCut):
            layers.append(layer.shortcut.ShortCutServer(socket, shapes[i], shapes[i+1], lyr))
            shortcuts[i] = i + lyr.otherlayer # lyr.otherlayer is a negative index
        elif isinstance(lyr, nn.Softmax):
            layers.append(layer.softmax.SoftmaxServer(socket, shapes[i], shapes[i+1], lyr))
        else:
            raise Exception("Unknown layer type: " + str(lyr))
    return layers, shortcuts
