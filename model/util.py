import torch
import torch.nn as nn
from torch_extension.shortcut import ShortCut

import layer_smp as layer


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
    linears = [] # linear layers
    scl = {} # shortcut layers
    locals = [] # local layers
    for i, lyr in enumerate(model):
        if isinstance(lyr, nn.Conv2d):
            layers.append(layer.conv.ConvClient(socket, shapes[i], shapes[i+1], he))
            linears.append(i)
        elif isinstance(lyr, nn.Linear):
            layers.append(layer.fc.FcClient(socket, shapes[i], shapes[i+1], he))
            linears.append(i)
        elif isinstance(lyr, nn.ReLU):
            layers.append(layer.relu.ReLUClient(socket, shapes[i], shapes[i+1], he))
            locals.append(i)
        elif isinstance(lyr, nn.MaxPool2d):
            layers.append(layer.maxpool.MaxPoolClient(socket, shapes[i], shapes[i+1], he, lyr))
        elif isinstance(lyr, nn.AvgPool2d):
            layers.append(layer.avgpool.AvgPoolClient(socket, shapes[i], shapes[i+1], he, lyr))
            linears.append(i)
        elif isinstance(lyr, nn.Flatten):
            layers.append(layer.flatten.FlattenClient(socket, shapes[i], shapes[i+1], he))
            locals.append(i)
        elif isinstance(lyr, ShortCut):
            layers.append(layer.shortcut.ShortCutClient(socket, shapes[i], shapes[i+1], he))
            scl[i] = i + lyr.otherlayer # lyr.otherlayer is a negative index
        elif isinstance(lyr, nn.Softmax):
            assert i == len(model) - 1
            layers.append(layer.softmax.SoftmaxClient(socket, shapes[i], shapes[i+1], he))
            locals.append(i)
        else:
            raise Exception("Unknown layer type: " + str(lyr))
    # set shortcuts inputs
    shortcuts = {} # {shortcut layer idx: input layer idx}
    for idx, oidx in scl.items():
        oidx += 1 # move to the outputo of the layer
        if isinstance(layers[oidx], layer.base.LocalLayerClient):
            raise Exception("Shortcut input should not be a local layer.")
        shortcuts[idx] = oidx
    return layers, linears, shortcuts, locals


def make_server_model(socket, model, inshape):
    shapes = compute_shape(model, inshape)
    layers = []
    linears = [] # linear layers
    scl = {} # shortcut layers
    locals = [] # local layers
    for i, lyr in enumerate(model):
        if isinstance(lyr, nn.Conv2d):
            layers.append(layer.conv.ConvServer(socket, shapes[i], shapes[i+1], lyr))
            linears.append(i)
        elif isinstance(lyr, nn.Linear):
            layers.append(layer.fc.FcServer(socket, shapes[i], shapes[i+1], lyr))
            linears.append(i)
        elif isinstance(lyr, nn.ReLU):
            layers.append(layer.relu.ReLUServer(socket, shapes[i], shapes[i+1], lyr))
            locals.append(i)
        elif isinstance(lyr, nn.MaxPool2d):
            layers.append(layer.maxpool.MaxPoolServer(socket, shapes[i], shapes[i+1], lyr))
        elif isinstance(lyr, nn.AvgPool2d):
            layers.append(layer.avgpool.AvgPoolServer(socket, shapes[i], shapes[i+1], lyr))
            linears.append(i)
        elif isinstance(lyr, nn.Flatten):
            layers.append(layer.flatten.FlattenServer(socket, shapes[i], shapes[i+1], lyr))
            locals.append(i)
        elif isinstance(lyr, ShortCut):
            layers.append(layer.shortcut.ShortCutServer(socket, shapes[i], shapes[i+1], lyr))
            scl[i] = i + lyr.otherlayer # lyr.otherlayer is a negative index
        elif isinstance(lyr, nn.Softmax):
            layers.append(layer.softmax.SoftmaxServer(socket, shapes[i], shapes[i+1], lyr))
            locals.append(i)
        else:
            raise Exception("Unknown layer type: " + str(lyr))
    # set shortcuts inputs
    shortcuts = {} # {shortcut layer idx: input layer idx}
    for idx, oidx in scl.items():
        oidx += 1 # move to the outputo of the layer
        if isinstance(layers[oidx], layer.base.LocalLayerServer):
            raise Exception("Shortcut input should not be a local layer.")
        shortcuts[idx] = oidx
    return layers, linears, shortcuts, locals

def find_last_non_local_layer(num_layer, local_layers):
    for i in range(num_layer-1, -1, -1):
        if i not in local_layers:
            return i
    return -1
    