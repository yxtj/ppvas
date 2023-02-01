# reference: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py

# import torch
import torch.nn as nn
import torch_extension as te

inshape = (3, 32, 32)

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 3, 
                     stride=stride, padding=1, bias=False)

def build_downsample_block(in_channels, out_channels, stride=2):
    layers = [
        conv3x3(in_channels, out_channels, stride),
        nn.ReLU(),
        conv3x3(out_channels, out_channels),
        # nn.ReLU(),
    ]
    if stride == 1:
        layers.append(te.ShortCut(-4))
    layers.append(nn.ReLU())
    return layers
    
def build_identity_block(channels):
    layers = [
        conv3x3(channels, channels),
        nn.ReLU(),
        conv3x3(channels, channels),
        te.ShortCut(-4),
        nn.ReLU(),
    ]
    return layers

def build_block(layer_size, in_channels, out_channels, stride):
    layers = build_downsample_block(in_channels, out_channels, stride)
    for i in range(layer_size-1):
        layers.extend(build_identity_block(out_channels))
    return layers

def build_resnet(num_blocks, num_class=100, version=1, residual=True):
    # layer 0: 3x32x32 -> 16x32x32
    layers = [ conv3x3(3, 16), nn.ReLU() ]
    # layer 1: 16x32x32 -> 16x32x32
    layers.extend(build_block(num_blocks[0], 16, 16, 1))
    # layer 2: 16x32x32 -> 32x16x16
    layers.extend(build_block(num_blocks[1], 16, 32, 2))
    # layer 3: 32x16x16 -> 64x8x8
    layers.extend(build_block(num_blocks[2], 32, 64, 2))
    # pooling and fc
    if version == 1: # cifar10
        layers.extend([
            nn.AvgPool2d(8),
            nn.Flatten(),
            nn.Linear(64, num_class),
        ])
    elif version == 2: # cifar10
        layers.extend([
            nn.Conv2d(64, 64, 8, 8, bias=False),
            nn.Flatten(),
            nn.Linear(64, num_class),
        ])
    elif version == 3: # cifar100
        layers.extend([
            nn.Flatten(),
            nn.Linear(4096, num_class),
        ])
    elif version == 4: # cifar100
        layers.extend([
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, num_class),
        ])
    if not residual:
        layers = [ lyr for lyr in layers if not isinstance(lyr, te.ShortCut) ]
    layers.append(nn.Softmax(1))
    return te.SequentialBuffer(*layers)


def resnet20(version=3, residual=True):
    return build_resnet([3, 3, 3], 100, version, residual)

def resnet32(version=3, residual=True):
    return build_resnet([5, 5, 5], 100, version, residual)

def resnet44(version=3, residual=True):
    return build_resnet([7, 7, 7], 100, version, residual)

def resnet56(version=3, residual=True):
    return build_resnet([9, 9, 9], 100, version, residual)

def resnet110(version=3, residual=True):
    return build_resnet([18, 18, 18], 100, version, residual)

def resnet152(version=3, residual=True):
    return build_resnet([24, 24, 24], 100, version, residual)


def build(depth, version=3, residual=True):
    if depth == 20:
        return resnet20(version, residual)
    elif depth == 32:
        return resnet32(version, residual)
    elif depth == 44:
        return resnet44(version, residual)
    elif depth == 56:
        return resnet56(version, residual)
    elif depth == 110:
        return resnet110(version, residual)
    elif depth == 152:
        return resnet152(version, residual)
    else:
        raise ValueError("depth must be 20, 32, 44, 56, 110, or 152")
