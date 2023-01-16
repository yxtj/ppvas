import torch
import torch.nn as nn
import sys

import model.resnet as resnet
import ml.util as util


if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) == 0:
        print("Usage: python resnet.py data_dir chkpt_dir epochs batch_size dump_interval lr device model_version")
        sys.exit(1)
    data_dir = argv[0] # 'E:/Data/CIFAR100'
    chkpt_dir = argv[1] # 'pretrained/'
    epochs = int(argv[2]) if len(argv) > 2 else 100
    batch_size = int(argv[3]) if len(argv) > 3 else 512
    dump_interval = int(argv[4]) if len(argv) > 4 else 10
    lr = float(argv[5]) if len(argv) > 5 else 0.001
    device = argv[6] if len(argv) > 6 else 'cpu'
    model_version = int(argv[7]) if len(argv) > 7 else 1
    assert model_version in [1, 2, 3, 4]
    
    prefix= 'resnet-' + str(model_version) + '_'
    
    model = resnet.resnet32(model_version)
    file, epn = util.find_latest_model(chkpt_dir, prefix)
    if file is None:
        print("No pretrained model found")
    else:
        print("Loading model from {}".format(file))
        util.load_model_state(model, file)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    trainset, testset = util.load_data('cifar100', data_dir, True, True)
    util.process(model, trainset, testset, batch_size, epochs, optimizer, loss_fn,
                 dump_interval, chkpt_dir, prefix, device)
