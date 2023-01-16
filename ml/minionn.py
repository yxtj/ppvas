import torch
import torch.nn as nn
import sys

import model.minionn as minionn
import ml.util as util

if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) == 0:
        print("Usage: python minionn.py data_dir chkpt_dir epochs batch_size dump_interval lr device")
        sys.exit(1)
    data_dir = argv[0] # 'E:/Data/CIFAR10'
    chkpt_dir = argv[1] # 'pretrained/'
    epochs = int(argv[2]) if len(argv) > 2 else 100
    batch_size = int(argv[3]) if len(argv) > 3 else 512
    dump_interval = int(argv[4]) if len(argv) > 4 else 10
    lr = float(argv[5]) if len(argv) > 5 else 0.001
    device = argv[6] if len(argv) > 6 else 'cpu'
    
    file, epn = util.find_latest_model(chkpt_dir, 'minionn_')
    if file is None:
        print("No pretrained model found")
    else:
        print("Loading model from {}".format(file))
    model = minionn.build(file)
    model = util.add_softmax(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    trainset, testset = util.load_data('cifar10', data_dir, True, True)
    util.process(model, trainset, testset, batch_size, epochs, optimizer, loss_fn,
                 dump_interval, chkpt_dir, 'minionn_', epn, device)
