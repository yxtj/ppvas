import torch
import torch.nn as nn
import sys

import model.minionn as minionn
import ml.util as util


def train(data_dir, chkpt_dir, model, lr=0.001, epochs=100, batch_size=512,
          epoch_start=0, device='cpu',):
    trainset, _ = util.load_data('cifar10', data_dir, True, False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    interval = min(10, epochs)
    i = 0
    while i < epochs:
        print("Training from epoch {} to {}".format(epoch_start+i, epoch_start+i+interval))
        i += interval
        util.train(model, trainset, batch_size, interval,
                   optimizer=optimizer, loss_fn=loss_fn, show_interval=60, device=device)
        file = chkpt_dir + "/minionn_" + str(epoch_start+i) + ".pt"
        print("Saving model to {}".format(file))
        util.save_model_state(model, file)
        interval = min(10, epochs - i)


def test(data_dir, model, batch_size, device):
    _, testset = util.load_data('cifar10', data_dir, False, True)
    util.test(model, testset, batch_size, show_interval=60, device=device)


if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) == 0:
        pass
    data_dir = argv[0] # 'E:/Data/CIFAR10'
    chkpt_dir = argv[1] # 'pretrained/'
    epochs = int(argv[2]) if len(argv) > 2 else 100
    batch_size = int(argv[3]) if len(argv) > 3 else 512
    lr = float(argv[4]) if len(argv) > 4 else 0.001
    device = argv[5] if len(argv) > 5 else 'cpu'
    
    file, epn = util.find_latest_model(chkpt_dir, 'minionn_')
    if file is None:
        print("No pretrained model found")
    else:
        print("Loading model from {}".format(file))
    model = minionn.build(file)
    model = util.add_softmax(model)
    
    train(data_dir, chkpt_dir, model, lr, epochs, batch_size, epn, device)
    test(data_dir, model, batch_size, device)
