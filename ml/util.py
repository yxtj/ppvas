import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

def load_data(name: str, folder: str, train: bool = True, test: bool = False):
    if name.lower() == 'cifar10':
        dsc = torchvision.datasets.CIFAR10
        tsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
        ])
    elif name.lower() == 'cifar100':
        dsc = torchvision.datasets.CIFAR100
        tsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047))
        ])
    elif name.lower() == 'mnist':
        dsc = torchvision.datasets.MNIST
        tsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.1307, 0.3081)
        ])
    # load data
    dataset_train = dsc(root=folder, train=True, download=True, transform=tsfm) if train else None
    dataset_test = dsc(root=folder, train=False, download=True, transform=tsfm) if test else None
    return dataset_train, dataset_test


def add_softmax(model):
    if not isinstance(model[-1], nn.Softmax):
        model.add_module(str(len(model)), nn.Softmax(dim=1))
    return model

def find_latest_model(folder: str, prefix: str) -> tuple[str, int]:
    import os
    files = os.listdir(folder)
    files = [f for f in files if os.path.isfile(folder+"/"+f) and f.startswith(prefix) and f.endswith('.pt')]
    latest = -1
    for f in files:
        try:
            t = int(f[len(prefix):-3])
            if t > latest:
                latest = t
        except:
            pass
    if latest > 0:
        return "{}/{}{}.pt".format(folder, prefix, latest), latest
    return None, None

def save_model_state(model, path: str):
    torch.save(model.state_dict(), path)

def load_model_state(model, path: str):
    model.load_state_dict(torch.load(path))
    return model


def train(model, dataset, batch_size: int=32, epochs: int=10, shuffle: bool=True,
          optimizer: torch.optim.Optimizer = None, loss_fn: nn.Module = None,
          *, n: int=None, show_interval: float = 60, device: str = 'cpu'):
    if device != 'cpu':
        assert torch.cuda.is_available(), 'CUDA is not available'
    model.to(device)
    model.train()
    # dataloader
    if n is None:
        n = len(dataset)
    if device == 'cpu':
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        # pin_memory_device is available for PyTorch >= 1.12
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        #                                          pin_memory=True, pin_memory_device=device)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    # optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # loss function
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    # train
    t0 = time.time()
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        running_loss = 0.0
        total = 0
        t1 = time.time()
        t2 = time.time()
        for batch in dataloader:
            # get data
            data, target = batch
            if device != 'cpu':
                data = data.to(device)
                target = target.to(device)
            # forward
            output = model(data)
            # loss
            loss = loss_fn(output, target)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print statistics
            total += len(target)
            running_loss += loss.item()
            if total >= n:
                break
            t3 = time.time()
            t = t3 - t2
            if t >= show_interval:
                t2 = t3
                eta = t / total * (n - total)
                print('  Progress {:.1f}% ({}/{}). Loss: {:.6g}% Time: {:.2f}s ETA: {:.2f}s'.format(
                    total / n, total, n, running_loss, time.time() - t2, eta))
        t = time.time() - t1
        eta = t / (epoch + 1) * (epochs - epoch - 1)
        print('  Epoch {}: Loss: {:.6g} Time: {:.2f}s ETA: {:.2f}s'.format(epoch, running_loss, t, eta))
    print('Finished Training. Time: {:.2f}s'.format(time.time()-t0))


def test(model, dataset, batch_size: int = 32, *, n: int = None, show_interval: float = 60, device: str = 'cpu'):
    if device != 'cpu':
        assert torch.cuda.is_available(), 'CUDA is not available'
    model.to(device)
    model.eval()
    # dataloader
    if n is None:
        n = len(dataset)
    if device == 'cpu':
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    else:
        # pin_memory_device is available for PyTorch >= 1.12
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
        #                                          pin_memory=True, pin_memory_device=device)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    # test
    t0 = time.time()
    correct = 0
    total = 0
    with torch.no_grad():
        t1 = time.time()
        for batch in dataloader:
            # get data
            data, target = batch
            if device != 'cpu':
                data = data.to(device)
                target = target.to(device)
            # forward
            output = model(data)
            # get prediction
            _, predicted = torch.max(output.data, 1)
            # count
            total += target.size(0)
            correct += (predicted == target).sum().item()
            if total >= n:
                break
            if time.time() - t1 >= show_interval:
                t = time.time() - t1
                t1 = time.time()
                eta = t / total * (n - total)
                print('  Progress {:.1f}% ({}/{}). Accuracy: {:.2f}% Time: {:.2f}s ETA: {:.2f}s'.format(
                    total / n, total, n, 100 * correct / total, t, eta))
    print('Accuracy: {:.2f}% ({}/{}) Time: {:.2f}s'.format(
        100 * correct / total, correct, total, time.time()-t0))
    return correct / total
