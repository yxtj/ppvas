import sys
import time
import torch
import socket
from Pyfhel import Pyfhel

from model import resnet
from system.runner import run_client, run_server

if __name__ == '__main__':
    argv = sys.argv
    if len(argv) < 2:
        print('Usage: python server|client [ver=3] [n=1] [host=localhost] [port=8100]')
        sys.exit(1)
    # model_name = argv[1]
    mode = argv[1]
    assert mode in ['server', 'client']
    ver = int(argv[2]) if len(argv) > 2 else 3
    n = int(argv[3]) if len(argv) > 3 else 1
    host = argv[4] if len(argv) > 4 else 'localhost'
    port = int(argv[5]) if len(argv) > 5 else 8100
    
    # set model and inshape
    inshape = resnet.inshape
    model = resnet.resnet32(ver)
    
    if mode == 'server':
        run_server(host, port, model, inshape, n)
    else:
        he = Pyfhel()
        he.contextGen(scheme='ckks', n=2**13, scale=2**30, qi_sizes=[30]*5)
        run_client(host, port, model, inshape, he, None, n)
