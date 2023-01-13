import sys
import time
import torch
import socket
from Pyfhel import Pyfhel

from model import minionn
import system



if __name__ == '__main__':
    argv = sys.argv
    if len(argv) < 3:
        print('Usage: python server|client [host=localhost] [port=8100]')
        sys.exit(1)
    # model_name = argv[1]
    mode = argv[1]
    assert mode in ['server', 'client']
    host = argv[2] if len(argv) > 2 else 'localhost'
    port = int(argv[3]) if len(argv) > 3 else 8100
    
    # set model and inshape
    inshape = minionn.inshape
    model = minionn.model
    
    if mode == 'server':
        s = socket.create_server((host, port))
        print("Server is running on {}:{}".format(host, port))
        conn, addr = s.accept()
        print("Client connected from: {}".format(addr))
        t0 = time.time()
        server = system.Server(conn, model, inshape)
        print("Server is ready")
        server.offline()
        t1 = time.time()
        print("Server offline finished")
        server.online()
        t2 = time.time()
        print("Server online finished")
        print("Statistics: ")
        print("Offline time: {:.2f}; Online time: {:.2f}".format(t1 - t0, t2 - t1))
        for i, lyr in enumerate(server.layers):
            print("  Layer {} {}: {}".format(i, lyr.__class__.__name__, lyr.stat))
    else:
        he = Pyfhel()
        he.contextGen(scheme='ckks', n=2**13, scale=2**30, qi_sizes=[30]*5)
        s = socket.create_connection((host, port))
        print("Client is connecting to {}:{}".format(host, port))
        t0 = time.time()
        client = system.Client(s, model, inshape, he)
        print("Client is ready")
        client.offline()
        t1 = time.time()
        print("Client offline finished")
        inshape = (1, *inshape)
        data = torch.rand(inshape)
        with torch.no_grad():
            res = client.online(data)
        t2 = time.time()
        print("Client online finished")
        print("System result: {}".format(res))
        with torch.no_grad():
            res2 = model(data)
        print("Local result: {}".format(res2))
        diff = (res - res2).abs().sum()
        print("Difference: {}".format(diff))
        print("Statistics: ")
        print("Offline time: {:.2f}; Online time: {:.2f}".format(t1 - t0, t2 - t1))
        for i, lyr in enumerate(client.layers):
            print("  Layer {} {}: {}".format(i, lyr.__class__.__name__, lyr.stat))
