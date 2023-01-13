import sys
import torch
import socket

from model import poc
import system


if __name__ == '__main__':
    argv = sys.argv
    if len(argv) < 3:
        print('Usage: python model server|client [host=localhost] [port=8100]')
        sys.exit(1)
    model_name = argv[1]
    mode = argv[2]
    assert mode in ['server', 'client']
    host = argv[3] if len(argv) > 3 else 'localhost'
    port = int(argv[4]) if len(argv) > 4 else 8100
    
    # set model and inshape
    if model_name == 'poc1':
        model, inshape = poc.Poc1Model, poc.Poc1Inshape
    elif model_name == 'poc2':
        model, inshape = poc.Poc2Model, poc.Poc2Inshape
    elif model_name == 'poc3':
        model, inshape = poc.Poc3Model, poc.Poc3Inshape
    else:
        raise ValueError("Unknown model name: {}".format(model_name))
    
    if mode == 'server':
        s = socket.create_server((host, port))
        print("Server is running on {}:{}".format(host, port))
        conn, addr = s.accept()
        print("Client connected from: {}".format(addr))
        server = system.Server(conn, poc.Poc1Model, poc.ishape)
        print("Server is ready")
        server.offline()
        print("Server offline finished")
        server.online()
        print("Server online finished")
    else:
        s = socket.create_connection((host, port))
        print("Client is connecting to {}:{}".format(host, port))
        client = system.Client(s, poc.Poc1Model, poc.ishape)
        print("Client is ready")
        client.offline()
        print("Client offline finished")
        inshape = (1, *poc.ishape)
        data = torch.rand(inshape)
        res = client.online(data)
        print("Client online finished")
        print("System result: {}".format(res))
        res2 = poc.Poc1Model(data)
        print("Local result: {}".format(res2))
        diff = (res - res2).abs().sum()
        print("Difference: {}".format(diff))
