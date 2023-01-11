import sys
import torch
import socket

from model import poc1
import system


if __name__ == '__main__':
    argv = sys.argv
    if len(argv) < 2:
        print('Usage: python [server|client] [host=localhost] [port=8100]')
        sys.exit(1)
    mode = argv[1]
    assert mode in ['server', 'client']
    host = argv[2] if len(argv) >= 3 else 'localhost'
    port = int(argv[3]) if len(argv) >= 4 else 8100
    if mode == 'server':
        s = socket.create_server((host, port))
        print("Server is running on {}:{}".format(host, port))
        conn, addr = s.accept()
        print("Client connected from: {}".format(addr))
        server = system.Server(conn, poc1.Poc1Model, poc1.ishape)
        print("Server is ready")
        server.offline()
        print("Server offline finished")
        server.online()
        print("Server online finished")
    else:
        s = socket.create_connection((host, port))
        print("Client is connecting to {}:{}".format(host, port))
        client = system.Client(s, poc1.Poc1Model, poc1.ishape)
        print("Client is ready")
        client.offline()
        print("Client offline finished")
        inshape = (1, *poc1.ishape)
        data = torch.rand(inshape)
        res = client.online(data)
        print("Client online finished")
        print("Client online result: {}".format(res))
