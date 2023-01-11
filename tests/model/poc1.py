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
        server = system.Server(s, poc1.Poc1Model, poc1.ishape)
    else:
        s = socket.create_connection((host, port))
        client = system.Client(s, poc1.Poc1Model, poc1.ishape)
