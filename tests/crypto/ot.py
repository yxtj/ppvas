from comm import ot
import sys
import socket
import time
import numpy as np
import multiprocessing as mp

host, port = 'localhost', 12345

def run_server(nbits, msg_len, n):
    prefix = '[Server]'
    sock = socket.create_server(('localhost', 12345))
    conn, addr = sock.accept()
    print(prefix, 'connection from', addr)
    server = ot.ObliviousTransferServer(conn, nbits)
    print(prefix, "setup")
    server.setup()
    for i in range(n):
        b = ot.Crypto.Random.random.randint(0, 1)
        print(prefix, f"Run {i}: choose {b}")
        t = time.time()
        data, n_send, n_recv = server.run(b)
        t = time.time() - t
        print(prefix, "send", n_send, "recv", n_recv, "time", t)
        print(prefix, "recv:", data)


def run_client(nbits, msg_len, n):
    prefix = '[Client]'
    sock = socket.create_connection(('localhost', 12345))
    print(prefix, "connected to server")
    client = ot.ObliviousTransferClient(sock, nbits)
    print(prefix, "setup")
    client.setup()
    m0= b'THIS IS MESSAGE ONE.'
    m1 = b'this is message two.'
    lm = len(m0)
    m0 = m0 * (msg_len//lm) + m0[:msg_len%lm]
    m1 = m1 * (msg_len//lm) + m1[:msg_len%lm]
    print(prefix, "msg0:", m0)
    print(prefix, "msg1:", m1)
    for i in range(n):
        print(prefix, f"Run {i}")
        t = time.time()
        n_send, n_recv = client.run(m0, m1)
        t = time.time() - t
        print(prefix, "send", n_send, "recv", n_recv, "time", t)


if __name__ == '__main__':
    # parse args
    args = sys.argv[1:]
    if len(args) > 3:
        print('usage: python ot.py nbits=1024 msg_len=128 n=1')
        sys.exit(1)
    # mode = args[0]
    nbits = int(args[0]) if len(args) > 0 else 1024
    msg_len = int(args[1]) if len(args) > 1 else 128
    n = int(args[2]) if len(args) > 2 else 1
    # run
    ps = mp.Process(target=run_server, args=(nbits, msg_len, n))
    ps.start()
    pc = mp.Process(target=run_client, args=(nbits, msg_len, n))
    pc.start()
    ps.join()
    pc.join()
