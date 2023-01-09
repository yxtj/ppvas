import socket


class Client():
    def __init__(self, host, port, model):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))
        self.socket.sendall(b"Hello, world")
        # model
        self.model = model
        