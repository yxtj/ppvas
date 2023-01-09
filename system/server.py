import socket

class Server():
    def __init__(self, host, port, model) -> None:
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((host, port))
        self.socket.listen()
        conn, addr = self.socket.accept()
        if conn:
            print("Connected by", addr)
            data = conn.recv(1024)
        # model
        self.model = model