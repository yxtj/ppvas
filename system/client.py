import socket
import numpy as np
import cv2
from videoutil.difference import FrameDiffMask
from .util import send_numpy, recv_numpy

class Client():
    def __init__(self, host, port, bufsize, model):
        self.host = host
        self.port = port
        self.bufsize = bufsize
        self.model = model
        self.fdm = FrameDiffMask()
        self.s = None
    
    def process_iframe(self, data: np.ndarray):
        self.s.sendall(b'iframe')
        reply = self.s.recv(self.bufsize)
        assert reply == b'ok'
        send_numpy(self.s, data)
        self.mpc_activation()
        r = recv_numpy(self.s)
        return r
    
    def process_dframe(self, data, ref):
        self.s.sendall(b'dframe')
        reply = self.s.recv(self.bufsize)
        assert reply == b'ok'
        send_numpy(self.s, data)
        self.mpc_activation()
        r = recv_numpy(self.s)
        return r
    
    def mpc_activation(self):
        pass
    
    def connect_server(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.host, self.port))
    
    def run(self, video_name, interval=10):
        cap = cv2.VideoCapture(video_name)
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        L = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"video info: H={H}, W={W}, fps={fps}, L={L}")
        
        interval = int(np.ceil(fps) * interval)
        
        result = []
        for i in range(L):
            ret, frame = cap.read()
            if i % interval == 0:
                ref = frame
                r = self.process_iframe(frame)
            else:
                r = self.process_dframe(frame, ref)
            result.append(r)
        return result
