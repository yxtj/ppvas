# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
import copy

from PIL import Image
import torch
from torchvision import transforms

from videoutil.difference import FrameDiffMask

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
model.eval()

# normal model
model_i = model.featuers

# delta model
model_d = copy.deepcopy(model.features)
for lyr in model_d:
    if isinstance(lyr, torch.nn.Conv2d):
        lyr.weight.data.zero_()
        lyr.weight.data += 1
        lyr.bias.data.zero_()


video_name="E:/Data/video/pose_video_dataset/002_dance.mp4"
cap = cv2.VideoCapture(video_name)
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)
L = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


fdm = FrameDiffMask(threshold=0.2)

interval = int(np.ceil(fps)) * 10

t = time.time()
for i in range(L):
    frame = cap.read()[1]
    if i % interval == 0:
        ref = fdm.get_gray(frame)
        with torch.no_grad():
            o_ref = model_i(frame)
    else:
        o = model_d(frame)
        o_delta = o + o_ref
t = time.time() - t
print("time: ", t)
