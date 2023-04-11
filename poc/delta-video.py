# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 21:35:48 2023

@author: yanxi
"""

import cv2
import numpy as np
import time


class FrameDiffMask():
    def __init__(self, frame=None, threshold=0.2):
        assert 0<=threshold<=1
        self.th = int(threshold*255)
        self.last = self.get_gray(frame) if frame is not None else None

    def get_gray(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray

    def apply(self, frame, last=None):
        gray = self.get_gray(frame)
        if last is None:
            last = self.last
        if len(last.shape) == 3:
            last = self.get_gray(last)
        frame_diff = cv2.absdiff(gray, last)
        _, mask = cv2.threshold(frame_diff, self.th, 255, cv2.THRESH_BINARY)
        self.last = gray
        return mask

    def pick_with_mask(self, frame, mask):
        color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        f = color_mask & frame
        return f

    def pick_color_diff(self, mask, frame1, frame2):
        f1 = self.pick_with_mask(frame1, mask)
        f2 = self.pick_with_mask(frame2, mask)
        f = f2-f1
        return f

    def get_bounding_box(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        # pick the largest contour
        # cnt = max(contours, key=cv2.contourArea)
        # x, y, w, h = cv2.boundingRect(cnt)
        # get the bounding rect of all contours
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        return x, y, w, h

    def pick_data_in_bb(self, frame, bb):
        x, y, w, h = bb
        return frame[y:y+h, x:x+w, :]

def get_bounding_box(mask, threshold=0.1):
    _, thresh = cv2.threshold(mask, 255 * threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    # pick the largest contour
    # cnt = max(contours, key=cv2.contourArea)
    # x, y, w, h = cv2.boundingRect(cnt)
    # get the bounding rect of all contours
    x, y, w, h = cv2.boundingRect(np.concatenate(contours))
    return x, y, w, h


# %% video

video_name="E:/Data/video/pose_video_dataset/002_dance.mp4"
cap = cv2.VideoCapture(video_name)
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)
L = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# %% template

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

ret,frame=cap.read()
fdm = FrameDiffMask(frame)

t = time.time()
i=0
while True:
    ret,frame=cap.read()
    if ret == False: break
    if i%1000 == 0:
        print('processed',i,'frames')
    i += 1
    mask = fdm.apply(frame)
    # actions

t = time.time() - t

# %% basic functions

import torch
import torchvision


@torch.no_grad()
def get_buff_features(x, model):
    return model.features(x)

@torch.no_grad()
def delta_general(x, model, buff):
    x = model.features(x)

    _,_,h,w = x.shape
    _,_,fh,fw = buff.shape
    i = np.random.randint(0, fh-h) if h < fh else 0
    j = np.random.randint(0, fw-w) if w < fw else 0
    buff[:,:,i:i+h,j:j+w] += x
    x = buff
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    x = model.classifier(x)
    return x, buff

def run_full_model(cap, model, n, itv=1000):
    ret,frame=cap.read()

    t = time.time()
    i=0
    while i<n:
        ret,frame=cap.read()
        if ret == False: break
        if i%itv == 0:
            print('processed',i,'frames')
        i += 1
        x=torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)/255).permute(2,0,1).unsqueeze(0).float().cuda()
        with torch.no_grad():
            y = model(x)
    t = time.time() - t
    print(t)
    return t

def run_delta_model(cap, model, sz, get_buff, delta_forward, n, itv=1000):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret,frame=cap.read()
    fdm = FrameDiffMask(frame)

    x=torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)/255).permute(2,0,1).unsqueeze(0).float().cuda()
    buff = get_buff(x)

    t = time.time()
    i=0
    while i<30*60:
        ret,frame=cap.read()
        if ret == False: break
        if i%1000 == 0:
            print('processed',i,'frames')
        i += 1
        mask = fdm.apply(frame)
        bb = fdm.get_bounding_box(mask)
        if bb is None or bb[2]*bb[3]<36: continue
        if bb[2]<sz or bb[3]<sz:
            bb = (min(W-sz, bb[0]), min(H-sz, bb[1]), max(sz, bb[2]), max(sz, bb[3]))
        delta = fdm.pick_data_in_bb(frame, bb)
        x=torch.from_numpy(cv2.cvtColor(delta, cv2.COLOR_BGR2RGB)/255).permute(2,0,1).unsqueeze(0).float().cuda()
        with torch.no_grad():
            y, buff = delta_forward(x, model, buff)
    t = time.time() - t
    print(t)
    return t

# %% resnet

MINISZ=36

@torch.no_grad()
def get_buff_resnet(x, model):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    return x

@torch.no_grad()
def delta_resnet(x, model, buff):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    _,_,h,w = x.shape
    _,_,fh,fw = buff.shape
    i = np.random.randint(0, fh-h) if h < fh else 0
    j = np.random.randint(0, fw-w) if w < fw else 0
    buff[:,:,i:i+h,j:j+w] += x
    x = buff
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    x = model.fc(x)
    return x, buff

model=torchvision.models.resnet18(pretrained=True).cuda()
model=torchvision.models.resnet50(pretrained=True).cuda()
run_full_model(cap, model, 30*60)
run_delta_model(cap, model, MINISZ, get_buff_resnet, delta_resnet, 30*60)

# %% vgg
MINISZ=32

model=torchvision.models.vgg19(pretrained=True).cuda()
run_full_model(cap, model, 30*60)
run_delta_model(cap, model, MINISZ, get_buff_features, delta_general, 30*60)

# %% mobilenet
MINISZ=32

model=torchvision.models.mobilenet_v3_large(pretrained=True).cuda()
run_full_model(cap, model, 30*60)
run_delta_model(cap, model, MINISZ, get_buff_features, delta_general, 30*60)

# %% alexnet
MINSZ=63

model=torchvision.models.alexnet(pretrained=True).cuda()
run_full_model(cap, model, 30*60)
run_delta_model(cap, model, MINISZ, get_buff_features, delta_general, 30*60)


