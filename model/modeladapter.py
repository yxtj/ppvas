import torch
import torch.nn as nn
import numpy as np

class ModelAdapter:
    def __init__(self) -> None:
        self.inshape = None
        self.imodel = None # model for i-frame
        self.dmodel = None # model for d-frame
    
    def compute_shape(self, model, inshape):
        t = torch.zeros(inshape)
        shapes = [inshape]
        for i, lyr in enumerate(model):
            t = lyr(t)
            shapes.append(t.shape)
        return shapes
    
    def prepare_imodel(self, model):
        pass
    
    def prepare_dmodel(self, imodel):
        pass
    
    def setup_secret_share(self, shapes:list):
        pass
    
    def process_iframe(self, image:np.ndarray):
        pass
    
    def process_dframe(self, image:np.ndarray):
        pass
    