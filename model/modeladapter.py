import torch
import torch.nn as nn
import numpy as np

def flatten_model(model:nn.Module):
    l = list(model.children())
    if len(l) == 0:
        return model
    return [flatten_model(m) for m in l]

class ModelAdapter:
    def __init__(self, model:nn.Module) -> None:
        self.model = model
        model.eval()
        self.imodel = None # model for i-frame
        self.dmodel = None # model for d-frame
    
    def process_model(self, model):
        for lyr in model:
            if isinstance(lyr, (nn.Conv2d, nn.Linear)):
                pass
            elif isinstance(lyr, nn.ReLU):
                pass
            elif isinstance(lyr, nn.MaxPool2d):
                pass
    
    def setup_secret_share(self, shapes:list):
        pass
    
    def process_iframe(self, data:np.ndarray):
        pass
    
    def process_dframe(self, data:np.ndarray):
        pass
    