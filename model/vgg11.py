import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import copy
from model.modeladapter import ModelAdapter, flatten_model



class VGG11(ModelAdapter):
    def __init__(self) -> None:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
        model.eval()
        model = model.features
        model = flatten_model(model)
        
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.inshape = (1, 3, 224, 224)
        self.shapes = self.compute_shape(model, self.inshape)
        
        self.imodel = self.prepare_imodel(model)
        self.dmodel = self.prepare_dmodel(model)

    
    def prepare_imodel(self, model):
        res = []
        for i, lyr in enumerate(model):
            if isinstance(lyr, (nn.Conv2d, nn.Linear)):
                res.append(lyr)
            if isinstance(lyr, nn.MaxPool2d):
                lyr = nn.AvgPool2d(lyr.kernel_size, lyr.stride, lyr.padding, lyr.ceil_mode, lyr.count_include_pad)
                res.append(lyr)
            if isinstance(lyr, nn.ReLU):
                res.append(lyr)
        return model
    
    def prepare_dmodel(self, imodel):
        withBias = False
        for lyr in imodel:
            if isinstance(lyr, (nn.Conv2d, nn.Linear)):
                if lyr.bias is not None:
                    withBias = True
                    break
        if withBias == False:
            return imodel
        
        model = copy.deepcopy(imodel)
        for lyr in model:
            if isinstance(lyr, (nn.Conv2d, nn.Linear)):
                lyr.bias.data.zero_()
        return model
    
    def compute_shape(self, model, inshape):
        t = torch.zeros(inshape)
        shapes = [inshape]
        for i, lyr in enumerate(model):
            t = lyr(t)
            shapes.append(t.shape)
        return shapes
    
    def setup_secret_share(self, shapes:list):
        pass

    def process_iframe(self, data):
        img_t = self.preprocess(data)
        d = torch.unsqueeze(img_t, 0)
        mid = []
        with torch.no_grad():
            for i, lyr in enumerate(self.imodel):
                if isinstance(lyr, (nn.Conv2d, nn.Linear)):
                    d = lyr(d)
                elif isinstance(lyr, nn.AvgPool2d):
                    d = lyr(d)
                elif isinstance(lyr, nn.ReLU):
                    d = lyr(d)
                else:
                    print('Unknown layer type: ', lyr)
                    d = lyr(d)
                mid.append(d)
        return d, mid
    
    def process_dframe(self, data):
        return super().process_dframe(data)