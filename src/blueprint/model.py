import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from pytorch_grad_cam import GradCAMElementWise
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.net = EfficientNet.from_pretrained("efficientnet-b4", advprop=True, num_classes=2)

    def forward(self, x):
        x = self.net(x)
        return x

def create_model(path="Weights/94_0.9485_val.tar", device=torch.device('cpu')):
    model = Detector()
    try:
        if device.type == 'cuda':
            model = model.half()
    except:
        model = model.float()
    model = model.to(device)
    if device == torch.device('cpu'):
        cnn_sd = torch.load(path, map_location=torch.device('cpu'))["model"]
    else:
        cnn_sd = torch.load(path)["model"]
    model.load_state_dict(cnn_sd)
    model.eval()
    return model

def create_cam(model):
    target_layers = [model.net._blocks[-1]]
    targets = [ClassifierOutputTarget(1)]
    cam_algorithm = GradCAMElementWise
    use_cuda = torch.cuda.is_available() and next(model.parameters()).is_cuda
    cam = cam_algorithm(model=model, target_layers=target_layers, use_cuda=use_cuda)
    return cam