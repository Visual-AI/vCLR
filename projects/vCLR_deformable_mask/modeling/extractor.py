import torch

class Extractor(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net 
    
    def forward(self, x):
        return self.net(x)