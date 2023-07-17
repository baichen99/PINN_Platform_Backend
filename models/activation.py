from torch import nn
import torch

class AdaTanh(nn.Module):
    def __init__(self, alpha=1.0):
        super(AdaTanh, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)
        
    def forward(self, x):
        # hyperbolic tangent
        return (torch.exp(self.alpha * x) - torch.exp(-self.alpha * x)) / (torch.exp(self.alpha * x) + torch.exp(-self.alpha * x))
    
class AdaSigmoid(nn.Module):
    def __init__(self, alpha=1.0):
        super(AdaSigmoid, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)
        
    def forward(self, x):
        # sigmoid
        return 1 / (1 + torch.exp(-self.alpha * x))