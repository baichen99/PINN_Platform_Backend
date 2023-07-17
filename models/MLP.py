import torch
from torch import nn

class MLP(torch.nn.Module):
    def __init__(self, seq_net, act: nn.Module):
        # seq_net: [3, 20, 1]
        super(MLP, self).__init__()
        self.seq_net = seq_net
        self.num_layers = len(seq_net)
        self.layers = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.layers.append(torch.nn.Linear(seq_net[i], seq_net[i + 1]))
            self.layers.append(act)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x