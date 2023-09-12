import torch
from torch import nn

class PFNN(torch.nn.Module):
    """Parallel Feedforward Neural Network"""
    def __init__(self, seq_net, act: nn.Module):
        # seq_net: [3, 20, 1]
        super(PFNN, self).__init__()
        self.seq_net = seq_net
        self.num_layers = len(seq_net)
        self.nets = torch.nn.ModuleList()

        # 对每一个输出都有一个独立的网络
        for i in range(seq_net[-1]):
            net = torch.nn.ModuleList()
            for j in range(self.num_layers - 1):
                # 最后一层的输出为1
                if j == self.num_layers - 2:
                    net.append(torch.nn.Linear(seq_net[j], 1))
                else:
                    net.append(torch.nn.Linear(seq_net[j], seq_net[j + 1]))
                    net.append(act)
            self.nets.append(net)
    
    def forward(self, x):
        preds = []
        for net in self.nets:
            x_ = x.clone()
            for layer in net:
                x_ = layer(x_)
            preds.append(x_)
        return torch.cat(preds, dim=1)