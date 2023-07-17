import torch

def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, 
                               grad_outputs=torch.ones_like(outputs),
                            #    retain_graph=True,
                               create_graph=True,
                               allow_unused=True)[0]
