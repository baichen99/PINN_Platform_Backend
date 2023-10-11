import torch
import numpy as np
import pandas as pd
from ..callback import Callback
from typing import TYPE_CHECKING
from utils.sample import uniform_sample


if TYPE_CHECKING:
    from ..pinn import PINN


class RAR(Callback):
    def on_epoch_begin(self, pinn: 'PINN'):
        if pinn.config.RAR and pinn.current_epoch % pinn.config.RAR_freq == 0:
            Xs = uniform_sample(pinn.config.lower_bound, pinn.config.upper_bound, pinn.config.RAR_num)
            # to device
            Xs = [X.to(pinn.device).requires_grad_() for X in Xs]
            # cal residual
            residual = pinn._compute_residual(Xs).sum(dim=1)
            # select top k residual using argsort
            topk = torch.argsort(residual)[:pinn.config.RAR_top_k]
            # update pde_Xs
            topk_Xs = [X[topk] for X in Xs]
            print(f'Residual based adaptive refinement')
            pinn.append_Xs(topk_Xs)
    