import torch
import numpy as np
from ..callback import Callback
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..pinn import PINN


class CausalTrain(Callback):
    def on_train_begin(self, pinn: 'PINN'):
        self.Nt = pinn.config.causal_Nt
        self.tol = pinn.config.causal_tol
        
        self.t_min = pinn.config.lower_bound[-1]
        self.t_max = pinn.config.upper_bound[-1]
        
        self.Ws = torch.ones(self.Nt).to(pinn.config.device)
        self.M = torch.tensor(np.triu(torch.ones(self.Nt, self.Nt), k=1).T).to(pinn.config.device)
        
    def on_calc_loss(self, pinn, residual, boundary_loss):
        Lt = torch.zeros(self.Nt).to(pinn.config.device)
        t_list = torch.linspace(self.t_min, self.t_max, self.Nt + 1).to(pinn.config.device)
        t = pinn.pde_Xs[-1]
        for i in range(self.Nt):
            t_0 = t_list[i]
            t_1 = t_list[i+1]
            # 取出对应区间的residual
            mask = (t >= t_0) & (t < t_1)
            mask = mask.squeeze()
            Lt[i] = residual[mask].pow(2).mean()

        if pinn.current_epoch % pinn.config.causal_train_freq == 0:
            # 更新Ws
            with torch.no_grad():
                self.Ws = torch.exp(-self.tol * torch.matmul(self.M, Lt))
            print(f"Ws: {self.Ws.tolist()}")
        loss = pinn.config.causal_L0_weight * boundary_loss + (self.Ws * Lt).mean()
        return loss
    