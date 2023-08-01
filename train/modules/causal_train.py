import torch
import numpy as np
import pandas as pd
from ..callback import Callback
from typing import TYPE_CHECKING
from utils.sample import uniform_sample
from visualize.visual_history import plot_xy
import tempfile
import imageio
import shutil


if TYPE_CHECKING:
    from ..pinn import PINN


class CausalTrain(Callback):
    def on_train_begin(self, pinn: 'PINN'):
        self.weights_history = pd.DataFrame(columns=['epoch'])
        self.loss_history = pd.DataFrame(columns=['epoch'])
        
        self.Nt = pinn.config.causal_Nt
        self.tol = pinn.config.causal_tol
        
        self.t_min = pinn.config.lower_bound[-1]
        self.t_max = pinn.config.upper_bound[-1]
        
        self.Ws = torch.ones(self.Nt).to(pinn.config.device)
        self.M = torch.tensor(np.triu(torch.ones(self.Nt, self.Nt), k=1).T).to(pinn.config.device)
        
        if not pinn.config.pde_data_path and pinn.config.causal_sample:
            Xs = uniform_sample(pinn.config.lower_bound, pinn.config.upper_bound, pinn.config.causal_sample_num)
            pinn.set_Xs([Xs[:, i:i+1] for i in range(Xs.shape[1])])
        
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
        # add to history
        self.loss_history.loc[pinn.current_epoch, 'epoch'] = pinn.current_epoch
        self.weights_history.loc[pinn.current_epoch, 'epoch'] = pinn.current_epoch
        for i in range(self.Nt):
            self.loss_history.loc[pinn.current_epoch, f'L{i}'] = Lt[i].detach().cpu().tolist()
            self.weights_history.loc[pinn.current_epoch, f'W{i}'] = self.Ws[i].detach().cpu().tolist()
        # to_csv
        self.loss_history.to_csv(f'{pinn.config.log_dir}/causal_loss_history.csv', index=False)
        self.weights_history.to_csv(f'{pinn.config.log_dir}/causal_weights_history.csv', index=False)
        
        # Causal sampling
        if pinn.config.causal_sample and pinn.current_epoch % pinn.config.causal_sample_freq == 0:
            # ratio = w_i*Li / sum(w_i * Li)
            # uniform generate according to lower and upper bound
            sample_num = pinn.config.causal_sample_num
            Xs = []
            ratio = self.Ws * Lt / (self.Ws * Lt).sum()
            for i in range(self.Nt):
                t_0 = t_list[i]
                t_1 = t_list[i+1]
                sample_num_i = int(sample_num * ratio[i])
                Xs.append(uniform_sample(pinn.config.lower_bound, pinn.config.upper_bound, sample_num_i))
            cated_Xs = torch.cat(Xs, dim=0)
            sampled_Xs = [cated_Xs[:, i:i+1] for i in range(cated_Xs.shape[1])]
            pinn.set_Xs(sampled_Xs)

        if pinn.current_epoch % pinn.config.causal_train_freq == 0:
            # 更新Ws
            with torch.no_grad():
                self.Ws = torch.exp(-self.tol * torch.matmul(self.M, Lt))
            print(f"Ws: {self.Ws.tolist()}")
        loss = pinn.config.causal_L0_weight * boundary_loss + (self.Ws * Lt).mean()
        return loss
    