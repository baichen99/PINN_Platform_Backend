from config.common import CommonConfig
import torch
import pandas as pd
import numpy as np
import pathlib


class Callbacks:
    def __init__(self, pinn_config: CommonConfig):
        # ------------log------------
        self.config = pinn_config
        self.df = pd.DataFrame(columns=['epoch'])
        
        # ------------adaptive loss------------
        if self.config.adaptive_loss:
            self.log_sigmas = [
                torch.tensor(2.0, dtype=torch.float32, requires_grad=True, device=self.config.device) for i in range(2)
            ]
        # ------------Causal sampling------------
        if self.config.causal_sampling:
            self.ws = [
                torch.tensor(1.0, dtype=torch.float32, requires_grad=True, device=self.config.device) for i in range(self.config.causal_time_step)
            ]

    def on_train_begin(self, pinn):
        ...
            
    def on_train_end(self, pinn):
        self.df.to_csv(self.config.log_filename, index=False)
    
    def on_calc_loss(self, pinn, pde_loss, boundary_loss):
        # ------------adaptive loss------------
        if not self.config.pde_weights:
            self.config.pde_weights = [1.0] * pde_loss.shape[0]
        if not self.config.bc_weights:
            self.config.bc_weights = [1.0] * boundary_loss.shape[0]
        pde_loss_w = torch.dot(pde_loss, torch.tensor(self.config.pde_weights).to(self.config.device))
        boundary_loss_w = torch.dot(boundary_loss, torch.tensor(self.config.bc_weights).to(self.config.device))
        
        
        if self.config.adaptive_loss:
            # w = 1 / 2sigma.pow(2)
            # 这里只有data_loss和pde_loss需要动态权重
            # self adaptive loss = w1 * pde_loss + w2 * boundary_loss + log(sigma1) + log(sigma2)
            self.ws = [1 / (2 * torch.exp(log_sigma).pow(2)) for log_sigma in self.log_sigmas]
            
            loss = pde_loss_w * self.ws[0] + boundary_loss_w * self.ws[1] + self.log_sigmas[0] + self.log_sigmas[1]
        else:
            loss = pde_loss_w + boundary_loss_w
            
        
        # ------------log------------
        for i in range(pde_loss.shape[0]):
            self.add_scalar(f"pde_loss_{i}", pinn.current_epoch, pde_loss[i])
            
        for i in range(boundary_loss.shape[0]):
            self.add_scalar(f"boundary_loss_{i}", pinn.current_epoch, boundary_loss[i])
        # print(f"Epoch {pinn.current_epoch}: pde_loss = {pde_loss}, boundary_loss = {boundary_loss}, loss = {loss}")
        if self.config.adaptive_loss:
            self.add_scalar("w1", pinn.current_epoch, self.ws[0])
            self.add_scalar("w2", pinn.current_epoch, self.ws[1])
            self.add_scalar("log_sigma1", pinn.current_epoch, self.log_sigmas[0])
            self.add_scalar("log_sigma2", pinn.current_epoch, self.log_sigmas[1])
        self.add_scalar("loss", pinn.current_epoch, loss)
        
        return loss
    
    def on_backward_end(self, pinn):
        ...
    
    def on_epoch_begin(self, pinn):
        # ------------log------------
        self.df.loc[pinn.current_epoch, 'epoch'] = pinn.current_epoch
        
        # ------------RAR------------
        if self.config.RAR and (pinn.current_epoch % self.config.resample_freq == 0):
            Xs = []
            if self.config.lower_bound is None or self.config.upper_bound is None:
                raise ValueError("lower_bound and upper_bound must be specified")
            assert len(self.config.lower_bound) == len(self.config.upper_bound)
            
            # uniform sample
            for i in range(len(self.config.lower_bound)):
                Xs.append((torch.rand(self.config.RAR_num, 1) * (self.config.upper_bound[i] - self.config.lower_bound[i]) + \
                    self.config.lower_bound[i]).to(self.config.device).requires_grad_())

            pde_residual = pinn._compute_residual(Xs)
            if type(pde_residual) in [list, tuple]:
                pde_residual = torch.cat(pde_residual, dim=1)
            abs_residual = pde_residual.abs().sum(axis=1)
            # find top k points
            argsort = torch.argsort(abs_residual, descending=True)[:self.config.RAR_k]
            topk_points = torch.cat(Xs, axis=1)[argsort]
            # add top k points to training data
            pinn.add_pde_data(topk_points)
        
        # ------------RAR------------
        elif self.config.RAR_D and (pinn.current_epoch % self.config.resample_freq == 0):
            Xs = []
            if self.config.lower_bound is None or self.config.upper_bound is None:
                raise ValueError("lower_bound and upper_bound must be specified")
            assert len(self.config.lower_bound) == len(self.config.upper_bound)
            
            # uniform sample
            for i in range(len(self.config.lower_bound)):
                Xs.append((torch.rand(self.config.RAR_D_num, 1) * (self.config.upper_bound[i] - self.config.lower_bound[i]) + \
                    self.config.lower_bound[i]).to(self.config.device).requires_grad_())

            pde_residual = pinn._compute_residual(Xs)
            if type(pde_residual) in [list, tuple]:
                pde_residual = torch.cat(pde_residual, dim=1)
            err = pde_residual.abs().sum(axis=1)
            err_eq_normalized = err.pow(self.config.RAR_D_k) / err.pow(self.config.RAR_D_k).mean() + self.config.RAR_D_c
            X_ids = torch.multinomial(err_eq_normalized.view(-1), self.config.RAR_D_sample_num, replacement=False)
            new_X = torch.cat(Xs, axis=1)[X_ids, :]
            pinn.add_pde_data(new_X)

    def on_epoch_end(self, pinn):
        # ------------checkpoints------------
        if not pathlib.Path(self.config.checkpoint_dir).exists():
            raise FileNotFoundError(f"Checkpoint directory {self.config.checkpoint_dir} not found")
        if self.config.save_checkpoints and (pinn.current_epoch % self.config.checkpoint_freq == 0):
                torch.save(pinn.model.state_dict(), f"{self.config.checkpoint_dir}/{self.config.checkpoint_name}_{pinn.current_epoch}.pth")

    def on_val_begin(self, pinn):
        pass
    
    def on_val_end(self, pinn):
        # ------------log------------
        self.add_scalar("val_loss", pinn.current_epoch, sum(pinn.current_val_loss))
        for i, l2_err in enumerate(pinn.current_l2_errs):
            self.add_scalar(f"l2_err_{i}", pinn.current_epoch, l2_err)
        print(f"Epoch {pinn.current_epoch}: val_loss = {pinn.current_val_loss}, train_loss = {pinn.current_train_loss}, l2_err = {pinn.current_l2_errs}")

    def add_scalar(self, name, epoch, value):
        # 检查name是否已经是DataFrame的列
        if name not in self.df.columns:
            self.df[name] = None  # 创建新的列
        # 序列化value
        if isinstance(value, torch.Tensor):
            value = value.item()
        elif isinstance(value, np.ndarray):
            value = value.item()
        elif isinstance(value, list):
            value = ', '.join(value)
            
        # 填入数据
        self.df.loc[epoch, name] = value
    
    def on_exception(self, pinn, e):
        self.df.to_csv(self.config.log_filename, index=False)
        print(f"Exception occurred at epoch {pinn.current_epoch}: {e}")
        raise e