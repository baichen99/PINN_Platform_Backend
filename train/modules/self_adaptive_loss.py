import torch
from train.callback import Callback

class SelfAdaptiveLoss(Callback):
    def on_train_begin(self, pinn):
        self.log_sigmas = [
                torch.tensor(2.0, dtype=torch.float32, requires_grad=True, device=pinn.config.device),
                torch.tensor(2.0, dtype=torch.float32, requires_grad=True, device=pinn.config.device),
            ]
        # append to optimizer
        pinn.optimizer.add_param_group({"params": self.log_sigmas, "lr": pinn.config.learning_rate})
    
    def on_calc_loss(self, pinn, residual: torch.tensor, boundary_loss: torch.tensor):
        pde_loss = residual.pow(2).mean(axis=0)
        pde_weights = [1.0] * pde_loss.shape[0] if not pinn.config.pde_weights else pinn.config.pde_weights
        bc_weights = [1.0] * boundary_loss.shape[0] if not pinn.config.bc_weights else pinn.config.bc_weights
        pde_loss_w = torch.dot(pde_loss, torch.tensor(pde_weights).to(pinn.config.device))
        boundary_loss_w = torch.dot(boundary_loss, torch.tensor(bc_weights).to(pinn.config.device))
        
        # w = 1 / 2sigma.pow(2)
        # self adaptive loss = w1 * pde_loss + w2 * boundary_loss + log(sigma1) + log(sigma2)
        self.ws = [1 / (2 * torch.exp(log_sigma).pow(2)) for log_sigma in self.log_sigmas]
        loss = pde_loss_w * self.ws[0] + boundary_loss_w * self.ws[1] + self.log_sigmas[0] + self.log_sigmas[1]
        
        pinn.logger.add_scalar("w1", pinn.current_epoch, self.ws[0])
        pinn.logger.add_scalar("w2", pinn.current_epoch, self.ws[1])
        pinn.logger.add_scalar("log_sigma1", pinn.current_epoch, self.log_sigmas[0])
        pinn.logger.add_scalar("log_sigma2", pinn.current_epoch, self.log_sigmas[1])
    
        return loss