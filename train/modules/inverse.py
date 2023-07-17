import torch
import numpy as np
from ..callback import Callback
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..pinn import PINN


class InverseTrain(Callback):
    def on_train_begin(self, pinn: 'PINN'):
        
        # initialize inverse problem parameters
        inverse_params = []
        if pinn.config.params_init:
            for param in pinn.config.params_init:
                inverse_params.append(torch.tensor(param, requires_grad=True, device=pinn.config.device))
            pinn.optimizer.add_param_group({"params": inverse_params, "lr": pinn.config.learning_rate})
        pinn.inverse_params = inverse_params
    
    def on_epoch_end(self, pinn: 'PINN'):
        if pinn.config.inverse_params:
            for i, param in enumerate(pinn.inverse_params):
                pinn.logger.add_scalar(f"param_{i}", pinn.current_epoch, param.detach().cpu().tolist())
