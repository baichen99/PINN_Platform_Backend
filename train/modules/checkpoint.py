import pathlib
import torch
from train.callback import Callback

class Checkpoint(Callback):
    def on_epoch_end(self, pinn):
        if pinn.config.checkpoint_dir and pinn.config.checkpoint_freq and \
            (pinn.current_epoch % pinn.config.checkpoint_freq == 0):
            filename = f"{pinn.config.checkpoint_dir}/{pinn.config.checkpoint_name}_{pinn.current_epoch}.pth"
            if pathlib.Path(pinn.config.checkpoint_dir).exists():
                torch.save(pinn.model.state_dict(), filename)