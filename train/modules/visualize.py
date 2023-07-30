import torch
from train.callback import Callback
from visualize.visual_history import plot_history
import pathlib

class PlotAfterTrain(Callback):
    def on_train_end(self, pinn):
        cols = pinn.logger.df.columns
        pde_loss_cols = [col for col in cols if col.startswith('pde_loss')]
        bc_loss_cols = [col for col in cols if col.startswith('bc_loss')]
        l2_error_cols = [col for col in cols if col.startswith('l2_err')]
        plot_history(pinn.logger.df, pde_loss_cols, pathlib.Path(pinn.config.log_dir) / 'pde_loss.png')
        plot_history(pinn.logger.df, bc_loss_cols, pathlib.Path(pinn.config.log_dir) / 'bc_loss.png')
        plot_history(pinn.logger.df, l2_error_cols, pathlib.Path(pinn.config.log_dir) / 'l2_error.png')
            