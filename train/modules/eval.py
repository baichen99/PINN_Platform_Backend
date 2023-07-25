import torch
from utils.metrics import cal_l2_relative_err
from utils.format import print_table, format_to_scientific_notation
from train.callback import Callback



class EvaluateL2Error(Callback):
    def on_epoch_begin(self, pinn):
        if pinn.current_epoch % pinn.config.val_freq == 0:
            with torch.no_grad():
                test_pred = pinn.model(pinn.test_X)[:, :pinn.config.U_dim]
                val_loss = (test_pred - pinn.test_y).pow(2).mean()
                l2_errs = cal_l2_relative_err(test_pred, pinn.test_y)
                pinn.current_l2_errs = l2_errs
                pinn.current_val_loss = val_loss
                
                pinn.logger.add_scalar("val_loss", pinn.current_epoch, val_loss)
                for i, l2_err in enumerate(l2_errs):
                    pinn.logger.add_scalar(f"l2_err_{i}", pinn.current_epoch, l2_err)
                # print(f"Epoch {pinn.current_epoch}: val_loss = {val_loss}, l2_err = {pinn.current_l2_errs}")
    
    def on_epoch_end(self, pinn):
        if pinn.current_epoch % pinn.config.val_freq == 0:
            if pinn.config.print_cols and "*" in pinn.config.print_cols:
                names = pinn.logger.df.columns.tolist()
            else:
                names = pinn.config.print_cols
            # 根据names从df里取出值
            values = [val for val in pinn.logger.df.iloc[pinn.current_epoch][names]]
            # to 2d array
            print_table(cols=['name', 'value'], data=[[name, format_to_scientific_notation(value, 3)] for name, value in zip(names, values)])