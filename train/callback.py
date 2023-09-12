import torch

from utils.format import print_table, format_to_scientific_notation


class Callback:
    def on_train_begin(self, pinn):
        pass
    
    def on_train_end(self, pinn):
        pass
    
    def on_backward_end(self, pinn):
        pass
    
    def on_epoch_begin(self, pinn):
        pass
    
    def on_epoch_end(self, pinn):
        pass
    
    def on_val_begin(self, pinn):
        pass
    
    def on_val_end(self, pinn):
        pass
    
    def on_exception(self, pinn, e):
        pass


class Callbacks:
    def __init__(self, modules: list = []):
        self.modules = modules

    def on_train_begin(self, pinn):
        for module in self.modules:
            module.on_train_begin(pinn)

    def on_train_end(self, pinn):
        for module in self.modules:
            module.on_train_end(pinn)

    
    def on_calc_loss(self, pinn, residual, boundary_loss):
        for module in self.modules:
            # if has methods, on cal once
            if hasattr(module, 'on_calc_loss'):
                return module.on_calc_loss(pinn, residual, boundary_loss)
        
        pde_loss = residual.pow(2).mean(axis=0)
        
        # cal loss by default weights
        if not pinn.config.pde_weights:
            pinn.config.pde_weights = [1.0] * pde_loss.shape[1]
        if not pinn.config.bc_weights:
            pinn.config.bc_weights = [1.0] * boundary_loss.shape[1]
        pde_loss_w = pde_loss * torch.tensor(pinn.config.pde_weights).to(pinn.config.device)
        boundary_loss_w = boundary_loss * torch.tensor(pinn.config.bc_weights).to(pinn.config.device)
        return pde_loss_w.sum() + boundary_loss_w.sum()

    def on_backward_end(self, pinn):
        for module in self.modules:
            module.on_backward_end(pinn)

    def on_epoch_begin(self, pinn):
        for module in self.modules:
            module.on_epoch_begin(pinn)

    def on_epoch_end(self, pinn):
        for module in self.modules:
            module.on_epoch_end(pinn)
            
        # 等所有模块都执行完毕后，再打印
        if pinn.current_epoch % pinn.config.val_freq == 0:
            if pinn.config.print_cols and "*" in pinn.config.print_cols:
                names = pinn.logger.df.columns.tolist()
            else:
                names = pinn.config.print_cols
            # 根据names从df里取出值
            values = [val for val in pinn.logger.df.iloc[pinn.current_epoch-1][names]]
            # to 2d array
            print_table(cols=['name', 'value'], data=[[name, format_to_scientific_notation(value, 3)] for name, value in zip(names, values)])

    def on_val_begin(self, pinn):
        for module in self.modules:
            module.on_val_begin(pinn)

    def on_val_end(self, pinn):
        for module in self.modules:
            module.on_val_end(pinn)

    def on_exception(self, pinn, e):
        for module in self.modules:
            module.on_exception(pinn, e)