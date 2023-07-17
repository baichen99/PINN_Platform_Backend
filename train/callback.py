import torch

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
        
        # cal loss by default weights
        pde_loss = residual.pow(2).mean(axis=0)
        return pde_loss.sum() + boundary_loss.sum()

    def on_backward_end(self, pinn):
        for module in self.modules:
            module.on_backward_end(pinn)

    def on_epoch_begin(self, pinn):
        for module in self.modules:
            module.on_epoch_begin(pinn)

    def on_epoch_end(self, pinn):
        for module in self.modules:
            module.on_epoch_end(pinn)

    def on_val_begin(self, pinn):
        for module in self.modules:
            module.on_val_begin(pinn)

    def on_val_end(self, pinn):
        for module in self.modules:
            module.on_val_end(pinn)

    def on_exception(self, pinn, e):
        for module in self.modules:
            module.on_exception(pinn, e)