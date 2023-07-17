import sys
import signal
import torch
from torch import nn
from .callbacks import Callbacks
from config.common import CommonConfig
import pandas as pd
from utils.metrics import cal_l2_relative_err

class PINN:
    def __init__(self, config: CommonConfig, model: nn.Module, device: torch.device):
        self.config = config
        self.model = model
        
        self.device = device
        self.optimizer = None
        self.callbacks = Callbacks(config)
        
        self.X = None
        self.y = None
        self.pde_points = None
        self.test_X = None
        self.test_y = None
        
        # load data
        bc_data = torch.from_numpy(pd.read_csv(self.config.bc_data_path).values).float().to(self.device)
        self.X = bc_data[:, :self.config.X_dim]
        self.y = bc_data[:, self.config.X_dim:self.config.X_dim+self.config.U_dim]
        
        test_data = torch.from_numpy(pd.read_csv(self.config.test_data_path).values).float().to(self.device)
        self.test_X = test_data[:, :self.config.X_dim]
        self.test_y = test_data[:, self.config.X_dim:self.config.X_dim+self.config.U_dim]
        
        pde_data = torch.from_numpy(pd.read_csv(self.config.pde_data_path).values).float().to(self.device)
        self.pde_Xs = [pde_data[:, i:i+1].requires_grad_() for i in range(self.config.X_dim)]
        
        # optimizer
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.config.adaptive_loss:
            parameters += self.callbacks.log_sigmas
        if self.config.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(parameters, lr=self.config.learning_rate)
        elif self.config.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(parameters, lr=self.config.learning_rate)
        else:
            raise ValueError("Invalid optimizer")
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.lr_decay_step, 
                                                    gamma=self.config.lr_decay)
        
        # for callback variables
        self.current_epoch = 0
        self.current_l2_errs = []
        self.current_train_loss = []
        self.current_val_loss = []
        
    def register_callbacks(self, callbacks: Callbacks):
        self.callbacks = callbacks
    
    def add_data(self, X, y):
        if self.X is None:
            self.X = X
            self.y = y
        else:
            self.X = torch.cat((self.X, X), dim=0)
            self.y = torch.cat((self.y, y), dim=0)
        
    def add_pde_data(self, Xs: torch.tensor):
        assert Xs.shape[1] == len(self.pde_Xs)
        for i in range(Xs.shape[1]):
            self.pde_Xs[i] = torch.cat([self.pde_Xs[i], Xs[:, i].reshape(-1, 1)], axis=0).requires_grad_()
        print(f'{Xs.shape[0]} points has been added to data')

    def set_pde_data(self, Xs: torch.tensor):
        assert Xs.shape[1] == len(self.pde_Xs)
        for i in range(Xs.shape[1]):
            self.pde_Xs[i] = Xs[:, i].reshape(-1, 1).requires_grad_()
        print(f'{Xs.shape[0]} points has been set to data')
    
    def _compute_residual(self, Xs: torch.tensor):
        pde_pred = self.model(torch.cat(Xs, dim=1))
        preds = [pde_pred[:, i:i+1] for i in range(pde_pred.shape[1])]
        pde_residual = self.config.pde_fn(*preds, *Xs)
        if type(pde_residual) in [list, tuple]:
            pde_residual = torch.cat(pde_residual, dim=1)
        return pde_residual
    
    def _train(self):
        self.model.train()
        self.callbacks.on_train_begin(self)
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            self.callbacks.on_epoch_begin(self)
            
            # cal pde loss
            pde_residual = self._compute_residual(self.pde_Xs)
            pde_loss = pde_residual.pow(2).mean(axis=0)
            # cal boundary loss
            boundary_pred = self.model(self.X)
            # mse
            boundary_loss = (boundary_pred - self.y).pow(2).mean(axis=0)
            self.current_train_loss = pde_loss.flatten().tolist() + boundary_loss.flatten().tolist()
            
            loss = self.callbacks.on_calc_loss(self, pde_loss, boundary_loss)
            # loss = pde_loss + boundary_loss
            
            self.optimizer.zero_grad()
            backward_second_time = self.config.RAR or self.config.RAR_D
            loss.backward(retain_graph=backward_second_time)
            self.optimizer.step()
            self.scheduler.step()
            self.callbacks.on_epoch_end(self)
                        
            if epoch % self.config.val_freq == 0:
                self.callbacks.on_val_begin(self)
                with torch.no_grad():
                    y_pred = self.model(self.test_X)
                    val_loss = (y_pred - self.test_y).pow(2).mean(axis=0).flatten().tolist()
                    self.current_val_loss = val_loss
                    # cal l2 err
                    self.current_l2_errs = cal_l2_relative_err(y_pred, self.test_y)
                self.callbacks.on_val_end(self)
        
        if self.config.use_lbfgs:
            optim = torch.optim.LBFGS(self.model.parameters(),
                                      lr=1,
                                      max_iter=self.config.lbfgs_max_iter,
                                      tolerance_grad=1e-8,
                                      tolerance_change=0,
                                      history_size=100,
                                      line_search_fn=None,
                                    )
            def closure():
                optim.zero_grad()
                pde_pred = self.model(torch.cat(self.pde_Xs, dim=1))
                preds = [pde_pred[:, i:i+1] for i in range(pde_pred.shape[1])]
                pde_residual = self.config.pde_fn(*preds, *self.pde_Xs)
                pde_loss = pde_residual.pow(2).mean(axis=0)
                pde_loss = pde_loss * torch.tensor(self.config.pde_weights).to(self.device)
                
                boundary_pred = self.model(self.X)
                boundary_loss = (boundary_pred - self.y).pow(2).mean()
                boundary_loss = boundary_loss * torch.tensor(self.config.bc_weights).to(self.device)
                loss = pde_loss + boundary_loss
                loss.backward()
                return loss
            optim.step(closure)
            # eval
            self.callbacks.on_val_begin(self)
            with torch.no_grad():
                y_pred = self.model(self.test_X)
                val_loss = (y_pred - self.test_y).pow(2).mean()
                self.current_val_loss = val_loss.item()
                # cal l2 err
                self.current_l2_errs = cal_l2_relative_err(y_pred, self.test_y)
            self.callbacks.on_val_end(self)
        self.callbacks.on_train_end(self)
    
    def get_train_info(self):
        info = {
            'current_epoch': self.current_epoch,
            'total_epochs': self.config.epochs,
            'current_train_loss': self.current_train_loss,
            'current_val_loss': self.current_val_loss,
            'current_l2_errs': self.current_l2_errs,
        }
        return info
    
    def train(self):
        def signal_handler(signal, frame):
            self.callbacks.on_exception(self, KeyboardInterrupt())
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)
        try:
            self._train()
        except Exception as e:
            self.callbacks.on_exception(self, e)
        # catch exit