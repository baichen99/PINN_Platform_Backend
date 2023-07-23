import torch
from torch import nn
from torch.utils.data import DataLoader
from config.common import CommonConfig
import pandas as pd
from .callback import Callbacks
from tqdm import tqdm


class Logger:
    def __init__(self):
        self.df = pd.DataFrame(columns=['epoch'])
        self.status = {}

    def add_scalar(self, name, epoch, value):
        if name not in self.df.columns:
            self.df[name] = None
        if type(value) == torch.Tensor:
            value = value.item()
        elif type(value) == list:
            value = value[0]
        else:
            pass
        self.df.loc[epoch, name] = value
    
    def save(self, filename):
        self.df.to_csv(filename, index=False)


class PINN:
    def __init__(self, config: CommonConfig, model: nn.Module):
        self.config = config
        self.model = model
        # if self.config.parallel:
        #     self.model = DataParallel(self.model, device_ids=[0, 1])
        self.device = self.config.device
        self.logger = Logger()
        self.callbacks = Callbacks(self.config.get_modules())

        self.X = None
        self.y = None
        self.pde_points = None
        self.test_X = None
        self.test_y = None
        self.ic_X = None
        self.ic_Y = None
        
        self.current_epoch = 0

        # load data
        if self.config.ic_data_path:
            ic_data = torch.from_numpy(pd.read_csv(self.config.ic_data_path).values).float().to(self.device)
            self.ic_X = ic_data[:, :self.config.X_dim]
            self.ic_y = ic_data[:, self.config.X_dim:self.config.X_dim+self.config.U_dim]
            self.ic_dataloader = DataLoader(list(zip(self.ic_X, self.ic_y)), batch_size=self.config.batch_size)
        
        bc_data = torch.from_numpy(pd.read_csv(self.config.bc_data_path).values).float().to(self.device)
        self.X = bc_data[:, :self.config.X_dim]
        self.y = bc_data[:, self.config.X_dim:self.config.X_dim+self.config.U_dim]
        self.bc_dataloader = DataLoader(list(zip(self.X, self.y)), batch_size=self.config.batch_size)
        
        test_data = torch.from_numpy(pd.read_csv(self.config.test_data_path).values).float().to(self.device)
        self.test_X = test_data[:, :self.config.X_dim]
        self.test_y = test_data[:, self.config.X_dim:self.config.X_dim+self.config.U_dim]
        self.test_dataloader = DataLoader(list(zip(self.test_X, self.test_y)), batch_size=self.config.batch_size)
        
        pde_data = torch.from_numpy(pd.read_csv(self.config.pde_data_path).values).float().to(self.device)
        self.pde_Xs = [pde_data[:, i:i+1].requires_grad_() for i in range(self.config.X_dim)]
        self.pde_dataloader = DataLoader(list(zip(*self.pde_Xs)), batch_size=self.config.batch_size)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.lr_decay_step, gamma=self.config.lr_decay)

    def _compute_residual(self, Xs):
        pde_pred = self.model(torch.cat(Xs, dim=1))
        preds = [pde_pred[:, i:i+1] for i in range(pde_pred.shape[1])]
        if self.config.params_init:
            pde_residual = self.config.pde_fn(*preds, *Xs, *self.inverse_params)
        else:
            pde_residual = self.config.pde_fn(*preds, *Xs)
        # to tensor
        pde_residual = torch.cat(pde_residual, dim=1)
        return pde_residual
    
    def _train(self):
        self.model.train()
        self.callbacks.on_train_begin(self)
        for epoch in tqdm(range(self.config.epochs)):
            self.logger.df.loc[epoch, 'epoch'] = epoch
            self.current_epoch = epoch
            self.callbacks.on_epoch_begin(self)
            
            residual = []
            for i, Xs in enumerate(self.pde_dataloader):
                residual.append(self._compute_residual(Xs))
            # to tensor 
            residual = torch.cat(residual, dim=0)
            
            # residual = self._compute_residual(self.pde_Xs)
            pde_loss = residual.pow(2).mean(axis=0)
            
            # bc_loss = (self.model(self.X)[:, :self.config.U_dim] - self.y).pow(2).mean(axis=0)
            bc_loss = []
            for i, (X, y) in enumerate(self.bc_dataloader):
                bc_loss.append((self.model(X)[:, :self.config.U_dim] - y).pow(2).mean(axis=0, keepdim=True))
            bc_loss = torch.cat(bc_loss, dim=0).mean(axis=0)
            
            self.current_pde_loss = pde_loss.detach().cpu().tolist()
            self.current_bc_loss = bc_loss.detach().cpu().tolist()
            
            loss = self.callbacks.on_calc_loss(self, residual, bc_loss)
            for i in range(pde_loss.shape[0]):
                self.logger.add_scalar(f"pde_loss_{i}", epoch, pde_loss[i])
            for i in range(bc_loss.shape[0]):
                self.logger.add_scalar(f"bc_loss_{i}", epoch, bc_loss[i])
            self.logger.add_scalar("loss", epoch, loss)
            
            self.optimizer.zero_grad()
            # self.callbacks.on_backward_begin(self)
            loss.backward()
            # self.callbacks.on_backward_end(self)
            self.optimizer.step()
            
            self.callbacks.on_epoch_end(self)
            self.logger.save(self.config.log_filename)

        self.callbacks.on_train_end(self)
        
    def train(self):
        try:
            self._train()
        except Exception as e:
            self.logger.save(self.config.log_filename)
            raise e
