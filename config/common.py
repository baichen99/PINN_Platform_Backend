from pydantic import BaseModel
import torch
from pdes.burgers import Burgers
from pdes.ns import *
from models.activation import AdaTanh, AdaSigmoid
from train.modules import *

class CommonConfig(BaseModel):
    # Training configuration
    epochs: int = 10000
    val_freq: int = 1000
    print_cols: list[str] = ["*"]
    batch_size: int = 1000
    learning_rate: float = 1e-3
    lr_decay: float = 0.99
    lr_decay_step: int = 10000
    use_lbfgs: bool = False
    lbfgs_max_iter: int = 15000
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    parallel: bool = False
    
    # visualize
    plot_after_train: bool = True
    
    # Network configuration
    net: list[int] = [3, 20, 1]
    activation: str = "tanh"
    adaptive_activation: bool = False
    optimizer: str = "Adam"
    
    # Inverse problem configuration
    params_init: list[float] = None

    # PDE configuration
    pde: str = 'burgers'
    X_dim: int = 2
    U_dim: int = 1
    
    # Data paths
    ic_data_path: str = None
    bc_data_path: str = "bc_data.csv"
    test_data_path: str = "test_data.csv"
    pde_data_path: str = "pde_data.csv"
    
    # Domain bounds
    lower_bound: list[float] = None
    upper_bound: list[float] = None
    
    # Resampling configuration
    resample_freq: int = 1000
    
    # Residual-based adaptive refinement (RAR)
    RAR: bool = False
    RAR_num: int = 10000
    RAR_freq: int = 1000
    RAR_top_k: int = 1
    
    # RAR-D configuration
    RAR_D: bool = False
    RAR_D_num: int = 1000000
    RAR_D_sample_num: int = 5000
    RAR_D_k: float = 2.0
    RAR_D_c: float = 0.0
    
    # Causal configuration
    causal_train: bool = False
    causal_train_freq: int = 1000
    causal_Nt: int = 10
    causal_L0_weight: float = 100.0
    causal_tol: float = 100.0
    
    # Loss weights
    pde_weights: list[float] = None
    bc_weights: list[float] = None
    
    # Adaptive loss configuration
    adaptive_loss: bool = False
    
    # Checkpoint configuration
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    checkpoint_freq: int = 1000
    checkpoint_name: str = "model"
    
    log_dir: str = "logs"
    
    def get_modules(cls) -> list:
        modules = []
        if cls.val_freq:
            modules.append(EvaluateL2Error())
        if cls.adaptive_loss:
            modules.append(SelfAdaptiveLoss())
        if cls.causal_train:
            modules.append(CausalTrain())
        if cls.save_checkpoints:
            modules.append(Checkpoint())
        if cls.params_init:
            modules.append(InverseTrain())
        if cls.plot_after_train:
            modules.append(PlotAfterTrain())
        if cls.RAR:
            modules.append(RAR())
        return modules
    
    @property
    def log_filename(cls):
        return f"{cls.log_dir}/log.csv"
    
    
    @property
    def pde_fn(cls):
        """Calculate the residual of the PDE"""
        if cls.pde == 'burgers':
            return Burgers
        elif cls.pde == 'ns2d':
            return NavierStocks2D
        elif cls.pde == 'ns3d':
            return NavierStocks3D
        elif cls.pde == 'CuAgNS':
            return CuAgNS
        elif cls.pde == 'InverseNS':
            return InverseNS
        elif cls.pde == 'LDJNS':
            return LDJNS
        else:
            raise ValueError("Invalid PDE")
        
    @property
    def activation_fn(cls):
        """Get the activation function"""
        if cls.adaptive_activation:
            if cls.activation == 'tanh':
                return AdaTanh()
            elif cls.activation == 'sigmoid':
                return AdaSigmoid()
            else:
                raise ValueError("Invalid activation function")
        else:
            if cls.activation == 'tanh':
                return torch.nn.Tanh()
            elif cls.activation == 'relu':
                return torch.nn.ReLU()
            elif cls.activation == 'sigmoid':
                return torch.nn.Sigmoid()
            elif cls.activation == 'leaky_relu':
                return torch.nn.LeakyReLU()
            elif cls.activation == 'elu':
                return torch.nn.ELU()
            else:
                raise ValueError("Invalid activation function")
