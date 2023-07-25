from config.common import CommonConfig
from train.pinn import PINN
from models.MLP import MLP


config = CommonConfig(
    epochs = 20001,
    val_freq = 10,
    batch_size = 1000,
    learning_rate = 1e-3,
    lr_decay = 0.1,
    lr_decay_step = 10000,
    device = "cuda",
    net = [3] + [50] * 3 + [3],
    pde = 'InverseNS',
    
    bc_data_path = "data/cylinder_wake_ns/bc_data.csv",
    test_data_path = "data/cylinder_wake_ns/bc_data.csv",
    pde_data_path = "data/cylinder_wake_ns/pde_data.csv",
    
    X_dim = 3,
    U_dim = 2,
    
    # domain bounds
    lower_bound = [-5e-2, -5e-2, 0.0, 5e-2],
    upper_bound = [5e-2, 5e-2, 5e-2, 5],
    
    # Residual-based adaptive refinement
    RAR = False,
    resample_freq = 1000,
    RAR_k = 1,  # choose residual top k points to resample
    
    # loss weights
    pde_weights = [1.0],
    bc_weights = [1.0],
    
    adaptive_loss = False,
    adaptive_activation = True,
    activation = "tanh",
    optimizer = "Adam",
    
    params_init = [0.0, 0.0],

    log_dir = f'logs',
)


model = MLP(config.net, config.activation_fn).to(config.device)

PINN(config, model).train()

