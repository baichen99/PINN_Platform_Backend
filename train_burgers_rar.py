from config.common import CommonConfig
from train.pinn import PINN
from models.MLP import MLP


config = CommonConfig(
    epochs = 500000,
    val_freq = 100,
    print_cols = ['epoch', 'loss', 'val_loss', 'pde_loss_0', 'bc_loss_0', 'l2_err_0'],
    batch_size = 1000,
    learning_rate = 1e-3,
    lr_decay = 0.1,
    lr_decay_step = 10000,
    device = "cuda:0",
    net = [2, 30, 30, 30, 1],
    pde = 'burgers',
    
    bc_data_path = "data/burgers/bc_data.csv",
    test_data_path = "data/burgers/test_data.csv",
    pde_data_path = "data/burgers/pde_data.csv",
    X_dim = 2,
    U_dim = 1,
    
    # domain bounds
    lower_bound = [-1.0, 0.0],
    upper_bound = [1.0, 1.0],
    
    # Residual-based adaptive refinement
    RAR = True,
    RAR_freq = 100,
    RAR_num = 1000,
    RAR_top_k = 100,  # choose residual top k points to resample
    
    log_dir = f'logs/burgers_rar',
)


model = MLP(config.net, config.activation_fn).to(config.device)

PINN(config, model).train()

