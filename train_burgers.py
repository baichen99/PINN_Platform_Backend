from config.common import CommonConfig
from train.pinn import PINN
from models.MLP import MLP


config = CommonConfig(
    epochs = 20001,
    val_freq = 1000,
    batch_size = 2000,
    learning_rate = 1e-3,
    lr_decay = 0.1,
    lr_decay_step = 10000,
    device = "cuda",
    net = [2, 30, 30, 30, 1],
    pde = 'burgers',
    
    ic_data_path = "data/burgers/ic_data.csv",
    bc_data_path = "data/burgers/bc_data.csv",
    test_data_path = "data/burgers/test_data.csv",
    pde_data_path = "data/burgers/pde_data.csv",
    X_dim = 2,
    U_dim = 1,
    
    # domain bounds
    lower_bound = [-1.0, 0.0],
    upper_bound = [1.0, 1.0],
    
    # Residual-based adaptive refinement
    RAR = False,
    resample_freq = 1000,
    RAR_k = 1,  # choose residual top k points to resample
    
    # loss weights
    pde_weights = [1.0],
    bc_weights = [1.0],
    
    adaptive_loss = False,
    adaptive_activation = False,
    activation = "tanh",
    optimizer = "Adam",
    
    # Causal configuration
    causal_train = False,
    causal_train_freq = 1000,
    causal_Nt = 10,
    causal_tol = 100.0,
    causal_ic_weight = 10.0,
    
    checkpoint_dir = "",
    checkpoint_freq = 1000,
    checkpoint_name = "model",
    log_dir = f'logs',
)


model = MLP(config.net, config.activation_fn).to(config.device)

PINN(config, model).train()

