from config.common import CommonConfig
from train.pinn import PINN
from models.PFNN import PFNN
from pdes.ns import NavierStocks2D

config = CommonConfig(
    epochs = 20000,
    val_freq = 100,
    # batch_size=1000,
    learning_rate = 1e-3,
    lr_decay = 0.1,
    lr_decay_step = 10000,
    device = "cuda",
    net = [3] + [50] * 3 + [3],
    pde = NavierStocks2D,
    
    bc_data_path = "data/cylinder_wake_ns/bc_data.csv",
    test_data_path = "data/cylinder_wake_ns/test_data.csv",
    pde_data_path = "data/cylinder_wake_ns/pde_data.csv",
    
    X_dim = 3,
    U_dim = 2,
    
    log_dir = f'logs/cylinder_pfnn',
)


model = PFNN(config.net, config.activation_fn).to(config.device)

PINN(config, model).train()

