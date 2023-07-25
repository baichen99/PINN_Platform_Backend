import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load training data
def gen_bc_data(num, save_path=''):
    data = loadmat("ns/cylinder_nektar_wake.mat")
    U_star = data["U_star"]  # N x 2 x T
    P_star = data["p_star"]  # N x T
    t_star = data["t"]  # T x 1
    X_star = data["X_star"]  # N x 2
    N = X_star.shape[0]
    T = t_star.shape[0]
    # Rearrange Data
    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T  # N x T
    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T
    PP = P_star  # N x T
    x = XX.flatten()[:, None]  # NT x 1
    y = YY.flatten()[:, None]  # NT x 1
    t = TT.flatten()[:, None]  # NT x 1
    u = UU.flatten()[:, None]  # NT x 1
    v = VV.flatten()[:, None]  # NT x 1
    p = PP.flatten()[:, None]  # NT x 1
    # training domain: X × Y = [1, 8] × [−2, 2] and T = [0, 7]
    data1 = np.concatenate([x, y, t, u, v, p], 1)
    data2 = data1[:, :][data1[:, 2] <= 7]
    data3 = data2[:, :][data2[:, 0] >= 1]
    data4 = data3[:, :][data3[:, 0] <= 8]
    data5 = data4[:, :][data4[:, 1] >= -2]
    data_domain = data5[:, :][data5[:, 1] <= 2]
    # choose number of training points: num =7000
    idx = np.random.choice(data_domain.shape[0], num, replace=False)
    x_train = data_domain[idx, 0:1]
    y_train = data_domain[idx, 1:2]
    t_train = data_domain[idx, 2:3]
    u_train = data_domain[idx, 3:4]
    v_train = data_domain[idx, 4:5]
    p_train = data_domain[idx, 5:6]
    data = np.concatenate([x_train, y_train, t_train, u_train, v_train, p_train], 1)
    np.savetxt(save_path, data, delimiter=',', header='x,y,t,u,v,p', comments='')

def gen_pde_data(num, save_path=''):
    x = np.random.uniform(1, 8, num)
    y = np.random.uniform(-2, 2, num)
    t = np.random.uniform(0, 7, num)
    data = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1), t.reshape(-1, 1)], axis=1)
    np.savetxt(save_path, data, delimiter=',', header='x,y,t', comments='')

if __name__ == '__main__':
    gen_bc_data(7000, 'ns/bc_data.csv')
    gen_bc_data(20000, 'ns/test_data.csv')
    gen_pde_data(10000, 'ns/pde_data.csv')
