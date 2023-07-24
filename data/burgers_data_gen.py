import numpy as np
import pandas as pd


# https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/burgers.html
def generate_bc_data(num_points):
    # dirichlet boundary condition: u(-1, t) = u(1, t) = 0
    x_0 = np.ones(num_points) * -1
    x_1 = np.ones(num_points) * 1
    t = np.random.uniform(0, 1, num_points)
    u = np.zeros(num_points)
    # initial condition: u(x, 0) = -sin(pi * x)
    x_ic = np.random.uniform(-1, 1, num_points)
    t_ic = np.zeros(num_points)
    u_ic = -np.sin(np.pi * x_ic)

    df1 = pd.DataFrame({'x': x_0, 't': t, 'u': u})
    df2 = pd.DataFrame({'x': x_1, 't': t, 'u': u})
    df3 = pd.DataFrame({'x': x_ic, 't': t_ic, 'u': u_ic})
    df = pd.concat([df1, df2, df3])
    return df

def generate_pde_data(num_points):
    x = np.random.uniform(-1, 1, num_points)
    t = np.random.uniform(0, 1, num_points)
    data = {'x': x, 't': t}
    df = pd.DataFrame(data)
    return df


def generate_ic_data(num_points):
    # dirichlet boundary condition: u(-1, t) = u(1, t) = 0
    x_0 = np.ones(num_points) * -1
    x_1 = np.ones(num_points) * 1
    t = np.random.uniform(0, 1, num_points)
    u = np.zeros(num_points)
    # initial condition: u(x, 0) = -sin(pi * x)
    x_ic = np.random.uniform(-1, 1, num_points)
    t_ic = np.zeros(num_points)
    u_ic = -np.sin(np.pi * x_ic)

    df = pd.DataFrame({'x': x_ic, 't': t_ic, 'u': u_ic})
    return df

# df = generate_bc_data(1000)
# df.to_csv('data/burgers/bc_data.csv', index=False)

# df = generate_pde_data(5000)
# df.to_csv('data/burgers/pde_data.csv', index=False)

# df = generate_bc_data(20000)
# df.to_csv('data/burgers/test_data.csv', index=False)
df = generate_ic_data(256)
df.to_csv('data/burgers/ic_data.csv', index=False)