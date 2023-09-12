from .gradients import gradients

def NavierStocks2D(u, v, p, x, y, t, nu=0.01):
    u_t = gradients(u, t)
    u_x = gradients(u, x)
    u_y = gradients(u, y)
    
    v_t = gradients(v, t)
    v_x = gradients(v, x)
    v_y = gradients(v, y)
    
    u_xx = gradients(u_x, x)
    u_yy = gradients(u_y, y)

    v_xx = gradients(v_x, x)
    v_yy = gradients(v_y, y)
    
    x_momentum = u_t + u*u_x + v*u_y - nu * (u_xx + u_yy)
    y_momentum = v_t + u*v_x + v*v_y - nu * (v_xx + v_yy)
    continuity = u_x + v_y
    return [x_momentum, y_momentum, continuity]

def CuAgNS(u, v, p, x, y, t):
    rho = 8920
    mu = 0.0032
    u_t = gradients(u, t)
    u_x = gradients(u, x)
    u_y = gradients(u, y)
    
    v_t = gradients(v, t)
    v_x = gradients(v, x)
    v_y = gradients(v, y)
    
    u_xx = gradients(u_x, x)
    u_yy = gradients(u_y, y)

    v_xx = gradients(v_x, x)
    v_yy = gradients(v_y, y)
    
    p_x = gradients(p, x)
    p_y = gradients(p, y)
    
    x_momentum = rho * (u_t + u*u_x + v*u_y) - mu * (u_xx + u_yy) + p_x
    y_momentum = rho * (v_t + u*v_x + v*v_y) - mu * (v_xx + v_yy) + p_y
    continuity = u_x + v_y
    return [x_momentum, y_momentum, continuity]

def LDJNS(u, v, p, x, y, t):
    rho = 5550
    mu = 7.22e-4
    u_t = gradients(u, t)
    u_x = gradients(u, x)
    u_y = gradients(u, y)
    
    v_t = gradients(v, t)
    v_x = gradients(v, x)
    v_y = gradients(v, y)
    
    u_xx = gradients(u_x, x)
    u_yy = gradients(u_y, y)

    v_xx = gradients(v_x, x)
    v_yy = gradients(v_y, y)
    
    p_x = gradients(p, x)
    p_y = gradients(p, y)
    
    x_momentum = rho * (u_t + u*u_x + v*u_y) - mu * (u_xx + u_yy) + p_x
    y_momentum = rho * (v_t + u*v_x + v*v_y) - mu * (v_xx + v_yy) + p_y
    continuity = u_x + v_y
    return [x_momentum, y_momentum, continuity]


def InverseNS(u, v, p, x, y, t, c1=0.0, c2=0.0):
    u_t = gradients(u, t)
    u_x = gradients(u, x)
    u_y = gradients(u, y)
    
    v_t = gradients(v, t)
    v_x = gradients(v, x)
    v_y = gradients(v, y)
    
    u_xx = gradients(u_x, x)
    u_yy = gradients(u_y, y)

    v_xx = gradients(v_x, x)
    v_yy = gradients(v_y, y)
    
    p_x = gradients(p, x)
    p_y = gradients(p, y)
    
    x_momentum = u_t + c1 * (u*u_x + v*u_y) - c2 * (u_xx + u_yy) + p_x
    y_momentum = v_t + c1 * (u*v_x + v*v_y) - c2 * (v_xx + v_yy) + p_y
    continuity = u_x + v_y
    return [x_momentum, y_momentum, continuity]

def NavierStocks3D(u, v, w, p, x, y, z, t):
    rho = 8935.7
    mu = 0.0032
    nu = mu / rho
    u_t = gradients(u, t)
    u_x = gradients(u, x)
    u_y = gradients(u, y)
    u_z = gradients(u, z)
    
    v_t = gradients(v, t)
    v_x = gradients(v, x)
    v_y = gradients(v, y)
    v_z = gradients(v, z)
    
    w_t = gradients(w, t)
    w_x = gradients(w, x)
    w_y = gradients(w, y)
    w_z = gradients(w, z)
    
    p_x = gradients(p, x)
    p_y = gradients(p, y)
    p_z = gradients(p, z)
    
    u_xx = gradients(u_x, x)
    u_yy = gradients(u_y, y)
    u_zz = gradients(u_z, z)

    v_xx = gradients(v_x, x)
    v_yy = gradients(v_y, y)
    v_zz = gradients(v_z, z)
    
    w_xx = gradients(w_x, x)
    w_yy = gradients(w_y, y)
    w_zz = gradients(w_z, z)
    
    x_momentum = rho * (u_t + u*u_x + v*u_y + w*u_z) - mu * (u_xx + u_yy + u_zz) + p_x
    y_momentum = rho * (v_t + u*v_x + v*v_y + w*v_z) - mu * (v_xx + v_yy + v_zz) + p_y
    z_momentum = rho * (w_t + u*w_x + v*w_y + w*w_z) - mu * (w_xx + w_yy + w_zz) + p_z
    continuity = u_x + v_y + w_z
    
    return [x_momentum, y_momentum, z_momentum, continuity]