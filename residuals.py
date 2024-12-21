import torch
from torch.autograd import grad

def navier_stokes_residual(x, y, z, t, solution_fn, nu=0.1):
    u, v, w, p = solution_fn(x, y, z, t)

    u_t = grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_z = grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_zz = grad(u_z, z, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]

    v_t = grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_x = grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_z = grad(v, z, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_xx = grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    v_zz = grad(v_z, z, grad_outputs=torch.ones_like(v_z), create_graph=True)[0]

    w_t = grad(w, t, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_x = grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_y = grad(w, y, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_z = grad(w, z, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_xx = grad(w_x, x, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
    w_yy = grad(w_y, y, grad_outputs=torch.ones_like(w_y), create_graph=True)[0]
    w_zz = grad(w_z, z, grad_outputs=torch.ones_like(w_z), create_graph=True)[0]

    p_x = grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_z = grad(p, z, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    f_u = u_t + u * u_x + v * u_y + w * u_z + p_x - nu * (u_xx + u_yy + u_zz)
    f_v = v_t + u * v_x + v * v_y + w * v_z + p_y - nu * (v_xx + v_yy + v_zz)
    f_w = w_t + u * w_x + v * w_y + w * w_z + p_z - nu * (w_xx + w_yy + w_zz)

    div = u_x + v_y + w_z
    return f_u, f_v, f_w, div
