import torch

def compute_pde_residual(model, u_branch, coords, alpha = 0.01):
    # Forward Pass

    if coords.shape[1] == 4:
        coords = coords[:,:3]

    u = model(u_branch, coords)

    # First Derivation
    grads = torch.autograd.grad(u, coords,
                                grad_outputs = torch.ones_like(u),
                                create_graph = True,
                                retain_graph = True,
                                )[0]

    u_t = grads[:, 0:1]
    u_x = grads[:, 1:2]


    u_xx = torch.autograd.grad(u_x, coords,
                               grad_outputs = torch.ones_like(u_x),
                               create_graph = True,
                               )[0][:,1:2]
    if coords.shape[1] == 3:
        u_y = grads[:, 2:3]
        u_yy = torch.autograd.grad(u_y, coords,
                                   grad_outputs = torch.ones_like(u_y),
                                   create_graph = True
                                   )[0][:,2:3]
        residual = u_t - alpha * (u_xx + u_yy)
    else:
        residual = u_t - alpha * u_xx


    return residual