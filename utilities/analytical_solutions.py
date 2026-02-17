import torch
import numpy as np

def get_analytical_solution(coords, alpha, L, dim=1):
    """
    Computes the exact Solution for the PDE.
    """
    t = coords[:, 0:1]
    x = coords[:, 1:2]

    if dim == 1:
        # u(t,x) = exp(-a * (pi/L)^2 * t) * sin(pi*x/L)
        decay = torch.exp(-alpha * (np.pi/L)**2 * t)
        spatial = torch.sin(np.pi * x / L)
        return decay * spatial
    elif dim ==2:
        y = coords[:, 2:3]
        decay = torch.exp(-alpha *2 * (np.pi/L)**2 * t)
        spatial = torch.sin(np.pi * x / L) * torch.sin(np.pi * y / L)
        return decay * spatial

def evaluate_model(model, initial_state,config, dim=1 ):
    model.eval()
    L = config.data['lx']
    alpha = config.data.get('alpha',0.01)

    # 1. Create a dense grid for testing.
    t_test = torch.linspace(0, config.data['t_max'], 100)
    x_test = torch.linspace(0, L, 100)

    if dim ==1:
        T,X = torch.meshgrid(t_test, x_test, indexing='ij')
        test_coords = torch.stack([T.flatten(), X.flatten()], dim =1).to(config.device)
    else:
        y_test = torch.linspace(0, config.data['ly'], 100)
        T, X , Y = torch.meshgrid(t_test, x_test, y_test, indexing='ij')
        test_coords = torch.stack([T.flatten(), X.flatten(), Y.flatten()], dim =1).to(config.device)

    with torch.no_grad():
        u_pred = model(initial_state, test_coords)
        u_exact = get_analytical_solution(test_coords, alpha, L, dim=dim)

    error_l2 = torch.norm(u_exact - u_pred)/ torch.norm(u_exact)

    print(f"-- {dim}D Validation Results --")
    print(f"Relative L2 Error: {error_l2.item():.4e}")

    return u_pred, u_exact