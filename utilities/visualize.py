from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

from utilities.analytical_solutions import get_analytical_solution


def visualize_points_1d(domain_points: torch.Tensor,
                        boundary_points: torch.Tensor,
                        bounds: list,
                        title: str = "Visualization of the Generated Points in 1d"
                        ):
    """
    Visualizes the 1D internal and the boundary points in 1D
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 4))

    if not bounds or len(bounds[0])!= 2:
        raise ValueError("Bounds should be a list of format [[x_min, x_max]].")

    # 1 Plot Domain Points
    x_min , x_max = bounds[0]
    plt.plot([x_min, x_max], [0,0],
            linestyle="--", color="black",linewidth=3, label="Domain Boundary")

    # 2. Plot Internal Points
    if domain_points.numel()> 0:
        x_int = domain_points.cpu().detach().flatten().numpy()
        y_int = torch.zeros_like(domain_points).cpu().detach().flatten().numpy()
        plt.scatter(x_int, y_int, s =40, color='#337AFF', marker='o', alpha= 0.6,
                    zorder=3, label = "Internal Points")

    # 3. Plot Boundary Points
    if boundary_points.numel()>0:
        x_bnd = boundary_points.cpu().detach().flatten().numpy()
        y_bnd = torch.zeros_like(boundary_points).cpu().detach().flatten().numpy()
        plt.scatter(x_bnd, y_bnd, s =100, color='#FF5733', marker='o', edgecolors='black',
                    linewidth=1.5, zorder=5, label = "Boundary Points")

    # Set Plot Properties
    x_range = x_max - x_min
    plt.xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
    plt.ylim(-0.5, 0.5)

    plt.title(title, fontsize = 14)
    plt.xlabel(f"Dimension 1 (x-axis) in {bounds[0]}", fontsize= 12)
    plt.yticks([])
    plt.legend(loc='upper right', bbox_to_anchor=(0.5, 1.15), ncol=3)
    plt.show()
def visualize_solution_1d(model: nn.Module, domain_bound: float,
                          initial_state: torch.Tensor, t_test: float,
                          n_test_points: int = 500, save_path=None):
    """
    Generates test points at a fixed time t_test and plots the NN solution vs. Analytical solution.
    """
    model.eval()  # Set model to evaluation mode
    device = next(model.parameters()).device

    # 1. Generate Test Points
    x_test_np = np.linspace(0, domain_bound, n_test_points, dtype=np.float32)
    x_test = torch.from_numpy(x_test_np).reshape(-1, 1).to(device)
    # Time point (t) is fixed
    t_test_tensor = torch.full_like(x_test, t_test).to(device)

    # Combining the coordinates for the trunk
    coords = torch.cat([t_test_tensor, x_test], dim=1)

    # 2. Calculate Neural Network Solution (u_NN)
    with torch.no_grad():
        u_nn = model(initial_state, coords)
        u_nn_np = u_nn.cpu().numpy().flatten()

    # 3. Calculate Analytical Solution (u_exact)
    u_exact_np = get_analytical_solution(coords.cpu(), alpha=0.01, L = domain_bound, dim =1).flatten().numpy()

    # 4. Calculate Absolute Error
    error_np = np.abs(u_nn_np - u_exact_np)

    # 5. Plotting

    # Create a figure with two subplots: Solution and Error
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # --- Top Subplot: Solution Comparison ---
    ax1.plot(x_test_np, u_exact_np, label='Analytical Solution', color='tab:blue', linewidth=2)
    ax1.plot(x_test_np, u_nn_np, label='PINNS Solution (u_NN)', color='tab:red', linestyle='--', linewidth=2)
    ax1.set_title(f'Solution Comparison at Time $t = {t_test:.2f}$', fontsize=14)
    ax1.set_ylabel('$u(x, t)$', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.7)

    # --- Bottom Subplot: Absolute Error ---
    ax2.plot(x_test_np, error_np, label='Absolute Error $|u_{NN} - u_{exact}|$', color='tab:green', linewidth=1)
    ax2.set_xlabel('Spatial Coordinate $x$', fontsize=12)
    ax2.set_ylabel('Abs. Error', fontsize=12)
    ax2.set_ylim(0, np.max(error_np) * 1.1)  # Auto-scale Y-axis for error
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))  # Use scientific notation for small errors

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def animate_solution_1d(model, domain_bound, t_max, num_frames=100, device='cpu'):
    model.eval()
    fig, ax = plt.subplots(figsize=(10, 6))

    # spatial grid
    x_test_np = np.linspace(0, domain_bound, 500)
    x_test = torch.from_numpy(x_test_np).float().reshape(-1, 1).to(device)

    # Initialize lines
    line_nn, = ax.plot([], [], label='DGM Solution', color='tab:red', lw=2, linestyle='--')
    line_exact, = ax.plot([], [], label='Analytical Solution', color='tab:blue', lw=2, alpha=0.7)

    ax.set_xlim(0, domain_bound)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t)')
    ax.legend()
    title = ax.set_title('')

    def update(frame):
        t_val = (frame / (num_frames - 1)) * t_max
        t_tensor = torch.full_like(x_test, t_val).to(device)

        with torch.no_grad():
            u_nn = model(t_tensor, x_test).cpu().numpy()
            u_exact = get_analytical_solution(t_tensor, x_test).cpu().numpy()

        line_nn.set_data(x_test_np, u_nn)
        line_exact.set_data(x_test_np, u_exact)
        title.set_text(f'Heat Equation 1D - Time: {t_val:.2f}s')
        return line_nn, line_exact, title

    ani = FuncAnimation(fig, update, frames=num_frames, blit=True)
    plt.close()
    return HTML(ani.to_jshtml())


def visualize_loss(trainer, title="Training Loss History", save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(trainer.total_loss, label='Total Loss')
    plt.plot(trainer.loss_pde, label='PDE Loss')
    plt.plot(trainer.loss_ic, label='IC Loss')
    plt.plot(trainer.loss_bc, label='BC Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    if save_path:
        plt.savefig(save_path)
    plt.show()

#%% Visualization functions for the 2D

def visualize_2d(model: nn.Module, bounds: List[List[float]], t_test: float, n_grid: int = 100, save_path=None):
    """
    Plots the PINNS solution as a contour map in the x-y plane.
    """
    model.eval()
    device = next(model.parameters()).device
    t = t_test
    # 1. create a 2d Meshgrid
    x_min  , x_max = bounds[0]
    y_min, y_max =  bounds[1]

    x_np = np.linspace(x_min, x_max, n_grid)
    y_np = np.linspace(y_min, y_max, n_grid)
    X, Y = np.meshgrid(x_np, y_np)

    # Flatten grid and convert to tensors for model input
    x_test = torch.from_numpy(X.flatten()).float().reshape(-1, 1).to(device)
    y_test = torch.from_numpy(Y.flatten()).float().reshape(-1, 1).to(device)
    t_test = torch.full_like(x_test, t_test).to(device)
    spatial_coords = torch.cat([x_test, y_test], 1)

    # 2. Calculate Solution
    with torch.no_grad():
        u_nn = model(t_test, spatial_coords)
        u_nn_np = u_nn.cpu().numpy().reshape(n_grid, n_grid)

        # Analytical Solution
        u_exact = get_analytical_solution(t_test, x_test, y_test)
        u_exact_np = u_exact.cpu().numpy().reshape(n_grid, n_grid)

    # 3. Calculate absolute Error
    error_np = np.abs(u_nn_np - u_exact_np)

    # 4 Plotting
    fig, axes = plt.subplots(1,2,figsize=(14, 6))

    # -- Determine Global Colorbar limits for Comparision
    vmax_solution = max(u_nn_np.max(), u_exact_np.max())
    vmin_solution = min(u_nn_np.min(), u_exact_np.min())
    levels_solution = np.linspace(vmin_solution, vmax_solution, 10)

    ax1 = axes[0]
    contour1 = ax1.contourf(X, Y, u_nn_np, levels=levels_solution, cmap='viridis')
    fig.colorbar(contour1, ax=ax1, label='$u_{NN}(x, y, t)$')
    ax1.set_xlabel('$x$', fontsize=12)
    ax1.set_ylabel('$y$', fontsize=12)
    ax1.set_title(f'DGM Solution at Time $t = {t:.2f}$', fontsize=14)
    ax1.set_aspect('equal', adjustable='box')

    ax2 = axes[1]

    error_vmax = error_np.max()
    contour2 = ax2.contourf(X, Y, error_np, levels=50, cmap='Reds', vmax=error_vmax, extend='max')
    cbar = fig.colorbar(contour2, ax=ax2, format = '%.1e',label='Absolute Error $|u_{NN} - u_{exact}|$')
    ax2.set_xlabel('$x$', fontsize=12)
    ax2.set_ylabel('$y$', fontsize=12)
    ax2.set_title(f'Absolute Error at Time $t = {t:.2f}$', fontsize=14)
    ax2.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_points_2d(domain_points:torch.Tensor,
                        boundary_points:torch.Tensor,
                        bounds: list,
                        title:str = "Visualization of the Generated Points in 2D"):
    """
        Visualizes 2D internal (collocation) and boundary points within a rectangular domain.
    """
    if len(bounds) != 2 or domain_points.shape[1] != 2:
        raise ValueError("This function requires 2D bounds and 2D points (N, 2).")

    sns.set_style("whitegrid")
    plt.figure(figsize=(7, 7))

    # --- Convert to CPU/NumPy/DataFrame for Plotting ---

    # FIX: Use .cpu().detach().numpy() for all tensors
    i_points_np = domain_points.cpu().detach().numpy()
    b_points_np = boundary_points.cpu().detach().numpy()

    df_int = pd.DataFrame(i_points_np, columns=['X', 'Y'])
    df_bnd = pd.DataFrame(b_points_np, columns=['X', 'Y'])

    # --- 1. Draw the Domain Boundary Box ---
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    plt.plot(
        [x_min, x_max, x_max, x_min, x_min],
        [y_min, y_min, y_max, y_max, y_min],
        linestyle='--',
        color='gray',
        linewidth=2.0,
        label='Domain Boundary'
    )
    # --- 2. Plot Internal Points ---
    if not df_int.empty:
        plt.scatter(
            df_int['X'], df_int['Y'],
            s=8,
            color='#337AFF',
            alpha=0.4,
            label='Internal Points'
        )

    # --- 3. Plot Boundary Points (BCs) ---
    if not df_bnd.empty:
        plt.scatter(
            df_bnd['X'], df_bnd['Y'],
            s=25,
            color='#FF5733',
            edgecolors='black',
            linewidths=0.5,
            label='Boundary Points (BCs)'
        )

    # --- Set Plot Properties ---
    x_range = x_max - x_min
    y_range = y_max - y_min
    plt.xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)
    plt.ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

    plt.title(title, fontsize=14)
    plt.xlabel(f'Dimension 1 ({bounds[0]})', fontsize=12)
    plt.ylabel(f'Dimension 2 ({bounds[1]})', fontsize=12)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def visualize_solution_2d(model: nn.Module, bounds: List[List[float]], t_test: float, n_grid: int = 100, save_path=None):
    """
    Plots the DGM solution as a contour map in the x-y plane.
    """
    model.eval()
    device = next(model.parameters()).device
    t = t_test
    # 1. create a 2d Meshgrid
    x_min  , x_max = bounds[0]
    y_min, y_max =  bounds[1]

    x_np = np.linspace(x_min, x_max, n_grid)
    y_np = np.linspace(y_min, y_max, n_grid)
    X, Y = np.meshgrid(x_np, y_np)

    # Flatten grid and convert to tensors for model input
    x_test = torch.from_numpy(X.flatten()).float().reshape(-1, 1).to(device)
    y_test = torch.from_numpy(Y.flatten()).float().reshape(-1, 1).to(device)
    t_test = torch.full_like(x_test, t_test).to(device)
    spatial_coords = torch.cat([x_test, y_test], 1)

    # 2. Calculate Solution
    with torch.no_grad():
        u_nn = model(t_test, spatial_coords)
        u_nn_np = u_nn.cpu().numpy().reshape(n_grid, n_grid)

        # Analytical Solution
        u_exact = get_analytical_solution(t_test, x_test, y_test)
        u_exact_np = u_exact.cpu().numpy().reshape(n_grid, n_grid)

    # Determine limits for Z axis
    v_max = max(u_nn_np.max(), u_exact_np.max()) * 1.05
    v_min = min(u_nn_np.min(), u_exact_np.min()) * 1.05

    # 3. Plotting the graphs
    fig = plt.figure(figsize=(20,8))
    ax2 = plt.subplot2grid((1, 2), (0, 0),  projection='3d')
    ax3 = plt.subplot2grid((1, 2), (0, 1),projection='3d')

    # Plot 2 (bottom left) - Analytical solution
    surf2 = ax2.plot_surface(X, Y, u_exact_np, cmap='plasma',edgecolor='none', alpha=0.9)
    ax2.set_title('Bottom-Left: Analytical Solution (Reference)', fontsize=14)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('u')
    ax2.set_zlim(v_min, v_max)
    ax2.view_init(elev=30, azim=45)

    # Plot 3 (Bottom Right) - DGM Solution
    surf3 = ax3.plot_surface(X, Y, u_nn_np, cmap='viridis', edgecolor='none', alpha=0.9)
    ax3.set_title('Bottom-Right: DGM Solution (Reference)', fontsize=14)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('u')
    ax3.set_zlim(v_min, v_max)
    ax3.view_init(elev=30, azim=45)

    fig.suptitle(f'DGM Solution vs Analytical Solution at Time $t = {t:.2f}$', fontsize=20, y = 0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path)
    plt.show()


def animate_solution_2d(model, bounds, t_max, n_grid=50, num_frames=50, device='cpu'):
    model.eval()

    # Setup Figure and 3D Axes
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    x_np = np.linspace(x_min, x_max, n_grid)
    y_np = np.linspace(y_min, y_max, n_grid)
    X, Y = np.meshgrid(x_np, y_np)

    x_test = torch.from_numpy(X.flatten()).float().reshape(-1, 1).to(device)
    y_test = torch.from_numpy(Y.flatten()).float().reshape(-1, 1).to(device)
    spatial_coords = torch.cat([x_test, y_test], 1)

    # Function to get data for a specific time
    def get_data(t_val):
        t_tensor = torch.full_like(x_test, t_val).to(device)
        with torch.no_grad():
            u_nn = model(t_tensor, spatial_coords).cpu().numpy().reshape(n_grid, n_grid)
            u_exact = get_analytical_solution(t_tensor, x_test, y_test).cpu().numpy().reshape(n_grid, n_grid)
        return u_nn, u_exact

    # Initialize plot with t=0
    u_nn_0, u_exact_0 = get_data(0.0)
    # Determine common Z-axis limits for comparison
    z_min = -1.1
    z_max = 1.1

    def update(frame):
        ax1.clear()
        ax2.clear()

        t_val = (frame / (num_frames - 1)) * t_max
        u_nn, u_exact = get_data(t_val)

        # Plot Analytical Solution (Left)
        surf_exact = ax1.plot_surface(X, Y, u_exact, cmap='plasma', edgecolor='none', alpha=0.9)
        ax1.set_title(f'Analytical Solution (Reference)', fontsize=12)
        ax1.set_zlim(z_min, z_max)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.view_init(elev=30, azim=45)

        # Plot DGM Solution (Right)
        surf_nn = ax2.plot_surface(X, Y, u_nn, cmap='viridis', edgecolor='none', alpha=0.9)
        ax2.set_title(f'DGM Solution', fontsize=12)
        ax2.set_zlim(z_min, z_max)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.view_init(elev=30, azim=45)

        fig.suptitle(f'Heat Equation 2D: Analytical vs DGM at Time t = {t_val:.2f}s', fontsize=16)
        return fig,

    ani = FuncAnimation(fig, update, frames=num_frames, blit=False)
    plt.close()
    return HTML(ani.to_jshtml())