from data_sampler import UniformDataSampler, SpaceDataSampler, TimeDataSampler
from config.logger_config import setup_logger
import torch
import yaml

logger = setup_logger()

def load_config(path= "config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    DEVICE = torch.device(config['hardware']['device'] if torch.backends.mps.is_available() else "cpu")
    batch_size = config['training']['batch_size']

    # Data Domain
    t_max = config['data']['t_max']
    lx = config['data']['lx']
    ly = config['data']['ly']

    # Dataset Size
    n_inter = config['data_sizes']['n_int']
    n_ic = config['data_sizes']['n_ic']
    n_bc = config['data_sizes']['n_bc']


    # Sampling the Data Points for 1D
    domain_1d = [[0.0, t_max],[0.0, lx]]
    bc_spatial = [[0.0],[lx]]

    x_ic = torch.linspace(0, lx, n_ic).view(-1,1)
    ic_coords = torch.cat([torch.zeros_like])

"""
if cfg['physics']['type'] == "1d":
        # Dom: [[t_min, t_max], [x_min, x_max]]
        full_dom = [[0.0, t_max], [0.0, lx]]
        # BC Spatial: Only two points (x=0 and x=lx)
        bc_spatial = [[0.0], [lx]]
        # IC Spatial: Points along x-axis at t=0
        x_ic = torch.linspace(0, lx, cfg['data_sizes']['n_ic']).view(-1, 1)
        ic_coords = torch.cat([torch.zeros_like(x_ic), x_ic], dim=1)

    else: # 2D Processing
        ly = cfg['physics']['ly']
        # Dom: [[t_min, t_max], [x_min, x_max], [y_min, y_max]]
        full_dom = [[0.0, t_max], [0.0, lx], [0.0, ly]]
        # BC Spatial: Corners/Edges of the square [x, y]
        bc_spatial = [[0.0, 0.0], [lx, ly], [0.0, ly], [lx, 0.0]]
        # IC Spatial: Grid on x-y plane at t=0
        x = torch.linspace(0, lx, 25) # 25 * 24 = 600 points
        y = torch.linspace(0, ly, 24)
        gx, gy = torch.meshgrid(x, y, indexing='ij')
        ic_coords = torch.stack([torch.zeros_like(gx), gx, gy], dim=-1).view(-1, 3)

    # --- 2. Initialize Samplers ---
    # Interior (N_INT)
    sampler_int = UniformDataSampler(
        dom=full_dom, 
        batch_size=cfg['data_sizes']['n_int'], 
        device=device
    )
    
    # Initial Condition (N_IC)
    sampler_ic = SpaceDataSampler(
        coords=ic_coords, 
        batch_size=cfg['data_sizes']['n_ic'], 
        device=device
    )
    
    # Boundary Condition (N_BC)
    sampler_bc = TimeDataSampler(
        spatial_coords=bc_spatial,
        temporal_dom=[0.0, t_max],
        batch_size=cfg['data_sizes']['n_bc'],
        device=device
    )

    # --- 3. Execution (Example for one training step) ---
    X_INT = sampler_int.sample().requires_grad_(True)
    X_IC  = sampler_ic.sample()
    X_BC  = sampler_bc.sample()

    print(f"Mode: {cfg['physics']['type'].upper()}")
    print(f"N_INT batch: {X_INT.shape}") # (N_INT, 2) for 1D, (N_INT, 3) for 2D
    print(f"N_IC batch:  {X_IC.shape}")
    print(f"N_BC batch:  {X_BC.shape}")
"""



if __name__ == "__main__":
    main()