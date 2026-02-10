from config.logger_config import setup_logger
import torch
import yaml

from data_sampler import UniformDataSampler, SpaceDataSampler, TimeDataSampler
from network_architectures import DeepONet
from trainer import Pinns_Trainer

logger = setup_logger()

def load_config(path= "config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    DEVICE = torch.device(config['hardware']['device'] if torch.backends.mps.is_available() else "cpu")
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    # Data Domain
    t_max = config['data']['t_max']
    lx, ly = config['data']['lx'], config['data']['ly']

    # Dataset Size
    n_inter, n_ic, n_bc = config['data_size']['n_int'], config['data_size']['n_ic'], config['data_size']['n_bc']


#%% Sampling the Data Points for 1D
    print("--- 1D Processing ---")
    time_bounds = [0.0, t_max]
    spatial_bounds = [0.0,lx]
    domain_1d = [time_bounds, spatial_bounds]
    x = torch.linspace(0, lx, n_ic).view(-1,1)
    ic_coords = torch.cat([torch.zeros_like(x), x], dim=1)

    domain_sampler = UniformDataSampler(domain = domain_1d,num_points = n_inter,device = DEVICE)
    ic_sampler = SpaceDataSampler(spatial_bound= ic_coords, num_points=n_ic, device=DEVICE)
    bc_sampler = TimeDataSampler(spatial_bound = spatial_bounds, temporal_dom=time_bounds,  num_points = n_bc,
                                 device=DEVICE)

    domain_1d = domain_sampler.sample()
    ic_1d = ic_sampler.sample()
    bc_1d = bc_sampler.sample()

    print(f"Interior: {domain_1d.shape}, IC: {ic_1d.shape}, BC: {bc_1d.shape}")

    model_1d = DeepONet(branch_in=100, trunk_in=2, hidden_dim=64, out_dim=1)
    trainer = Pinns_Trainer(model_1d, config, DEVICE)

    for epoch in range(epochs):
        loss = trainer.pinns_trainer()


    print("1D processing is finished ")

 #%% 2D Data Samples
    print("--- 2D Processing ---")
    time_bounds = [0.0, t_max]
    spatial_bounds_2d =  [[0.0, lx],[0.0, ly]]
    domain_2d = [time_bounds] + spatial_bounds_2d
    bc_spatial_2d = [[0.0, 0.0], [lx, 0.0], [0.0, ly], [lx, ly]]
    print(domain_2d)
    grid_size = int(n_ic ** 0.5)

    x_2d = torch.linspace(0, lx, grid_size)
    y_2d = torch.linspace(0, ly, grid_size)
    gx, gy = torch.meshgrid(x_2d, y_2d, indexing="ij")
    ic_coords_2d = torch.stack([torch.zeros_like(gx), gx, gy], dim=-1).view(-1, 3)

    domain_sampler_2d = UniformDataSampler(domain = domain_2d,num_points = n_inter,device = DEVICE)
    ic_sampler_2d = SpaceDataSampler(spatial_bound = spatial_bounds_2d, num_points=n_ic, device=DEVICE)
    bc_sampler_2d = TimeDataSampler(spatial_bound=spatial_bounds_2d, temporal_dom=time_bounds, num_points=n_bc,
                                    device=DEVICE)

    domain_2d = domain_sampler_2d.sample()
    ic_2d = ic_sampler_2d.sample()
    bc_2d = bc_sampler_2d.sample()
    print(f"Interior: {domain_2d.shape}, IC: {ic_2d.shape}, BC: {bc_2d.shape}")
    print("2D processing is finished ")

#%%


if __name__ == "__main__":
    main()