from config.config_parser import get_args, Config
from config.logger_config import setup_logger
import torch
from data_sampler import UniformDataSampler, SpaceDataSampler, TimeDataSampler
from network_architectures import DeepONet
from trainer import Pinns_Trainer


def main():
    # Loading the Configuration file passed in Runtime.
    config_path = get_args()
    config = Config(config_path)

    logger = setup_logger(config)
    DEVICE = config.device if torch.backends.mps.is_available() else "cpu"

    logger.info(f"Starting the Project: {config.project_name} ")
    logger.info(f"Configuration Loaded from {config_path} ")

    batch_size = config.training['batch_size']
    epochs = config.training['epochs']
    # Data Domain
    t_max = config.data['t_max']
    lx, ly = config.data['lx'], config.data['ly']

    # Dataset Size
    n_inter, n_ic, n_bc = config.data_size['n_int'], config.data_size['n_ic'], config.data_size['n_bc']


#%% Sampling the Data Points for 1D
    print("--- 1D Processing ---")
    time_bounds = [0.0, t_max]
    spatial_bounds = [0.0, lx]
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

    n_sensors = config.model['branch_in']
    sensor_coords = UniformDataSampler([[0.0, 0.0], spatial_bounds_1d], n_sensors, DEVICE).sample()
    u_branch_1d = get_initial_condition_values(sensor_coords).T  # Shape: [1, n_sensors]

    logger.info(f"Interior: {domain_1d.shape}, IC: {ic_1d.shape}, BC: {bc_1d.shape}")

    model_1d = DeepONet(config).to(DEVICE)

    logger.info(f"Initialized {config.model['arch_name']} with "
          f"{config.model['num_trunk_layers']} trunk layers.")

    trainer = Pinns_Trainer(model_1d, config)
    for epoch in range(epochs):
        interior_1d = domain_sampler_1d.sample()
        ic_1d = ic_sampler_1d.sample()
        bc_1d = bc_sampler_1d.sample()
        u_ic_target_1d = get_initial_condition_values(ic_1d)

        losses = trainer_1d.train_step(
            u_branch=u_branch_1d,
            interior_batch=interior_1d,
            ic_batch=ic_1d,
            bc_batch=bc_1d,
            u_ic_target=u_ic_target_1d
        )

        if epoch % 500 == 0:
            logger.info(f"1D Epoch {epoch} | Total Loss: {losses['total']:.6f} | PDE: {losses['pde']:.6f}")

    logger.info("1D processing is finished ")

 #%% 2D Data Samples
    print("--- 2D Processing ---")
    time_bounds = [0.0, t_max]
    spatial_bounds_2d =  [[0.0, lx],[0.0, ly]]
    domain_2d = [time_bounds] + spatial_bounds_2d
    bc_spatial_2d = [[0.0, 0.0], [lx, 0.0], [0.0, ly], [lx, ly]]
    logger.info(domain_2d)
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
    logger.info(f"Interior: {domain_2d.shape}, IC: {ic_2d.shape}, BC: {bc_2d.shape}")
    logger.info("2D processing is finished ")

#%%


if __name__ == "__main__":
    main()