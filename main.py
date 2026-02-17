from config.config_parser import get_args, Config
from config.logger_config import setup_logger
import torch
from data_sampler import UniformDataSampler, SpaceDataSampler, TimeDataSampler, get_initial_condition_values
from network_architectures import DeepONet
from trainer import Pinns_Trainer
from utilities.visualize import visualize_points_1d, visualize_solution_1d, visualize_points_2d, visualize_solution_2d, \
    visualize_loss


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
    epochs_2d = config.training['epochs_2d']
    # Data Domain
    t_max = config.data['t_max']
    lx, ly = config.data['lx'], config.data['ly']
    # Dataset Size
    n_coll, n_ic, n_bc = config.data_size['n_int'], config.data_size['n_ic'], config.data_size['n_bc']


#%% Sampling the Data Points for 1D
    print("--- 1D Processing ---")
    time_bounds = [0.0, t_max]
    spatial_bounds_1d = [0.0, lx]
    domain_1d = [time_bounds, spatial_bounds_1d]

    # Samplers
    collocation_sampler = UniformDataSampler(domain = domain_1d,num_points = n_coll,device = DEVICE)
    ic_sampler = UniformDataSampler(domain=[[0.0, 0.0], spatial_bounds_1d],num_points = n_bc,device = DEVICE)
    bc_sampler = TimeDataSampler(spatial_bound = spatial_bounds_1d, temporal_dom=time_bounds,  num_points = n_bc,
                                 device=DEVICE)

    collocation_1d = collocation_sampler.sample()
    ic_1d = ic_sampler.sample()
    bc_1d = bc_sampler.sample()

    # Branch Inputs:
    n_func_points = config.model['branch_in']
    func_coords = UniformDataSampler([[0.0, 0.0], spatial_bounds_1d], n_func_points, DEVICE).sample()
    initial_state_1d = get_initial_condition_values(func_coords).T  # Shape: [1, n_sensors]

    logger.info(f"Interior: {collocation_1d.shape}, IC: {ic_1d.shape}, BC: {bc_1d.shape}")
    logger.info("Visualizing the Data Samples")
    visualize_points_1d(domain_points=collocation_1d[:, 1:],
        boundary_points=bc_1d[:, 1:],
        bounds=[spatial_bounds_1d],
        title="1D Collocation and Boundary Points"
    )

    model_1d = DeepONet(config).to(DEVICE)
    logger.info(f"Initialized {config.model['arch_name']} with "
          f"{config.model['num_trunk_layers']} trunk layers.")

    trainer_1d = Pinns_Trainer(model_1d, config)
    for epoch in range(epochs):
        target_ic = get_initial_condition_values(ic_1d)
        losses = trainer_1d.train_step(
            initial_state=initial_state_1d,
            collocation_pts=collocation_1d ,
            initial_pts=ic_1d,
            boundary_pts=bc_1d,
            initial_targets=target_ic
        )

        if epoch % 50 == 0:
            print(f"1D Epoch {epoch} | Total Loss: {losses['total']:.6f} | PDE: {losses['pde']:.6f} | IC: {losses['ic']:.6f} | BC: {losses['bc']:.6f}")

    # visualizing the solution in 1d
    visualize_solution_1d(model=model_1d,initial_state=initial_state_1d, domain_bound=lx,t_test=t_max * 0.5)
    # visualizing the loss
    visualize_loss(trainer_1d, title="1D Loss Curve")
    logger.info("1D processing is finished ")

 #%% 2D Data Samples
    print("--- 2D Processing ---")
    time_bounds = [0.0, t_max]
    spatial_bounds_2d =  [[0.0, lx],[0.0, ly]]
    domain_2d = [time_bounds] + spatial_bounds_2d

    collocation_sampler_2d = UniformDataSampler(domain = domain_2d,num_points = n_coll,device = DEVICE)
    ic_sampler_2d = UniformDataSampler(domain=[[0.0, 0.0]] + spatial_bounds_2d, num_points=n_ic, device=DEVICE)
    bc_sampler_2d = TimeDataSampler(spatial_bound=spatial_bounds_2d, temporal_dom=time_bounds, num_points=n_bc,
                                    device=DEVICE)

    collocation_2d = collocation_sampler_2d.sample()
    ic_2d = ic_sampler_2d.sample()
    bc_2d = bc_sampler_2d.sample()

    # 2D Branch Input
    func_coords_2d = UniformDataSampler([[0.0, 0.0]] + spatial_bounds_2d, n_func_points, DEVICE).sample()
    initial_state_2d = get_initial_condition_values(func_coords_2d).T

    # Visualizing the points in 2D
    visualize_points_2d(domain_points=func_coords_2d[:, 1:],
                        boundary_points=bc_2d[:, 1:],
                        bounds=spatial_bounds_2d,
                        title="2D Collocation and Boundary Points")

    model_2d = DeepONet(config).to(DEVICE)
    trainer_2d = Pinns_Trainer(model_2d, config)

    for epoch in range(epochs_2d):
        target_ic_2d = get_initial_condition_values(ic_2d)
        losses = trainer_2d.train_step(
            initial_state=initial_state_2d,
            collocation_pts=collocation_2d,
            initial_pts=ic_2d,
            boundary_pts=bc_2d,
            initial_targets=target_ic_2d
        )
        if epoch % 50 == 0:
            print(
                f"1D Epoch {epoch} | Total Loss: {losses['total']:.6f} | PDE: {losses['pde']:.6f} | IC: {losses['ic']:.6f} | BC: {losses['bc']:.6f}")

    # visualizing the solution in 2d
    visualize_solution_2d(model=model_2d, bounds=spatial_bounds_2d, t_max=t_max, num_frames=100)

    # visualizing the loss
    visualize_loss(trainer_2d, title="2D Loss Curve")

    logger.info("2D processing is finished ")

#%%


if __name__ == "__main__":
    main()