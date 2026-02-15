import torch
from torch.utils.data import Dataset

class BaseDataSampler:
    def __init__(self,num_points, device):
        self.num_points = num_points
        self.device = device

    def sample(self):
        raise NotImplementedError

class UniformDataSampler(BaseDataSampler):
    """
    > Takes the domain and used the random.uniform to create the data points in the domain.
    > Dense and Continuous
    > Mostly used for Interior data
    """
    def __init__(self,domain: list,
                 num_points: int,
                 device="cuda"):
        super().__init__(num_points, device)
        self.domain = torch.tensor(domain, dtype=torch.float32).to(self.device)
        self.dim = self.domain.shape[0]

    def sample(self):
        low = self.domain[:, 0]
        high = self.domain[:, 1]
        return low + (high - low) * torch.rand(self.num_points,self.dim,device =self.device)

class SpaceDataSampler(BaseDataSampler):
    """
    > Works with the predefined set of points. Instead of picking anywhere in the range, it selects from a list of
      coordinates provided.
    > Used random.choice
    > Discrete and returns the points which exist in the Coordinates
    > Mostly used for sampling from a specific mesh, a complex boundary shape, or experimental data points
    """
    def __init__(self,spatial_bound, num_points, device = "cuda"):
        super().__init__(num_points, device)
        self.spatial_bound = torch.as_tensor(spatial_bound).float().to(device)

    def sample(self):
        idx = torch.randint(0, len(self.spatial_bound), size=(self.num_points,), device = self.device)
        return self.spatial_bound[idx]

class TimeDataSampler(BaseDataSampler):
    """
    > It treats time as a continuous dimension but space as fixed set of points.
    > Picks a random time (t) uniformly between two values and picks random location (x,y,z) from the spatial coords
    > Concatenate both the time and spatial coordinates to create a (t,x,y,z) vector.
    > Time is sampled continuously, space is sampled from fixed grid.
    > Solves time-dependent PDEs to evaluate the physics at particular time.
    """
    def __init__(self,spatial_bound, temporal_dom, num_points, device = "cuda"):
        super().__init__(num_points, device)
        self.spatial_bound = torch.tensor(spatial_bound).float().to(device)
        self.t_min, self.t_max = float(temporal_dom[0]), float(temporal_dom[1])

        if isinstance(spatial_bound, list):
            self.spatial_bound = torch.tensor(spatial_bound, dtype=torch.float32).to(device)
        else:
            self.spatial_bound = spatial_bound.to(device)
    def sample(self):
        idx = torch.randint(0, len(self.spatial_bound), size=(self.num_points,), device = self.device)
        t = self.t_min + (self.t_max - self.t_min) * torch.rand(self.num_points,1,device = self.device)
        x = self.spatial_bound[idx]
        if x.dim() ==1:
            x = x.unsqueeze(1)
        vector = torch.cat((t,x),dim = 1)
        return vector

def get_initial_condition_values(coords):
    """Calculates the scalar values for the initial wound shape."""
    spatial = coords[:, 1:] # Ignore time dimension
    # Gaussian shape: centered at middle, width adjusted
    dist_sq = torch.sum((spatial - 1.0)**2, dim=1, keepdim=True)
    return torch.exp(-5.0 * dist_sq)

