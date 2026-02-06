import torch
from torch.utils.data import Dataset

class BaseDataSampler:
    def __init__(self,batch_size, device):
        self.batch_size = batch_size
        self.device = device

    def sample(self):
        raise NotImplementedError

class UniformDataSampler(BaseDataSampler):
    """
    > Takes the domain and used the random.uniform to create the data points in the domain.
    > Dense and Continuous
    > Mostly used for Interior data
    """
    def __init__(self,dom, batch_size, device="cuda"):
        super().__init__(batch_size, device)
        self.dom = torch.tensor(dom).float().to(self.device)

    def sample(self):
        low = self.dom[:, 0]
        high = self.dom[:, 1]
        return low + (high - low) * torch.rand(self.batch_size,len(low),device =self.device)

class SpaceDataSampler(BaseDataSampler):
    """
    > Works with the predefined set of points. Instead of picking anywhere in the range, it selects from a list of
      coordinates provided.
    > Used random.choice
    > Discrete and returns the points which exist in the Coordinates
    > Mostly used for sampling from a specific mesh, a complex boundary shape, or experimental data points
    """
    def __init__(self,coords, batch_size, device = "cuda"):
        super().__init__(batch_size, device)
        self.coords = torch.tensor(coords).float().to(device)

    def sample(self):
        idx = torch.randint(0, len(self.coords), size=(self.batch_size,), device = self.device)
        return self.coords[idx]

class TimeDataSampler(BaseDataSampler):
    """
    > It treats time as a continuous dimension but space as fixed set of points.
    > Picks a random time (t) uniformly between two values and picks random location (x,y,z) from the spatial coords
    > Concatenate both the time and spatial coordinates to create a (t,x,y,z) vector.
    > Time is sampled continuously, space is sampled from fixed grid.
    > Solves time-dependent PDEs to evaluate the physics at particular time.
    """
    def __init__(self,spatial_coords, temporal_dom, batch_size, device = "cuda"):
        super().__init__(batch_size, device)
        self.spatial_coords = torch.tensor(spatial_coords).float().to(device)
        self.t_min, self.t_max = temporal_dom

    def sample(self):
        idx = torch.randint(0, len(self.spatial_coords), size=(self.batch_size,), device = self.device)
        t = self.t_min + (self.t_max - self.t_min) * torch.rand(self.batch_size,1,device = self.device)
        x = self.spatial_coords[idx]
        vector = torch.cat((t,x),dim = 1)
        return vector



def sample_data()