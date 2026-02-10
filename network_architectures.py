import torch
from torch import nn


class FourierEmbeddings(nn.Module):
    def __init__(self,in_features, embed_dim, embed_scale):
        super().__init__()
        self.register_buffer("B", torch.randn(in_features, embed_dim//2)* embed_scale)

    def forward(self,x):
        proj = torch.matmul(x, self.B)
        embeddings = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return embeddings

class PeriodicEmbeddings(nn.Module):
    def __init__(self,axis_indices, periods):
        super().__init__()
        self.axis_indices = axis_indices
        self.periods = nn.Parameter(torch.tensor(periods, dtype=torch.float32))

    def forward(self,x):
        out = []
        period_idx = 0
        for i in range(x.shape[-1]):
            xi = x[:, i:i+1]
            if i in self.axis_indices:
                p = self.periods[period_idx]
                out.append(torch.cos(p * xi))
                out.append(torch.sin(p * xi))
                period_idx += 1
            else:
                out.append(xi)
        return torch.cat(out, dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, activation = torch.tanh):
        super().__init__()
        self.activation = activation

        # Encoder Architecture
        self.u_gate = nn.Linear(in_dim, hidden_dim)
        self.v_gate = nn.Linear(in_dim, hidden_dim)

        # Hidden Layers
        self.layes = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(num_layers)])

        # Output Layers
        self.out_layers = nn.Linear(hidden_dim, out_dim)

    def forward(self,x):
        # Creating te gates
        U = self.u_gate(x)
        U = self.activation(U)

        V = self.v_gate(x)
        V = self.activation(V)

        # Initial Hidden State
        h = U

        for layer in self.layes:
            z = self.activation(layer(h))
            h = z * U + (1-z) * V
        return self.out_layers(h)


class DeepONet(nn.Module):
    def __init__(self, branch_in, trunk_in, hidden_dim, out_dim):
        super().__init__()
        # Branch Network: this network processess the input function (u)
        self.branch = nn.Sequential(
            nn.Linear(branch_in, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Branch Network: Processes coordinates (x,t)
        self.trunk = nn.Sequential(
            nn.Linear(branch_in, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.b = nn.Parameter(torch.zeros(1))
        self.final_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self,u_func, x_coord):
        # 1. This processes the input using the respective networks.
        B = self.branch(u_func)
        T = self.trunk(x_coord)

        # 2. Pointwise Multiplication
        combined = B * T

        return self.final_linear(combined) + self.b
