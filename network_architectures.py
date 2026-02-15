import torch
from torch import nn


class FourierEmbeddings(nn.Module):
    def __init__(self,config, in_features):
        super().__init__()
        f_cfg = config.model['fourier_emb']
        self.register_buffer("B", torch.randn(in_features, f_cfg['embed_dim'] // 2) * f_cfg['embed_scale'])

    def forward(self,x):
        proj = torch.matmul(x, self.B)
        embeddings = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return embeddings

class PeriodicEmbeddings(nn.Module):
    def __init__(self,config):
        super().__init__()
        p_cfg = config.model['periodicity']

        self.axis_indices = p_cfg['axis_indices']
        self.period_params = nn.ParameterList([
            nn.Parameter(torch.tensor(float(p)), requires_grad=t)
            for p,t in zip(p_cfg['period'], p_cfg['trainable'])
        ])
    def forward(self,x):
        out = []
        period_idx = 0
        for i in range(x.shape[-1]):
            xi = x[:, i:i+1]
            if i in self.axis_indices:
                p = self.period_params[self.axis_indices.index(i)]
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
    def __init__(self, config):
        super().__init__()
        m_cfg = config.model
        self.activation = {"tanh":nn.Tanh(), "silu":nn.SiLU()}[m_cfg['activation']]

        # Trunk Preprocessing (Embeddings)
        self.trunk_emb = nn.Sequential()
        trunk_dim = m_cfg['trunk_in']
        if 'periodicity' in m_cfg:
            self.trunk_emb.add_module("periodicity", PeriodicEmbeddings(config))
            trunk_dim += len(m_cfg['periodicity']['axis_indices'])

        if 'fourier_emb' in m_cfg:
            self.trunk_emb.add_module("fourier_emb", FourierEmbeddings(config, trunk_dim))
            trunk_dim = m_cfg['fourier_emb']['embed_dim']

        # Branch Netwwork
        branch_layers = [nn.Linear(m_cfg['branch_in'], m_cfg['hidden_dim']), self.activation]
        for _ in range(m_cfg['num_branches'] -1):
            branch_layers.extend([nn.Linear(m_cfg['hidden_dim'], m_cfg['hidden_dim']), self.activation])
        self.branch_net = nn.Sequential(*branch_layers)

        self.encoder_u = nn.Linear(trunk_dim,m_cfg['hidden_dim'])
        self.encoder_v = nn.Linear(trunk_dim,m_cfg['hidden_dim'])
        self.trunk_layers = nn.ModuleList([
            nn.Linear(m_cfg['hidden_dim'], m_cfg['hidden_dim'])
            for _ in range(m_cfg['num_trunk_layers'])
        ])

        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, u_func, x_coord):
        x = self.trunk_emb(x_coord)
        u_gate = self.activation(self.encoder_u(x))
        v_gate = self.activation(self.encoder_v(x))

        h_trunk = u_gate
        for layer in self.trunk_layers:
            z = self.activation(layer(h_trunk))
            h_trunk = z * u_gate + (1-z) * v_gate

        h_branch = self.branch_net(u_func)
        output = torch.sum(h_branch * h_trunk, dim=-1, keepdim=True) + self.b
        return output
