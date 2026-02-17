import torch
import torch.optim as optim
import torch.nn as nn
from losses import compute_pde_residual


class Pinns_Trainer:
    def __init__(self, model, config):
        self.model = model
        self.device = config.device

        self.alpha = config.data.get('alpha',0.01)
        self.lr = config.training['learning_rate']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = config.training['learning_rate'])
        self.criterion = nn.MSELoss()
        self.total_loss = []
        self.loss_pde = []
        self.loss_ic = []
        self.loss_bc = []
    def train_step(self, initial_state, collocation_pts, initial_pts, boundary_pts, initial_targets):
        self.model.train()
        self.optimizer.zero_grad()

        collocation_pts.requires_grad_(True)

        # Calculating the PDE loss
        residual = compute_pde_residual(self.model, initial_state, collocation_pts, self.alpha)
        loss_pde = torch.mean(residual**2)

        # Calculating IC and BC Losses
        pred_ic = self.model(initial_state, initial_pts)
        loss_ic = self.criterion(pred_ic, initial_targets)
        pred_bc = self.model(initial_state, boundary_pts)
        loss_bc = torch.mean(pred_bc**2)

        # Backpropagation
        total_loss = loss_pde + loss_ic + loss_bc
        total_loss.backward()
        self.optimizer.step()

        self.total_loss.append(total_loss.item())
        self.loss_pde.append(loss_pde.item())
        self.loss_ic.append(loss_ic.item())
        self.loss_bc.append(loss_bc.item())

        return {
            "total": total_loss.item(),
            "ic": loss_ic.item(),
            "bc": loss_bc.item(),
            "pde": loss_pde.item()
        }