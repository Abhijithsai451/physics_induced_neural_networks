import torch

from losses import compute_pde_residual


class Pinns_Trainer:
    def __init__(self, model, config):
        self.model = model
        self.device = config.device

        self.alpha = config.data.get('alpha',0.01)
        self.lr = config.training['learning_rate']
        self.optimizer = config.training['optimizer']
        self.criterion = config.training['criterion']

    def train_step(self, u_branch, coords_int, coords_ic, coords_bc, u_ic_target):
        self.model.train()
        self.optimizer.zero_grad()

        coords_int.requires_grad_(True)
        u_pred = self.model(u_branch, coords_int)

        # Calculating the PDE loss
        residual = compute_pde_residual(self.model, u_branch, coords_int, self.alpha)
        loss_pde = self.mean(residual**2)

        # Calculating IC and BC Losses
        pred_ic = self.model(u_branch, coords_ic)
        loss_ic = self.criterion(pred_ic, u_ic_target)
        pred_bc = self.model(u_branch, coords_bc)
        loss_bc = torch.mean(pred_bc**2)

        # Backpropagation
        total_loss = loss_pde + loss_ic + loss_bc
        total_loss.backward()
        self.optimizer.step()
        total_loss.item()
        return {
            "total": total_loss.item(),
            "ic": loss_ic.item(),
            "bc": loss_bc.item(),
            "pde": loss_pde.item()
        }