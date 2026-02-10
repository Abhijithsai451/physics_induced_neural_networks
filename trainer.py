import torch
import torch.nn as nn
from utilities.training_device import get_device
class Pinns_Trainer:
    def __init__(self, model, lr = 1e-3):
        self.model = model
        self.lr = lr
        self.device = get_device()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, u_batch, y_batch, target_batch):
        # Standard Supervised Training Step
        self.model.train()
        self.optimizer.zero_grad()

        # Moving data to the device
        u_batch = u_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        target_batch = target_batch.to(self.device)

        preds = self.model(u_batch, y_batch)
        loss = self.criterion(preds, target_batch)

        loss.backward()
        self.optimizer.step()
        return loss.item()
    def pinns_trainer(self, u_batch, y_batch, pde_fn):
        # This is particularly used for the PINNS
        self.model.train()
        self.optimizer.zero_grad()

        u_batch = u_batch.to(self.device)
        y_batch = y_batch.to(self.device).requires_grad_(True)

        residual = pde_fn(self.model, u_batch, y_batch)
        loss = torch.mean(residual**2)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save_checkpoint(self):
        torch.save(self.model.state_dict(), './checkpoints/model.pth')
