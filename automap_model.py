import torch
import pytorch_lightning as pl
import torch.nn as nn


class AUTOMAPModel(pl.LightningModule):
    def __init__(self, n_H0, n_W0, norm_factor=1.0):
        super(AUTOMAPModel, self).__init__()
        self.n_H0 = n_H0
        self.n_W0 = n_W0
        self.norm_factor = norm_factor
        # Define the model architecture
        self.fc1 = nn.Linear(n_H0 * n_W0 * 2, n_H0 * n_W0)
        self.fc2 = nn.Linear(n_H0 * n_W0, n_H0 * n_W0)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding='same')
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding='same')
        self.conv3 = nn.Conv2d(64, 1, kernel_size=7, padding='same')

        # Activation and Regularization
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        # Reshape x to match the input dimensions
        x = x.view(-1, self.n_H0 * self.n_W0 * 2)

        # Forward pass
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        # Reshape for convolutional layers
        x = x.view(-1, 1, self.n_H0, self.n_W0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x.squeeze(1)  # Remove channel dimension for the output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.mean((y_hat - y) ** 2)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=0.001)
        return optimizer
