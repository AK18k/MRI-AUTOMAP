from pytorch_lightning import seed_everything
from PIL import Image
import cv2
import os
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from torch import nn
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
import os   # for os.path.join
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


# Set the default data type for PyTorch tensors
torch.set_float32_matmul_precision('medium')

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  # Name of the metric to monitor
    dirpath='checkpoints/',  # Directory where checkpoints will be saved
    filename='best-checkpoint',  # Checkpoint file name
    save_top_k=1,  # Number of best models to save; set to -1 to save all
    mode='min',  # Minimize 'val_loss' to determine the best model
    save_last=True,  # Optionally save the last model in addition to the best one
    verbose=True,  # Print a message when a new best model is saved
)




n_H0, n_W0 = 110, 110  # Example dimensions, modify as per your dataset


# Set your seed
seed = 42
seed_everything(seed)



class AUTOMAPModel(pl.LightningModule):
    def __init__(self, n_H0, n_W0):
        super(AUTOMAPModel, self).__init__()
        self.n_H0 = n_H0
        self.n_W0 = n_W0
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


def load_data(X_train, Y_train, batch_size=64):
    # Convert numpy arrays to PyTorch tensors
    tensor_x = torch.Tensor(X_train)  # transform to torch tensor
    tensor_y = torch.Tensor(Y_train)

    # Create a dataset and dataloader
    my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    # create your dataloader
    my_dataloader = DataLoader(my_dataset, batch_size=batch_size, num_workers=10, shuffle=True) 
    return my_dataloader




import os
import torch
from torchvision.transforms import functional as F
from PIL import Image

# Define the directory where the images are located
automap_train_dir = '/home/avi/data/imagenet_20/automap_train'

# Define the number of rotations
num_rotations = 4

# Get the list of files in the automap_train_dir
files = os.listdir(automap_train_dir)

# Create an empty tensor to store the augmented images
augmented_images = torch.empty((len(files) * num_rotations, n_H0, n_W0), dtype=torch.float32)

# Iterate over each file in the automap_train_dir
for i, file in enumerate(files):
    # Open the image file
    with Image.open(os.path.join(automap_train_dir, file)) as img:
        # Convert the image to a PyTorch tensor
        img_tensor = F.to_tensor(img)

        # Rotate the image and store the augmented images
        for j in range(num_rotations):
            rotated_img_tensor = F.rotate(img_tensor, 90 * (j + 1))
            augmented_images[i * num_rotations + j] = rotated_img_tensor

# Print the shape of the augmented images tensor
print(augmented_images.shape)



# Define the directory where the images are located
automap_train_dir = '/home/avi/data/imagenet_20/automap_train'

# Define the number of rotations
num_rotations = 4

# Get the list of files in the automap_train_dir
files = os.listdir(automap_train_dir)

# Create an empty tensor to store the augmented images
augmented_images = torch.empty(
    (len(files) * num_rotations, n_H0, n_W0), dtype=torch.float32)

# Iterate over each file in the automap_train_dir
for i, file in enumerate(files):
    # Open the image file
    with Image.open(os.path.join(automap_train_dir, file)) as img:
        # Convert the image to a PyTorch tensor
        img_tensor = F.to_tensor(img)

        # Rotate the image and store the augmented images
        for j in range(num_rotations):
            rotated_img_tensor = F.rotate(img_tensor, 90 * (j + 1))
            augmented_images[i * num_rotations + j] = rotated_img_tensor

# Print the shape of the augmented images tensor
print(augmented_images.shape)


fft_images = torch.fft.fft2(augmented_images)

real_part = fft_images.real
imaginary_part = fft_images.imag
sensor_data = torch.stack([real_part, imaginary_part], dim=-1)

# Print the shape of the stacked images tensor
print(sensor_data.shape)


def check_device(model):
    return next(model.parameters()).device


# Example usage

model = AUTOMAPModel(n_H0, n_W0)

# device = check_device(model)
# print(device)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# summary(model.to(device), input_size=(2, n_H0, n_W0))


# Assuming X_train and Y_train are available as numpy arrays
X_train = sensor_data
Y_train = augmented_images
train_dataloader = load_data(X_train, Y_train, batch_size=1)

trainer = pl.Trainer(callbacks=[checkpoint_callback], 
                     max_epochs=10)

trainer.fit(model, train_dataloader)
