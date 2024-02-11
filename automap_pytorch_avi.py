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
from torch.nn import functional as F_torch
from torchvision.transforms import functional as F_torchvision
import os   # for os.path.join
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback
import automap_model as automap



# Set the default data type for PyTorch tensors
torch.set_float32_matmul_precision('medium')

checkpoint_callback = ModelCheckpoint(
    monitor='train_loss',  # Name of the metric to monitor
    dirpath='checkpoints/',  # Directory where checkpoints will be saved
    filename='best-checkpoint',  # Checkpoint file name
    save_top_k=1,  # Number of best models to save; set to -1 to save all
    mode='min',  # Minimize 'val_loss' to determine the best model
    save_last=True,  # Optionally save the last model in addition to the best one
    verbose=True,  # Print a message when a new best model is saved
)


class NewLineEpochCallback(Callback):
    def on_epoch_start(self, trainer, pl_module):
        print("\n")  # Print a new line at the beginning of each epoch




n_H0, n_W0 = 110, 110  # Example dimensions, modify as per your dataset


# Set your seed
seed = 42
seed_everything(seed)



def load_data(X_train, Y_train, batch_size=64):
    # Convert numpy arrays to PyTorch tensors
    tensor_x = torch.Tensor(X_train)  # transform to torch tensor
    tensor_y = torch.Tensor(Y_train)

    # Create a dataset and dataloader
    my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    # create your dataloader
    my_dataloader = DataLoader(my_dataset, batch_size=batch_size, num_workers=10, shuffle=True) 
    return my_dataloader



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
        img_tensor = F_torchvision.to_tensor(img)

        # Rotate the image and store the augmented images
        for j in range(num_rotations):
            rotated_img_tensor = F_torchvision.rotate(img_tensor, 90 * (j + 1))
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
        img_tensor = F_torchvision.to_tensor(img)

        # Rotate the image and store the augmented images
        for j in range(num_rotations):
            rotated_img_tensor = F_torchvision.rotate(img_tensor, 90 * (j + 1))
            augmented_images[i * num_rotations + j] = rotated_img_tensor



# normalize the images, each image is normalized by substracting the mean of each image and dividing by the maximal value of pixel of all images.
mean = augmented_images.mean(dim=(1, 2))
max_val = augmented_images.max()
print(f'Max pixel value (norm_factor): {max_val}')
mean_tensor = mean.view(-1, 1, 1)  # Reshape mean to (4000, 1, 1)
mean_tensor = mean_tensor.expand(-1, n_H0, n_W0)  # Expand mean_tensor to (4000, 110, 110)
augmented_images = (augmented_images - mean_tensor) / max_val


fft_images = torch.fft.fft2(augmented_images)
real_part = fft_images.real
imaginary_part = fft_images.imag
sensor_data = torch.stack([real_part, imaginary_part], dim=-1)


def check_device(model):
    return next(model.parameters()).device


# Example usage

model = automap.AUTOMAPModel(n_H0, n_W0, max_val)

# Assuming X_train and Y_train are available as numpy arrays
X_train = sensor_data
Y_train = augmented_images
train_dataloader = load_data(X_train, Y_train, batch_size=1)

trainer = pl.Trainer(callbacks=[checkpoint_callback, NewLineEpochCallback()],
                     max_epochs=2)

trainer.fit(model, train_dataloader)


