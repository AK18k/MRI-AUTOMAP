import torch
import automap_model as automap
import os
from torchvision.transforms import functional as F_torchvision
from PIL import Image
import matplotlib.pyplot as plt



n_H0 = 110
n_W0 = 110



# Specify the path to the checkpoint file
checkpoint_path = "checkpoints/best-checkpoint-v2.ckpt"

# Example: Load a model from the checkpoint
model_trained = automap.AUTOMAPModel.load_from_checkpoint(checkpoint_path, n_H0=n_H0, n_W0=n_W0)

automap_test_dir = '/home/avi/data/imagenet_20/automap_test'
files = os.listdir(automap_test_dir)
file_1 = files[0]
with Image.open(os.path.join(automap_test_dir, file_1)) as img:
    # Convert the image to a PyTorch tensor
    img_1 = F_torchvision.to_tensor(img)


mean = img_1.mean(dim=(1, 2))
max_val = 1
print(f'Max pixel value (norm_factor): {max_val}')
mean_tensor = mean.view(-1, 1, 1)  # Reshape mean to (4000, 1, 1)
# Expand mean_tensor to (4000, 110, 110)
mean_tensor = mean_tensor.expand(-1, n_H0, n_W0)
img_1_normalized = (img_1 - mean_tensor) / max_val

fft_images = torch.fft.fft2(img_1_normalized)
real_part = fft_images.real
imaginary_part = fft_images.imag
sensor_data = torch.stack([real_part, imaginary_part], dim=-1)
sensor_data = sensor_data.to('cuda')

# apply the trained model to the sensor data
model_trained.eval()
model_trained = model_trained.to('cuda')

with torch.no_grad():
    estimated_img = model_trained(sensor_data)

    # Convert the estimated image tensor to a numpy array
    estimated_img_np = estimated_img.cpu().numpy()

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(img_1.permute(1, 2, 0), cmap='gray')
    plt.title('Original Image')

    # Plot the estimated image
    plt.subplot(1, 2, 2)
    plt.imshow(estimated_img_np.squeeze(), cmap='gray')
    plt.title('Estimated Image')

    # Show the plot
    plt.show()








