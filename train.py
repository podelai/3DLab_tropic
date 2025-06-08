import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from Unet import UNet2

import numpy as np
import skimage.transform


def random_crop_stack(image_stack, crop_size):
    """Randomly crops a subvolume from an image stack."""
    if not isinstance(image_stack, np.ndarray):
        raise TypeError("image_stack must be a numpy.ndarray")

    for i in range(3):
        if crop_size[i] > image_stack.shape[i]:
            raise ValueError(f"crop_size[{i}] ({crop_size[i]}) cannot be larger than image_stack.shape[{i}] ({image_stack.shape[i]})")

    # Calculate maximum starting indices for cropping
    max_h_start = image_stack.shape[0] - crop_size[0]
    max_w_start = image_stack.shape[1] - crop_size[1]
    max_d_start = image_stack.shape[2] - crop_size[2]

    # Generate random starting indices
    h_start = np.random.randint(0, max_h_start + 1)
    w_start = np.random.randint(0, max_w_start + 1)
    d_start = np.random.randint(0, max_d_start + 1)

    # Perform the crop
    cropped_stack = image_stack[h_start:h_start + crop_size[0],
                                w_start:w_start + crop_size[1],
                                d_start:d_start + crop_size[2]]
    return cropped_stack


def downscale_upscale_height(img, resolution_ratio):
    """Downscales/upscales along height (axis 0)."""
    original_shape = img.shape
    # Calculate target height for downscaling
    target_height_downscaled = int(original_shape[0] / resolution_ratio)

    # Ensure the target height is at least 1
    target_height_downscaled = max(1, target_height_downscaled)

    img_float = img.astype(float)
    downscale_shape = (target_height_downscaled, original_shape[1], original_shape[2])
    # Downscale using nearest-neighbor interpolation (order=0) to maintain binary nature
    img_downscaled = skimage.transform.resize(img_float, downscale_shape, order=0, preserve_range=True)

    # Convert back to binary after downscaling
    img_downscaled_binary = (img_downscaled > 0.5)

    # Upscale back to original height using nearest-neighbor interpolation
    upscale_shape = original_shape
    img_upscaled = skimage.transform.resize(img_downscaled_binary.astype(float), upscale_shape, order=0, preserve_range=True)

    return img_upscaled.astype(bool)


def upscale_depth(img, resolution_ratio):
    """Upscales along height (axis 0)."""
    original_shape = img.shape
    # Calculate target height for upscaling
    target_depth_upscaled = int(original_shape[2] * resolution_ratio)

    upscale_shape = (original_shape[0], original_shape[1], target_depth_upscaled)
    # Upscale using nearest-neighbor interpolation (order=0)
    img_upscaled = skimage.transform.resize(img.astype(float), upscale_shape, order=1, preserve_range=True)

    return img_upscaled.astype(bool)


def create_cropped_image_dataset(image_stack, train_size, crop_size, resolution_ratio, random_state):
    """Creates a dataset of cropped image stacks for training."""
    if random_state is not None:
        if isinstance(random_state, int):
            np.random.seed(random_state)
        elif isinstance(random_state, np.random.RandomState):
            np.random.set_state(random_state.get_state())
        else:
            raise TypeError("random_state must be an int or a numpy.random.RandomState")

    crop_height, crop_width, crop_depth = crop_size

    # Initialize arrays for the dataset
    # Label dataset will be 2D slices (H, W)
    label_dataset = np.zeros((train_size, crop_height, crop_width), dtype=image_stack.dtype)
    # Downscaled dataset will be 3D volumes (H, W, D)
    downscaled_dataset = np.zeros((train_size, crop_height, crop_width, crop_depth), dtype=image_stack.dtype)

    for i in range(train_size):
        # Get a random crop
        cropped_image = random_crop_stack(image_stack, crop_size)
        # The label is the middle slice of the original resolution crop
        label_dataset[i] = cropped_image[:, :, crop_depth//2]

        # Create the downscaled/upscaled input volume
        downscaled_dataset[i] = downscale_upscale_height(cropped_image, resolution_ratio)

    # Add channel dimension for labels (1 channel)
    label_dataset = np.expand_dims(label_dataset, axis=1)
    # Transpose downscaled dataset to (N, D, H, W) for PyTorch input
    downscaled_dataset = np.transpose(downscaled_dataset, axes=[0, 3, 1, 2])

    return label_dataset, downscaled_dataset


class CustomDataset(Dataset):
    """PyTorch Dataset for image and label numpy arrays."""
    def __init__(self, images_np, labels_np):
        # Convert numpy arrays to PyTorch tensors
        self.images = torch.from_numpy(images_np).float()
        self.labels = torch.from_numpy(labels_np).float()
        # Ensure the number of samples matches
        assert self.images.shape[0] == self.labels.shape[0], "Number of images and labels must match."
        self.num_samples = self.images.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return image and label tensor for a given index
        return self.images[idx], self.labels[idx]


def setup_model(in_channels, out_channels, learning_rate, device):
    """Sets up the UNetSR model, loss function, and optimizer."""
    model = UNet2(in_channels, out_channels, dropout_rate=0.2).to(device)
    # Using Binary Cross-Entropy Loss for binary segmentation
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    """Trains the UNet model."""
    print("Starting training...")
    model.train() # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Move data to the specified device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Print statistics every few steps
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        # Print average loss for the epoch
        print(f'Epoch [{epoch+1}/{num_epochs}] finished. Average Loss: {running_loss / len(train_loader):.4f}')
    print("Training finished.")


def train_execution_process(RAW_STACK, TRAIN_XY_SIZE, TRAIN_Z_SIZE, TRAIN_NB, RESOLUTION_FACTOR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, RANDOM_STATE):
    
    
    TRAIN_CROP_SIZE = (TRAIN_XY_SIZE, TRAIN_XY_SIZE, TRAIN_Z_SIZE)
    
    
    if RAW_STACK is None:
        print("Failed to load image stack. Exiting.")
        #exit()
    
    label_dataset, downscaled_dataset = create_cropped_image_dataset(
        RAW_STACK, TRAIN_NB, TRAIN_CROP_SIZE, RESOLUTION_FACTOR, random_state=RANDOM_STATE
    )


    train_dataset = CustomDataset(downscaled_dataset, label_dataset)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model input channels = TRAIN_DEPTH, output channels = 1 (for binary segmentation)
    model, criterion, optimizer = setup_model(TRAIN_Z_SIZE, 1, LEARNING_RATE, device)

    train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS, device)
    
    return model