import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import skimage.transform
import matplotlib.pyplot as plt

# Import the new, more powerful model
from AttentionResUNet import AttentionResUNet

def random_crop_stack(image_stack, crop_size):
    """
    Randomly crops a sub-volume from a 3D image stack.
    Ensures that the crop dimensions do not exceed the stack dimensions.
    """
    if not isinstance(image_stack, np.ndarray):
        raise TypeError("image_stack must be a numpy.ndarray")
    for i in range(3):
        if crop_size[i] > image_stack.shape[i]:
            raise ValueError(f"Crop size at dim {i} ({crop_size[i]}) is larger than stack size ({image_stack.shape[i]})")
    
    max_d_start = image_stack.shape[0] - crop_size[0]
    max_h_start = image_stack.shape[1] - crop_size[1]
    max_w_start = image_stack.shape[2] - crop_size[2]
    
    d_start = np.random.randint(0, max_d_start + 1)
    h_start = np.random.randint(0, max_h_start + 1)
    w_start = np.random.randint(0, max_w_start + 1)

    
    return image_stack[d_start:d_start + crop_size[0],
                       h_start:h_start + crop_size[1],
                       w_start:w_start + crop_size[2]
                       ]

def downscale_upscale_height(img, resolution_ratio):
    """
    Simulates a low-resolution input by downscaling and then upscaling an image stack
    along its height (axis 1) using nearest-neighbor interpolation.
    """
    original_shape = img.shape
    target_height_downscaled = max(1, int(original_shape[1] / resolution_ratio))
    
    downscale_shape = (original_shape[0], target_height_downscaled, original_shape[2])
    # Downscale using nearest-neighbor (order=0) to preserve sharp edges
    img_downscaled = skimage.transform.resize(img.astype(float), downscale_shape, order=0, preserve_range=True)
    
    # Upscale back to original height, also with nearest-neighbor
    img_upscaled = skimage.transform.resize((img_downscaled > 0.5).astype(float), original_shape, order=0, preserve_range=True)
    
    return img_upscaled.astype(bool)

def augment_data(input_volume, label_image):
    """
    Applies random geometric augmentations to an input volume and its corresponding label.
    Augmentations include horizontal/vertical flips and 90-degree rotations.
    """
    # Make copies to avoid modifying the original arrays
    input_volume = input_volume.copy()
    label_image = label_image.copy()

    # 50% chance of horizontal flip
    if np.random.rand() > 0.5:
        input_volume = np.flip(input_volume, axis=2).copy()  # Flip along width axis (W)
        label_image = np.flip(label_image, axis=2).copy()   # Flip along width axis (W)

    # 50% chance of vertical flip
    if np.random.rand() > 0.5:
        input_volume = np.flip(input_volume, axis=1).copy()  # Flip along height axis (H)
        label_image = np.flip(label_image, axis=1).copy()   # Flip along height axis (H)

    return input_volume, label_image

class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset to handle on-the-fly data augmentation.
    """
    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.augment:
            image, label = augment_data(image, label)

        return torch.from_numpy(image).float(), torch.from_numpy(label).float()

def create_cropped_image_dataset(image_stack, num_samples, crop_size, resolution_ratio, random_state, min_content_ratio=0.1):
    """
    Generates a dataset of input/label pairs by randomly cropping the original stack.
    The input is a low-resolution simulation, and the label is the high-resolution center slice.
    This version filters out samples where the label's content is below the specified ratio.
    """
    if random_state is not None:
        np.random.seed(random_state)

    crop_depth, _, _ = crop_size
    
    valid_labels = []
    valid_inputs = []
    
    # To prevent an infinite loop if no valid samples can be found
    max_attempts = num_samples * 20 
    attempts = 0

    print(f"Generating {num_samples} training samples with at least {min_content_ratio*100:.2f}% content...")
    while len(valid_labels) < num_samples and attempts < max_attempts:
        cropped_image = random_crop_stack(image_stack, crop_size)
        
        # The high-resolution label is the center slice of the crop
        label = cropped_image[crop_depth // 2, :, :]
        
        # Check if the label's content ratio is above the minimum threshold
        # np.mean of a boolean array gives the ratio of True values
        if np.mean(label) >= min_content_ratio:
            # The input is the simulated low-resolution version of the crop
            input_data = downscale_upscale_height(cropped_image, resolution_ratio)
            
            valid_labels.append(label)
            valid_inputs.append(input_data)
        
        attempts += 1

    if len(valid_labels) < num_samples:
        print(f"Warning: Could only find {len(valid_labels)} valid samples after {max_attempts} attempts.")

    # Convert lists to numpy arrays
    if not valid_labels: # Handle case where no valid samples are found
        return np.array([]), np.array([])
        
    labels_np = np.array(valid_labels)
    inputs_np = np.array(valid_inputs)

    # Reshape for PyTorch: (N, C, H, W) for labels and (N, D, H, W) for inputs
    labels_np = np.expand_dims(labels_np, axis=1)
    
    return labels_np, inputs_np


def setup_model(in_channels, out_channels, learning_rate, device, dropout_rate=0.2):
    """Initializes the model, loss function, and optimizer."""
    model = AttentionResUNet(in_channels, out_channels, dropout_rate=dropout_rate).to(device)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary tasks
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    Trains the model and validates it at the end of each epoch.
    Returns the trained model and a history of training/validation losses.
    """
    print("Starting training...")
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
        
        avg_train_loss = running_train_loss / len(train_loader.dataset)
        history['train_loss'].append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)

        avg_val_loss = running_val_loss / len(val_loader.dataset)
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f}")

    print("Training finished.")
    return model, history

def plot_loss_history(history):
    """Plots the training and validation loss curves from the history dictionary."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Training & Validation Loss History', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

def display_training_results(model, train_loader, val_loader, device, num_samples=3):
    """
    Displays a comparison of input, ground truth, and model prediction
    for a few samples from the training and validation sets.
    """
    model.eval()
    
    def plot_samples(loader, set_name):
        # Get a single batch from the data loader
        try:
            inputs, labels = next(iter(loader))
        except StopIteration:
            print(f"Warning: Could not get a batch from the {set_name} loader. It might be empty.")
            return

        inputs, labels = inputs.to(device), labels.to(device)

        # Get model predictions
        with torch.no_grad():
            outputs = model(inputs)
        
        # Move data to CPU and convert to numpy for plotting
        inputs_np = inputs.cpu().numpy()
        labels_np = labels.cpu().numpy().squeeze(1)  # Remove channel dim, shape (N, H, W)
        outputs_np = outputs.cpu().numpy().squeeze(1) # Remove channel dim, shape (N, H, W)

        print(f"\n--- Displaying results for {set_name} set ---")
        for i in range(min(num_samples, len(inputs_np))):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # The input to the model is a z-stack (N, D, H, W), we show the middle slice
            # which corresponds to the label.
            input_slice = inputs_np[i, inputs_np.shape[1] // 2, :, :]
            
            axes[0].imshow(input_slice, cmap='gray')
            axes[0].set_title('Input (Center Slice of Low-Res Crop)')
            axes[0].axis('off')

            axes[1].imshow(labels_np[i], cmap='gray')
            axes[1].set_title('Ground Truth Label (High-Res)')
            axes[1].axis('off')

            # Apply a threshold to the prediction for visualization
            prediction = (outputs_np[i] > 0.5).astype(np.float32)
            axes[2].imshow(prediction, cmap='gray')
            axes[2].set_title('Model Prediction (Thresholded at 0.5)')
            axes[2].axis('off')

            plt.suptitle(f'{set_name} Sample {i+1}', fontsize=16)
            plt.tight_layout()
            plt.show()

    print("\nGenerating visual comparison of model predictions...")
    plot_samples(train_loader, "Training")
    plot_samples(val_loader, "Validation")


def train_execution_process(RAW_STACK_DHW, TRAIN_XY_SIZE, TRAIN_Z_SIZE, TRAIN_NB, RESOLUTION_FACTOR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, RANDOM_STATE):
    """
    Main execution function to orchestrate the data preparation, model training, and evaluation.
    """
    TRAIN_CROP_SIZE = (TRAIN_Z_SIZE, TRAIN_XY_SIZE, TRAIN_XY_SIZE)
    VAL_SPLIT = 0.2  # Reserve 20% of data for validation
    MIN_CONTENT_RATIO = 0.1 # Minimum percentage of non-black pixels in a label to be included

    if RAW_STACK_DHW is None:
        print("Error: The provided raw stack is empty. Aborting.")
        return None, None

    # 1. Generate the full dataset of input-label pairs
    labels_full, images_full = create_cropped_image_dataset(
        RAW_STACK_DHW, TRAIN_NB, TRAIN_CROP_SIZE, RESOLUTION_FACTOR, 
        random_state=RANDOM_STATE, min_content_ratio=MIN_CONTENT_RATIO
    )

    if images_full.size == 0:
        print("Error: No valid training data could be generated based on the criteria. Aborting.")
        return None, None

    # 2. Split the dataset into training and validation sets
    images_train, images_val, labels_train, labels_val = train_test_split(
        images_full, labels_full, test_size=VAL_SPLIT, random_state=RANDOM_STATE
    )
    
    # 3. Create PyTorch CustomDatasets and DataLoaders
    # Apply augmentations only to the training set
    train_dataset = CustomDataset(images_train, labels_train, augment=True)
    val_dataset = CustomDataset(images_val, labels_val, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Dataset prepared: {len(train_dataset)} training samples (with augmentation), {len(val_dataset)} validation samples.")

    # 4. Set up and train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation will be performed on device: {device}")

    model, criterion, optimizer = setup_model(
        in_channels=TRAIN_Z_SIZE, 
        out_channels=1, 
        learning_rate=LEARNING_RATE, 
        device=device,
        dropout_rate=0.3  # Dropout rate can be tuned
    )

    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device)
    
    # 5. Plot the training and validation history for analysis
    plot_loss_history(history)
    
    # 6. Display visual results on training and validation data
    display_training_results(model, train_loader, val_loader, device)

    return model, history
