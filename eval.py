import os
from PIL import Image
import numpy as np
import skimage.transform
from scipy.ndimage import distance_transform_edt
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from train import train_execution_process
from apply import apply_model_to_stack_tiled_slice_by_slice


def import_binary_tiff_sequence(directory_path: str):
    """
    Imports a sequence of binary TIFF images from a directory into a single,
    memory-pre-allocated 3D numpy array.

    Args:
        directory_path (str): The path to the directory containing the TIFF images.

    Returns:
        Optional[np.ndarray]: A 3D boolean numpy array (depth, height, width)
                              if successful, otherwise None.
    """
    try:
        file_names = sorted([f for f in os.listdir(directory_path) if f.lower().endswith(('.tif', '.tiff'))])
    except FileNotFoundError:
        print(f"Error: Directory not found at {directory_path}")
        return None

    if not file_names:
        print(f"Warning: No TIFF images found in {directory_path}.")
        return None

    try:
        with Image.open(os.path.join(directory_path, file_names[0])) as first_img:
            height, width = np.array(first_img).shape

        num_images = len(file_names)
        print(f"Found {num_images} images. Allocating stack of shape ({num_images}, {height}, {width}).")
        image_stack = np.empty((num_images, height, width), dtype=bool)

        for i, file_name in enumerate(file_names):
            full_path = os.path.join(directory_path, file_name)
            with Image.open(full_path) as img:
                img_array = np.array(img)
                image_stack[i] = (img_array == 255)

    except (IOError, OSError) as e:
        print(f"Error: Could not process TIFF image: {full_path}. Error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    return image_stack


def morph_stack_interpolated(downsampled_stack: np.ndarray, original_indices: np.ndarray, target_depth: int):
    """
    Applies morphing between consecutive slices of a downsampled 3D image stack 
    to create a higher resolution interpolated stack of a specific target depth.
    This version correctly handles unevenly spaced slices.

    Args:
        downsampled_stack (np.ndarray): The input 3D binary image stack (D, H, W)
                                        containing sparsely sampled slices.
        original_indices (np.ndarray): A 1D array of the original depth indices
                                       corresponding to the slices in downsampled_stack.
        target_depth (int): The desired depth of the final interpolated stack (e.g., the depth of the raw stack).

    Returns:
        np.ndarray: The full interpolated 3D stack as a numpy array with shape (target_depth, H, W).
    """
    if downsampled_stack.ndim != 3:
        raise ValueError("Input 'downsampled_stack' must be a 3D numpy array.")
    if len(downsampled_stack) != len(original_indices):
        raise ValueError("The number of slices in downsampled_stack must match the number of original_indices.")

    num_slices, height, width = downsampled_stack.shape
    
    if num_slices < 2:
        print("Warning: Downsampled stack has less than 2 slices. No morphing performed.")
        if num_slices == 1:
            # If only one slice, tile it to fill the target depth
            return np.repeat(downsampled_stack, target_depth, axis=0).astype(np.uint8)
        else:
            return np.zeros((target_depth, height, width), dtype=np.uint8)

    # Pre-allocate the full output stack with the correct target depth
    full_interpolated_stack = np.zeros((target_depth, height, width), dtype=np.uint8)

    # Place the original downsampled slices into the output stack at their correct indices
    for i, idx in enumerate(original_indices):
        if idx < target_depth:
            full_interpolated_stack[idx] = downsampled_stack[i].astype(np.uint8)

    # Iterate through the GAPS between the downsampled slices to interpolate
    for i in range(num_slices - 1):
        slice_start = downsampled_stack[i].astype(bool)
        slice_end = downsampled_stack[i+1].astype(bool)

        # Get the start and end indices in the final stack for the current gap
        index_start = original_indices[i]
        index_end = original_indices[i+1]

        # Number of new frames to generate is determined by the gap size
        num_steps_in_gap = index_end - index_start
        
        if num_steps_in_gap <= 0:
            continue

        # Calculate Signed Euclidean Distance Transform (SEDT) for the start and end shapes
        edt_start = distance_transform_edt(~slice_start) - distance_transform_edt(slice_start)
        edt_end = distance_transform_edt(~slice_end) - distance_transform_edt(slice_end)

        # Generate intermediate frames to fill the gap
        for j in range(1, num_steps_in_gap):
            # The interpolation factor `alpha` is now specific to each gap's size
            alpha = j / num_steps_in_gap
            
            interpolated_edt = (1 - alpha) * edt_start + alpha * edt_end
            morphed_shape = (interpolated_edt <= 0).astype(np.uint8)
            
            output_index = index_start + j
            if output_index < target_depth:
                full_interpolated_stack[output_index] = morphed_shape

    return full_interpolated_stack


def view_stacks_interactively(raw_stack, morphed_stack, predicted_stack):
    """
    Creates an interactive viewer to compare three 3D image stacks side-by-side
    with a slider to navigate through the slices.

    Args:
        raw_stack (np.ndarray): The ground truth stack (D, H, W).
        morphed_stack (np.ndarray): The interpolated stack (D, H, W).
        predicted_stack (np.ndarray): The model's prediction stack (D, H, W).
    """

    # Pad the predicted stack if its depth is smaller than the raw stack
    depth_diff = raw_stack.shape[0] - predicted_stack.shape[0]
    if depth_diff > 0:
        padding = ((0, depth_diff), (0, 0), (0, 0))
        predicted_stack = np.pad(predicted_stack, padding, mode='constant', constant_values=0)

    # Ensure all stacks have the same dimensions for consistent slicing
    if not (raw_stack.shape == morphed_stack.shape == predicted_stack.shape):
        print("Warning: Stack shapes do not match after processing. Visualization might be inconsistent.")
        print(f"Shapes: Raw={raw_stack.shape}, Morphed={morphed_stack.shape}, Predicted={predicted_stack.shape}")

    # Binarize the prediction for clear visualization
    predicted_stack_binary = (predicted_stack > 0.5).astype(np.uint8)

    depth = raw_stack.shape[0]
    initial_slice = depth // 2

    # --- Set up the plot ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(bottom=0.25) # Make room for slider

    im1 = ax1.imshow(raw_stack[initial_slice], cmap='gray')
    ax1.set_title('Raw Ground Truth')
    ax1.axis('off')

    im2 = ax2.imshow(morphed_stack[initial_slice], cmap='gray')
    ax2.set_title('Morphed Interpolation')
    ax2.axis('off')

    im3 = ax3.imshow(predicted_stack_binary[initial_slice], cmap='gray')
    ax3.set_title('Model Prediction')
    ax3.axis('off')

    fig.suptitle(f'Slice Viewer: {initial_slice}/{depth-1}', fontsize=16)

    # --- Create the slider ---
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(
        ax=ax_slider,
        label='Slice Index (z)',
        valmin=0,
        valmax=depth - 1,
        valinit=initial_slice,
        valstep=1
    )

    # --- Update function for the slider ---
    def update(val):
        slice_idx = int(slider.val)
        im1.set_data(raw_stack[slice_idx])
        im2.set_data(morphed_stack[slice_idx])
        im3.set_data(predicted_stack_binary[slice_idx])
        fig.suptitle(f'Slice Viewer: {slice_idx}/{depth-1}')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


# --- Evaluation Functions ---
def calculate_iou(pred: np.ndarray, true: np.ndarray) -> float:
    """Calculates Intersection over Union for binary images."""
    # Ensure stacks have the same shape for comparison
    if pred.shape != true.shape:
        # Pad the smaller stack to match the larger one for IoU calculation
        max_shape = np.maximum(pred.shape, true.shape)
        
        pad_pred = [(0, max_d - s) for s, max_d in zip(pred.shape, max_shape)]
        pred = np.pad(pred, pad_pred, mode='constant', constant_values=0)
        
        pad_true = [(0, max_d - s) for s, max_d in zip(true.shape, max_shape)]
        true = np.pad(true, pad_true, mode='constant', constant_values=0)
        
    pred_bool = (pred > 0.5)
    true_bool = (true > 0.5)

    intersection = np.logical_and(pred_bool, true_bool)
    union = np.logical_or(pred_bool, true_bool)

    if np.sum(union) == 0:
        return 1.0

    iou = np.sum(intersection) / np.sum(union)
    return iou


# --- Constants ---
INPUT_IMAGES_DIRECTORY = "C:/Users/lucie/Desktop/3Dlab_tropic_eval/data/mito_isotrope"
OUTPUT_PREDICTION_OUTDIRECTORY = "C:/Users/lucie/Desktop/3Dlab_tropic_eval/result/predicted_segmentation"
OUTPUT_INTERPOLATED_DIRECTORY ="C:/Users/lucie/Desktop/3Dlab_tropic_eval/result/interpolated_segmentation"

XY_RESOLUTION = 2
Z_RESOLUTION = 20
TRAIN_XY_SIZE = 256
TRAIN_Z_SIZE = 5

TRAIN_NB = 200
RANDOM_STATE = 3
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

if __name__ == "__main__":

    print(f"Attempting to load binary TIFF sequence from: {INPUT_IMAGES_DIRECTORY}")
    RAW_STACK_DHW = import_binary_tiff_sequence(INPUT_IMAGES_DIRECTORY)
    
    if RAW_STACK_DHW is None:
        exit("Failed to load the raw stack. Aborting.")     
    
    RESOLUTION_FACTOR = Z_RESOLUTION // XY_RESOLUTION
    original_depth = RAW_STACK_DHW.shape[0]

    # --- Corrected Downsampling ---
    # Create indices for downsampling, ensuring the very last slice of the
    # original stack is included so we can interpolate all the way to the end.
    downsampled_indices = np.arange(0, original_depth, RESOLUTION_FACTOR)
    if original_depth - 1 not in downsampled_indices:
        downsampled_indices = np.append(downsampled_indices, original_depth - 1)
        # Ensure indices are sorted and unique after appending
        downsampled_indices = np.unique(np.sort(downsampled_indices))

    DOWNSCALED_STACK_DHW = RAW_STACK_DHW[downsampled_indices, :, :]

    if TRAIN_Z_SIZE % 2 == 0:
        print("Error: TRAIN_Z_SIZE must be an odd number for the middle slice to be the label.")
        exit()

    print("Training")
    model, history = train_execution_process(DOWNSCALED_STACK_DHW, TRAIN_XY_SIZE, TRAIN_Z_SIZE, TRAIN_NB, RESOLUTION_FACTOR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, RANDOM_STATE)
    
    print("Apply")
    # Note: the prediction is on n-1 slices
    final_predicted_stack = apply_model_to_stack_tiled_slice_by_slice(model, DOWNSCALED_STACK_DHW[:-1,:,:], TRAIN_Z_SIZE, RESOLUTION_FACTOR, TRAIN_XY_SIZE, BATCH_SIZE)
    
    print("Morphing")
    # --- Corrected Morphing Call ---
    # Pass the downscaled stack (D,H,W), its original indices, and the target depth.
    morphed_stack_dhw = morph_stack_interpolated(
        DOWNSCALED_STACK_DHW, 
        downsampled_indices,
        original_depth
    )

    
    print(f"\nRAW_STACK_DHW shape: {RAW_STACK_DHW.shape}")
    print(f"Morphed stack shape: {morphed_stack_dhw.shape}")
    # Note: predicted stack has shape (H, W, D)
    print(f"Predicted stack shape: {final_predicted_stack.shape}\n")

    # --- IoU Calculation ---
    # The morphed stack should now have the same dimensions as the raw stack
    IOU_morphed = calculate_iou(morphed_stack_dhw, RAW_STACK_DHW)
    # Transpose prediction for IoU calculation
    IOU_prediction = calculate_iou(final_predicted_stack, RAW_STACK_DHW)

    print(f"IoU for Morphed Stack: {IOU_morphed:.4f}")
    print(f"IoU for Predicted Stack: {IOU_prediction:.4f}")

    # --- Interactive Visualization ---
    print("\nLaunching interactive stack viewer...")
    view_stacks_interactively(RAW_STACK_DHW, morphed_stack_dhw, final_predicted_stack)

    print("DONE")
