
import os
from PIL import Image
import numpy as np
from scipy.ndimage import distance_transform_edt


from train import train_execution_process


from apply import apply_model_to_stack_tiled_slice_by_slice


def import_binary_tiff_sequence(directory_path: str) -> np.ndarray | None:
    """
    Imports a sequence of binary TIFF images from a directory into a 3D numpy array.

    Args:
        directory_path (str): The path to the directory containing the TIFF images.

    Returns:
        np.ndarray | None: A 3D boolean numpy array representing the image stack
                          (depth, height, width) if successful, otherwise None.
    """
    images = []
    try:
        # Filter for TIFF files and sort them to maintain sequence order
        file_names = sorted([f for f in os.listdir(directory_path) if f.lower().endswith(('.tif', '.tiff'))])
    except FileNotFoundError:
        print(f"Error: Directory not found at {directory_path}")
        return None
    except OSError as e:
        print(f"Error accessing directory: {directory_path}. Error: {e}")
        return None

    if not file_names:
        print(f"Warning: No TIFF images found in {directory_path}. Please ensure files end with .tif or .tiff")
        return None

    for file_name in file_names:
        full_path = os.path.join(directory_path, file_name)
        try:
            # Use 'with' statement for proper resource management (ensures image is closed)
            with Image.open(full_path) as img:
                img_array = np.array(img)

                # Assuming binary is 0 or 255. Convert to boolean (True for foreground/255, False for background/0).
                # This makes the array truly binary (True/False) which is often more memory efficient and clear.
                img_array_binary = (img_array == 255)

                # Optional: Add a check for image mode if strict binary input is required
                # if img.mode != '1' and img.mode != 'L': # '1' for 1-bit binary, 'L' for 8-bit grayscale
                #     print(f"Warning: Image {file_name} is not a typical binary or grayscale image. "
                #           "Proceeding with 255 thresholding.")

                images.append(img_array_binary)
        except (IOError, OSError) as e:
            print(f"Error: Could not open or process TIFF image: {full_path}. Error: {e}")
            # Continue to the next file if one fails
            continue
        except Exception as e:
            print(f"An unexpected error occurred while processing {full_path}: {e}")
            # Continue to the next file
            continue

    if not images:
        print(f"Error: No valid binary TIFF images were successfully loaded from {directory_path}")
        return None

    try:
        # Stack images along a new depth dimension. The result will be (num_images, height, width).
        image_stack = np.stack(images)
        return image_stack.astype(bool) # Ensure the final stack is boolean
    except ValueError as e:
        print(f"Error: Could not stack images. They might have inconsistent dimensions. Error: {e}")
        return None

def morph_stack_interpolated(original_stack: np.ndarray, num_steps_per_slice_pair: int) -> list[np.ndarray]:
    """
    Applies morphing between consecutive slices of a 3D image stack to create
    a higher resolution interpolated stack. This function combines the logic
    of iterating through slices and performing the Euclidean Distance Transform (EDT)
    interpolation for each pair.

    Args:
        original_stack (np.ndarray): The input 3D binary image stack (depth, height, width).
                                     Expected to be a boolean array.
        num_steps_per_slice_pair (int): The number of intermediate steps to generate
                                        between each pair of original slices.
                                        The total number of frames generated per pair
                                        will be `num_steps_per_slice_pair + 1`.

    Returns:
        list: A list of numpy arrays, representing the full interpolated 3D stack.
              Each array is a binary image (np.uint8, 0 or 1).
    """
    if original_stack.ndim != 3:
        raise ValueError("Input 'original_stack' must be a 3D numpy array.")
    if original_stack.shape[0] < 2:
        # If there's only one slice, there's nothing to morph between.
        # If 0 slices, it's an empty stack.
        print("Warning: Original stack has less than 2 slices. No morphing performed.")
        return [original_stack[0].astype(np.uint8)] if original_stack.shape[0] == 1 else []

    full_interpolated_stack = []
    num_slices = original_stack.shape[0]

    for i in range(num_slices - 1):
        slice_start = original_stack[i]
        slice_end = original_stack[i+1]

        # Ensure slices are boolean for EDT calculation
        if not np.issubdtype(slice_start.dtype, np.bool_):
            slice_start = slice_start.astype(bool)
        if not np.issubdtype(slice_end.dtype, np.bool_):
            slice_end = slice_end.astype(bool)

        # Calculate Signed Euclidean Distance Transform (SEDT) for both shapes.
        # SEDT is negative inside the shape, positive outside, and zero at the boundary.
        # distance_transform_edt expects True for foreground and False for background.
        # To get distance from foreground (inside), use `image`.
        # To get distance from background (outside), use `~image`.
        # SEDT = distance_from_background - distance_from_foreground
        edt_start = distance_transform_edt(~slice_start) - distance_transform_edt(slice_start)
        edt_end = distance_transform_edt(~slice_end) - distance_transform_edt(slice_end)

        morphed_frames_for_pair = []
        for j in range(num_steps_per_slice_pair + 1):
            # Calculate the interpolation factor (alpha) from 0 to 1
            alpha = j / num_steps_per_slice_pair
            
            # Linearly interpolate between the two signed distance fields
            interpolated_edt = (1 - alpha) * edt_start + alpha * edt_end
            
            # Threshold the interpolated EDT back to a binary image.
            # Pixels where the interpolated EDT is less than or equal to 0 are considered part of the shape.
            # Convert to np.uint8 (0 or 1) for consistency with image processing libraries.
            morphed_shape = (interpolated_edt <= 0).astype(np.uint8)
            morphed_frames_for_pair.append(morphed_shape)
        
        # Append all morphed frames from this sequence to the full interpolated stack.
        # We skip the first frame of subsequent sequences (i > 0) as it's a duplicate
        # of the last frame of the previous morphing sequence.
        if i == 0:
            full_interpolated_stack.extend(morphed_frames_for_pair)
        else:
            full_interpolated_stack.extend(morphed_frames_for_pair[1:]) # Avoid duplicating the end frame of previous morph
    
    return np.array(full_interpolated_stack)


def export_predicted_stack_to_tiff_sequence(STACK, OUTPUT_DIRECTORY):
    """
    Exports a 3D numpy array (predicted stack) as a sequence of binary TIFF images.

    Args:
        predicted_stack (np.ndarray): The 3D numpy array (H, W, D) containing
                                      predicted probabilities (0-1).
        PREDICTION_DIRECTORY (str): The path to the directory where TIFF files will be saved.
    """
    if not isinstance(STACK, np.ndarray) or STACK.ndim != 3:
        print("Error: predicted_stack must be a 3D numpy array.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    h, w, d = STACK.shape

    

    # Iterate through each slice in the depth dimension
    for k in range(d):
        # Get the current slice (H, W)
        slice_data = STACK[:, :, k]

        # Threshold the probability mask to create a binary image (0 or 255)
        # Values > 0.5 become 255 (white), others become 0 (black)
        binary_slice = (slice_data > 0.5).astype(np.uint8) * 255

        # Convert the numpy array slice to a PIL Image
        img = Image.fromarray(binary_slice, mode='L') # 'L' mode for 8-bit grayscale

        # Create a filename with zero-padding (e.g., slice_0000.tif)
        file_name = f"predicted_slice_{k:04d}.tif"
        file_path = os.path.join(OUTPUT_DIRECTORY, file_name)

        try:
            # Save the image as a TIFF file
            img.save(file_path, format='TIFF')
            # print(f"Saved {file_name}") # Optional: print for each file
        except (IOError, OSError) as e:
            print(f"Error saving TIFF image: {file_path}. Error: {e}")
            # Continue trying to save other slices
            continue
        except Exception as e:
            print(f"An unexpected error occurred while saving {file_path}: {e}")
            # Continue trying to save other slices
            continue

    #print(f"Export finished. {d} TIFF files saved to {OUTPUT_DIRECTORY}.")

def jaccard_index_binary_stacks(stack1: np.ndarray, stack2: np.ndarray) -> float:
    """
    Calculates the Jaccard index between two binary stacks.

    Args:
        stack1 (np.ndarray): The first binary stack (NumPy array).
                              Should contain only 0s and 1s.
        stack2 (np.ndarray): The second binary stack (NumPy array).
                              Should contain only 0s and 1s.

    Returns:
        float: The Jaccard index, a value between 0.0 and 1.0.
               Returns 1.0 if both stacks are empty (all zeros) and identical.
               Returns 0.0 if the union is zero (i.e., both stacks are entirely empty)
               and there's no intersection, or if one stack is empty and the other is not.

    Raises:
        ValueError: If the input stacks have different shapes or if they
                    contain values other than 0 or 1.
    """

    if stack1.shape != stack2.shape:
        raise ValueError("Input stacks must have the same shape.")

    if not np.all(np.isin(stack1, [0, 1])):
        raise ValueError("Stack1 must contain only binary values (0 or 1).")
    if not np.all(np.isin(stack2, [0, 1])):
        raise ValueError("Stack2 must contain only binary values (0 or 1).")

    # Flatten the stacks to treat them as 1D arrays for easier comparison
    flat_stack1 = stack1.flatten()
    flat_stack2 = stack2.flatten()

    # Calculate the intersection (where both are 1)
    intersection = np.sum((flat_stack1 == 1) & (flat_stack2 == 1))

    # Calculate the union (where at least one is 1)
    union = np.sum((flat_stack1 == 1) | (flat_stack2 == 1))

    if union == 0:
        # If both stacks are entirely empty (all zeros), the Jaccard index is often
        # considered 1.0 because they are perfectly similar.
        # If only one stack is empty, and the other is not, union will not be 0.
        return 1.0 if intersection == 0 else 0.0
    else:
        jaccard_index = intersection / union
        return jaccard_index

# --- Constants ---
INPUT_IMAGES_DIRECTORY = "C:/Users/lucie/Desktop/3Dlab_tropic_eval/data/mito_isotrope"
OUTPUT_PREDICTION_OUTDIRECTORY = "C:/Users/lucie/Desktop/3Dlab_tropic_eval/result/predicted_segmentation" # New output directory
OUTPUT_INTERPOLATED_DIRECTORY ="C:/Users/lucie/Desktop/3Dlab_tropic_eval/result/interpolated_segmentation"

XY_RESOLUTION = 2 # Resolution in x and y
Z_RESOLUTION = 20 # Resolution in z

TRAIN_XY_SIZE = 256 # Size of the training dataset in x and y
TRAIN_Z_SIZE = 1 # Size of the training dataset in z. Must be odd for centered slice label
TRAIN_NB = 80 #Number of .. for the training dataset
RANDOM_STATE = 3

BATCH_SIZE = 2
NUM_EPOCHS = 80
LEARNING_RATE = 0.001



RESOLUTION_FACTOR = int(Z_RESOLUTION / XY_RESOLUTION)


 

if __name__ == "__main__":

    print(f"Attempting to load binary TIFF sequence from: {INPUT_IMAGES_DIRECTORY}")
    RAW_STACK = import_binary_tiff_sequence(INPUT_IMAGES_DIRECTORY)
    #RAW_STACK = RAW_STACK.transpose(1,2,0)
    
    DOWNSCALED_STACK = RAW_STACK[::RESOLUTION_FACTOR]
    
    # Check if TRAIN_Z_SIZE is odd
    if TRAIN_Z_SIZE % 2 == 0:
        print("Error: TRAIN_Z_SIZE must be an odd number for the middle slice to be the label.")
        exit()
    
    print("Training")
    model = train_execution_process(DOWNSCALED_STACK.transpose(1,2,0), TRAIN_XY_SIZE, TRAIN_Z_SIZE, TRAIN_NB, RESOLUTION_FACTOR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, RANDOM_STATE)
    
    print("Apply")
    final_predicted_stack = apply_model_to_stack_tiled_slice_by_slice(model, DOWNSCALED_STACK, TRAIN_Z_SIZE, RESOLUTION_FACTOR, TRAIN_XY_SIZE, BATCH_SIZE)
    
    # Call the new merged function
    morphed_stack = morph_stack_interpolated(DOWNSCALED_STACK, RESOLUTION_FACTOR)
    
    print(f"\nExporting final_predicted_stack to {OUTPUT_PREDICTION_OUTDIRECTORY}...")
    export_predicted_stack_to_tiff_sequence(final_predicted_stack.transpose(1,2,0), OUTPUT_PREDICTION_OUTDIRECTORY)
    
    print(f"\nExporting morphed_stack to {OUTPUT_INTERPOLATED_DIRECTORY}...")
    #export_predicted_stack_to_tiff_sequence(morphed_stack.transpose(1,2,0), OUTPUT_INTERPOLATED_DIRECTORY)
    
    jaccard_index_morphed = jaccard_index_binary_stacks(RAW_STACK[0:491], morphed_stack)
    jaccard_index_prediction = jaccard_index_binary_stacks(RAW_STACK, (final_predicted_stack > 0.5).astype(np.uint8))
    
    print("DONE")
    


