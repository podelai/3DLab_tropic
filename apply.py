import numpy as np
import skimage.transform
import torch


def upscale_height(img, resolution_ratio):
    """Upscales along height (axis 0)."""
    original_shape = img.shape
    # Calculate target height for upscaling
    target_height_upscaled = int(original_shape[0] * resolution_ratio)

    upscale_shape = (target_height_upscaled, original_shape[1], original_shape[2])
    # Upscale using nearest-neighbor interpolation (order=0)
    img_upscaled = skimage.transform.resize(img.astype(float), upscale_shape, order=0, preserve_range=True)

    return img_upscaled.astype(bool)

def apply_model_to_stack_tiled_slice_by_slice(model, STACK, train_depth, RESOLUTION_FACTOR, TRAIN_XY_SIZE, batch_size):
    """
    Applies the trained model to a large image stack slice by slice with spatial tiling.

    For each slice, extracts a volume of size (H, W, train_depth) centered around it,
    tiles it spatially, applies transformations, predicts, and stitches the
    segmentation for the central slice of each tile back.

    Args:
        model (nn.Module): The trained PyTorch model.
        large_stack (np.ndarray): The large 3D numpy array (H, W, D) to process.
        train_depth (int): The expected depth of the input volume for the model.
        train_height (int): The expected height of the input tile for the model.
        train_width (int): The expected width of the input tile for the model.
        batch_size (int): The batch size for processing tiles.
        device (torch.device): The device to use for inference ('cuda' or 'cpu').

    Returns:
        np.ndarray: The predicted segmentation mask for the large stack (H, W, D).
                    Returns None if the input stack is invalid or processing fails.
                    The output values are the raw sigmoid outputs (probabilities between 0 and 1).
    """
    
    #STACK = np.transpose(STACK, axes=[2, 0, 1])
    print("STACK shape : " , STACK.shape)
    
    STACK = upscale_height(STACK, RESOLUTION_FACTOR)
    print("STACK shape : " , STACK.shape)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    

    if not isinstance(STACK, np.ndarray) or STACK.ndim != 3:
        print("Error: Input large_stack must be a 3D numpy array.")
        return None

    h, w, d = STACK.shape
    pad_depth = train_depth // 2 # Padding needed on each side in depth

    # Calculate padding for height and width to be divisible by tile size
    pad_h = (TRAIN_XY_SIZE - (h % TRAIN_XY_SIZE)) % TRAIN_XY_SIZE
    pad_w = (TRAIN_XY_SIZE - (w % TRAIN_XY_SIZE)) % TRAIN_XY_SIZE

    # Pad the stack in H, W, and D dimensions
    # Pad with zeros (constant mode)
    padded_stack = np.pad(STACK,
                          ((0, pad_h), (0, pad_w), (pad_depth, pad_depth)),
                          mode='constant',
                          constant_values=0) # Pad with 0 for binary data
    padded_h, padded_w, padded_d = padded_stack.shape

    # Initialize the output prediction stack. It will have the padded H and W,
    # but the original D since we predict one slice per original slice.
    # Store raw probabilities (float32)
    predicted_stack = np.zeros((padded_h, padded_w, d), dtype=np.float32)

    model.eval() # Set model to evaluation mode

    # Iterate through each slice of the original stack (index k from 0 to d-1)
    for k in range(d):
        #☻print(f"Processing slice {k+1}/{d}")
        # Extract a volume centered around slice k from the padded stack.
        # The volume starts at index k in the padded stack depth dimension
        # and has a depth of train_depth.
        index_pad_list = []
        for index_pad in range(train_depth):
            
            index_slice = int(k - (pad_depth*RESOLUTION_FACTOR) + (RESOLUTION_FACTOR*index_pad))
            #index_slice = int(k - (pad_depth*resolution_ratio) + (index_pad))
            
            if(index_slice < 0):
                #print("index : " , 0)
                pad_slice = 0
            elif(index_slice >= d):
                #print("index : " , d)
                pad_slice = d
            else:
                #print("index : " , int(k - (pad_depth*resolution_ratio) + (resolution_ratio*index_pad)))
                pad_slice = index_slice
                
            
            index_pad_list.append(pad_slice)
            
        #print(index_pad_list)    
        volume_hwd_padded = padded_stack[:, :, index_pad_list]
        #print(volume_hwd_padded.shape)

        # Process spatial tiles within this volume
        tile_volumes_batch = []
        tile_coords = [] # To store where to place the prediction

        # Iterate through tiles in height and width
        for i in range(0, padded_h, TRAIN_XY_SIZE):
            for j in range(0, padded_w, TRAIN_XY_SIZE):
                # Extract spatial tile from the volume
                
                tile_volume_hwd = volume_hwd_padded[i : i + TRAIN_XY_SIZE, j : j + TRAIN_XY_SIZE, :] # Shape (train_height, train_width, train_depth)

                # Transpose to (D, H, W) for model input
                tile_input = np.transpose(tile_volume_hwd, axes=[2, 0, 1])

                tile_volumes_batch.append(tile_input)
                tile_coords.append((i, j))

        if not tile_volumes_batch:
            # This case should be avoided with proper padding, but handle defensively
            print(f"Warning: No tiles generated for slice {k}. Skipping.")
            continue

        # Process tiles in batches
        for batch_idx in range(0, len(tile_volumes_batch), batch_size):
            batch_slice_volumes = tile_volumes_batch[batch_idx : batch_idx + batch_size]
            batch_coords = tile_coords[batch_idx : batch_idx + batch_size]

            # Stack the batch of tile volumes and convert to tensor
            batch_tensor = torch.from_numpy(np.stack(batch_slice_volumes, axis=0)).float().to(device)

            with torch.no_grad():
                # Get predictions for the batch of tiles
                # Output shape: (Batch_size, 1, TRAIN_HEIGHT, TRAIN_WIDTH)
                predictions = model(batch_tensor)

            # Convert predictions back to numpy and remove the channel dimension
            predictions_np = predictions.squeeze(1).cpu().numpy() # Shape (Batch_size, TRAIN_HEIGHT, TRAIN_WIDTH)

            # Store predictions in the output stack
            for idx, (i, j) in enumerate(batch_coords):
                # Store the prediction for the current tile at slice k
                predicted_stack[i : i + TRAIN_XY_SIZE, j : j + TRAIN_XY_SIZE, k] = predictions_np[idx, :, :]

    # Unpad the final predicted stack in H and W to match the original stack dimensions
    final_predicted_stack = predicted_stack[:h, :w, :]

    return final_predicted_stack

