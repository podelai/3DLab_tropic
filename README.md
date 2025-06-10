# SR AttentionResNet for Non-Isotropic Microscopy Image Stacks

## Overview
This repository contains the tools and implementation for training and applying a Super-Resolution (SR) model based on an Attention Residual U-Net (`AttentionResUNet`). The primary objective is to enhance the resolution of non-isotropic 3D microscopy image stacks, effectively performing super-resolution along the lower-resolution axis.

## Problem Description
Three-dimensional microscopy image stacks often exhibit non-isotropic resolution, meaning the resolution along one axis (typically the axial or Z-axis) is significantly lower than the resolution in the lateral (X and Y) planes. This anisotropy poses challenges for accurate 3D visualization, segmentation, and quantitative analysis of fine structures.

This project aims to address this by employing a deep learning-based super-resolution technique. The goal is to computationally reconstruct a higher-resolution representation of the image stack, particularly by improving the effective resolution along the Z-axis, thereby making the data more isotropic and suitable for detailed downstream analysis.

## Model Architecture
The core of this project is the **`AttentionResUNet`**, a deep neural network designed for image-to-image tasks. Its architecture combines several powerful concepts:

*   **U-Net Backbone**: The model is fundamentally a U-Net, characterized by its symmetric encoder-decoder structure. The encoder path progressively downsamples the input to capture contextual information, while the decoder path upsamples these features to reconstruct the output image. Skip connections between corresponding encoder and decoder layers allow the network to reuse low-level feature maps, which is crucial for preserving detail.

*   **Residual Blocks**: Instead of plain convolutional layers, the network utilizes Residual Blocks. Each block typically contains two convolutional layers with batch normalization and ReLU activation. A skip connection within each block adds the input of the block to its output. This design helps in training deeper networks by mitigating the vanishing gradient problem and allowing the network to learn identity mappings more easily.

*   **Attention Gates (AGs)**: Attention Gates are integrated into the skip connections. Before features from an encoder layer are passed to the corresponding decoder layer, they are processed by an attention gate. The gate takes input from both the encoder feature map and the feature map from the preceding decoder layer. It learns to generate an attention mask that emphasizes salient regions and suppresses irrelevant ones in the encoder features. This allows the decoder to focus on the most informative parts of the image for the reconstruction task.

The combination of the U-Net structure with residual connections and attention mechanisms makes the `AttentionResUNet` well-suited for complex image analysis tasks like super-resolution, where both contextual understanding and fine detail preservation are important.

## Workflow
The overall workflow is orchestrated by the `eval.py` script, which handles data loading, calls the training process, applies the trained model, and evaluates the results.

### Data Preparation
The data preparation pipeline involves several key steps:
1.  **Loading Raw Data**: The process starts by loading the original high-resolution 3D image stack, typically from a sequence of TIFF files (using `import_binary_tiff_sequence()` in `eval.py`).
2.  **Simulating Non-Isotropy**: To train the super-resolution model, a lower-resolution version of the stack is simulated. This is done by downsampling the raw stack along the Z-axis based on the `RESOLUTION_FACTOR` (e.g., if `XY_RESOLUTION` is 2 and `Z_RESOLUTION` is 20, the `RESOLUTION_FACTOR` would be 10, meaning every 10th slice is kept). The original Z-indices of these selected slices are stored for later use in evaluation (e.g., morphing). This downsampled stack (`DOWNSCALED_STACK_DHW` in `eval.py`) serves as the basis for generating training inputs.
3.  **Generating Training Samples**: The `create_cropped_image_dataset()` function in `train.py` is responsible for creating input/label pairs for training:
    *   **Random Cropping**: Sub-volumes of a defined `TRAIN_CROP_SIZE` (Z, Y, X) are randomly extracted from the `DOWNSCALED_STACK_DHW`.
    *   **Input Simulation**: For each cropped sub-volume, a low-resolution input for the model is simulated using `downscale_upscale_height()`. This function degrades the resolution along one spatial axis (e.g., height/Y-axis of the *crop*) and then upscales it back, mimicking the appearance of a low-resolution image.
    *   **Label Creation**: The corresponding high-resolution label is the central slice of the *original* (before `downscale_upscale_height()`) cropped sub-volume.
    *   **Content Filtering**: Samples are filtered to ensure the label slice contains a minimum amount of foreground content (`min_content_ratio`).
4.  **Data Augmentation**: The `augment_data()` function in `train.py` applies on-the-fly augmentations (random horizontal/vertical flips) to the training samples to increase dataset variability and improve model generalization.

### Training
The training process is managed by the `train_execution_process()` function within `train.py`, which is invoked by `eval.py`. Key aspects include:
*   **Model Initialization**: An instance of `AttentionResUNet` is created with specified parameters (e.g., input channels, number of classes, depth).
*   **Loss Function**: Binary Cross-Entropy Loss (`torch.nn.BCELoss`) is employed, suitable for tasks where the output is a probability map (e.g., segmentation or probability of being "high resolution").
*   **Optimizer**: The Adam optimizer (`torch.optim.Adam`) is used for its adaptive learning rate capabilities.
*   **Data Handling**:
    *   A `CustomDataset` (defined in `train.py`) is used to wrap the generated training samples. This dataset class applies data augmentation during data loading.
    *   PyTorch `DataLoader` instances are used to manage batching, shuffling, and parallel loading of training and validation data.
*   **Training Loop**: The model is trained for a specified number of `epochs`. In each epoch:
    *   The model is set to training mode (`model.train()`).
    *   It iterates through the training `DataLoader`, performs forward passes, calculates the loss, and updates model weights via backpropagation.
    *   The model is then set to evaluation mode (`model.eval()`).
    *   It iterates through the validation `DataLoader` to compute validation loss, without updating weights.
*   **Output**: The primary output is the trained `model` object (weights are in memory) and a history of training and validation losses, which can be plotted to assess training progress and identify potential overfitting.

### Application (Inference)
Once the model is trained, it can be applied to enhance the resolution of new (or existing) non-isotropic image stacks. This is handled by the `apply_model_to_stack_tiled_slice_by_slice()` function in `apply.py` (called by `eval.py`):
1.  **Input**: A trained `AttentionResUNet` model and a downsampled 3D image stack (similar to the input used for training, potentially a large volume).
2.  **Initial Upscaling**: The input stack (`STACK`, typically H,W,D) is first processed by `upscale_height()` (from `apply.py`). This function, as currently implemented in `apply.py`, upscales the stack along its first axis (Height) using the `RESOLUTION_FACTOR`. The `apply_model_to_stack_tiled_slice_by_slice()` function then proceeds with this modified stack.
3.  **Slice-by-Slice Processing**: The model processes the upscaled stack one slice at a time (iterating along the depth/Z-axis). For each target slice to be predicted:
    *   A local 3D volume of `train_depth` slices (centered around the current slice index) is extracted from the upscaled input stack.
    *   **Tiled Inference**: If the XY dimensions of this local volume are larger than `TRAIN_XY_SIZE` the model was trained on, the volume is processed in spatial tiles.
    *   The model predicts the super-resolved central slice for each tile or the entire volume (if no tiling was needed).
4.  **Stitching**: If tiling was used, the predicted tiles for each slice are stitched back together (e.g., by averaging overlapping regions or taking the center of predictions) to form the complete super-resolved slice.
5.  **Output**: The result is a new 3D stack where each slice is the model's prediction, representing an enhanced-resolution version of the input. These are typically probability maps if `BCELoss` was used.

### Evaluation
The `eval.py` script also manages the evaluation of the model's performance:
1.  **Baseline Generation**: A baseline for comparison is created using morphological interpolation. The `morph_stack_interpolated()` function takes the `DOWNSCALED_STACK_DHW` (which has sparsely sampled Z-slices) and uses the `downsampled_indices` to interpolate the missing slices, attempting to reconstruct the full `original_depth`. This provides a non-deep-learning benchmark.
2.  **Metrics**: The primary metric used is Intersection over Union (IoU), calculated by `calculate_iou()`. This function compares binary masks, so the raw ground truth, morphed stack, and predicted stack are likely binarized before comparison. IoU is calculated for:
    *   The morphed stack vs. the original (raw) ground truth stack.
    *   The model's predicted super-resolved stack vs. the original (raw) ground truth stack.
3.  **Visualization**: The `view_stacks_interactively()` function offers a crucial qualitative evaluation tool. It provides a slider-based interface to visually compare, slice by slice, the original raw stack, the morphed baseline stack, and the model's predicted stack side-by-side. This allows for direct visual assessment of the super-resolution quality.

## Usage Instructions

### Dependencies
This project relies on several Python libraries. It's recommended to use a virtual environment (e.g., `venv` or `conda`) to manage dependencies.

Key libraries include:
*   **PyTorch**: For building and training the neural network.
*   **NumPy**: For numerical operations, especially array manipulation.
*   **scikit-image**: For image processing tasks like resizing.
*   **`scikit-learn`**: For utilities like splitting data (`train_test_split`).
*   **Matplotlib**: For plotting loss curves and viewing images.
*   **`Pillow (PIL)`**: For loading TIFF image sequences.
*   **`SciPy`**: For specific functions like `distance_transform_edt()` used in morphing.

You would typically install these using pip:
```bash
pip install torch numpy scikit-image scikit-learn matplotlib pillow scipy
```
(Note: For PyTorch, follow the installation instructions on the official website (pytorch.org) to get the version compatible with your CUDA setup if GPU support is desired.)

### Running the Code
The main script to execute the entire workflow (data processing, training, inference, and evaluation) is `eval.py`.

1.  **Configure Paths**: Before running, you **must** update the following path constants at the beginning of `eval.py` to point to your data and desired output locations:
    *   `INPUT_IMAGES_DIRECTORY`
    *   `OUTPUT_PREDICTION_OUTDIRECTORY`
    *   `OUTPUT_INTERPOLATED_DIRECTORY`

2.  **Execute the script**:
    ```bash
    python eval.py
    ```
    This will:
    *   Load the raw image stack.
    *   Prepare downsampled and training data.
    *   Train the `AttentionResUNet` model.
    *   Apply the trained model to the downsampled stack for super-resolution.
    *   Perform morphological interpolation as a baseline.
    *   Calculate IoU metrics.
    *   Display an interactive viewer for comparing results.

### Configuration
Several parameters can be adjusted to control the behavior of the training and evaluation process. These are primarily located at the beginning of `eval.py`:

*   **Data Paths**:
    *   `INPUT_IMAGES_DIRECTORY`: Path to the directory containing the input TIFF sequence.
    *   `OUTPUT_PREDICTION_OUTDIRECTORY`: Directory to save model predictions (not explicitly saved in the current `eval.py` but planned for).
    *   `OUTPUT_INTERPOLATED_DIRECTORY`: Directory to save morphing results (not explicitly saved in the current `eval.py` but planned for).

*   **Resolution and Downsampling**:
    *   `XY_RESOLUTION`: In-plane resolution of the raw data (e.g., microns per pixel).
    *   `Z_RESOLUTION`: Axial resolution of the raw data (e.g., microns per pixel).
        *   These are used to calculate `RESOLUTION_FACTOR = Z_RESOLUTION // XY_RESOLUTION`, which determines the downsampling rate for the Z-axis and the target upscaling factor for the model.
*   **Training Data Generation (`eval.py` and `train.py` settings)**:
    *   `TRAIN_XY_SIZE`: The height and width of the 2D patches/tiles used for training and inference (e.g., 256).
    *   `TRAIN_Z_SIZE`: The depth (number of slices) of the 3D input volumes fed to the model during training (e.g., 5). **Must be an odd number.**
    *   `TRAIN_NB`: The total number of training samples to generate (e.g., 200).
    *   `MIN_CONTENT_RATIO` (in `train.py`): Minimum percentage of foreground pixels required in a label patch for it to be included in the training set.
    *   `VAL_SPLIT` (in `train.py`): Fraction of the generated dataset to be used for validation (e.g., 0.2 for 20%).

*   **Training Process (`eval.py` settings for `train_execution_process()`)**:
    *   `RANDOM_STATE`: Seed for random number generators to ensure reproducibility (e.g., 3).
    *   `BATCH_SIZE`: Number of samples per batch during training and inference (e.g., 4).
    *   `NUM_EPOCHS`: Total number of times the training algorithm will iterate over the entire training dataset (e.g., 10).
    *   `LEARNING_RATE`: The step size at which the optimizer updates model weights (e.g., 0.001).
    *   `dropout_rate` (internal to `setup_model()` in `train.py`, but can be exposed): Dropout probability for regularization in the model's bottleneck.

Adjust these parameters based on your specific dataset, hardware capabilities, and desired outcomes.

## Expected Output
After running the `eval.py` script, you can expect the following outputs:

1.  **Console Output**:
    *   **Progress Updates**: Messages indicating the current stage of the process (e.g., "Loading TIFF sequence...", "Generating training samples...", "Starting training...", "Epoch [X/Y] Train Loss: ..., Val Loss: ...", "Applying model...", "Calculating IoU...").
    *   **Device Information**: Indication of whether the computations are being performed on CPU or GPU (e.g., "Computation will be performed on device: cuda").
    *   **IoU Scores**: Printed Intersection over Union (IoU) values comparing the morphological interpolation and the model's prediction against the ground truth raw stack. For example:
        ```
        IoU for Morphed Stack: 0.XXXX
        IoU for Predicted Stack: 0.YYYY
        ```

2.  **Visualizations**:
    *   **Loss Curves**: A Matplotlib window will pop up showing the training and validation loss curves over epochs, generated by `plot_loss_history()`. This helps assess the training progress and look for signs of overfitting.
    *   **Training Sample Previews**: During the training phase (if `display_training_results()` is called), Matplotlib windows will show a few examples of input slices, ground truth labels, and corresponding model predictions from both the training and validation sets.
    *   **Interactive Stack Viewer**: After training and prediction, an interactive Matplotlib window titled "Slice Viewer" will appear, generated by `view_stacks_interactively()`. This viewer displays three panels side-by-side:
        *   Raw Ground Truth stack
        *   Morphed Interpolation stack
        *   Model Prediction stack (binarized)
        A slider allows you to navigate through the different Z-slices of the stacks for visual comparison.

3.  **Trained Model (In Memory)**:
    *   The script trains and returns a `model` object (an instance of `AttentionResUNet`). While the current `eval.py` script does not automatically save this model to a file, it is available in memory and could be saved using `torch.save(model.state_dict(), 'path_to_your_model.pth')` if you wish to reuse it later without retraining.

4.  **Processed Image Stacks (In Memory)**:
    *   The `final_predicted_stack` (output from the model) and `morphed_stack_dhw` (output from morphological interpolation) are generated as NumPy arrays. Similar to the model, these are not explicitly saved to disk by `eval.py` but are used for IoU calculation and visualization. They could be saved as TIFF stacks or other formats if needed.

The primary outputs are geared towards immediate evaluation and visualization of the model's performance.

## Contributing

## License
