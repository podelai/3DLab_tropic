# U-Net Image Segmentation

## Overview

This project implements a U-Net model for image segmentation tasks. It includes scripts for training the model, evaluating its performance, and applying it to segment large image stacks. The primary focus is on segmenting 2D slices from 3D volumetric data, often from TIFF image sequences.

## Features

- **U-Net Model Architectures:**
    - `UNet`: A lightweight U-Net implementation.
    - `UNet2`: An improved U-Net with Batch Normalization and Dropout for better training stability and generalization.
- **Training (`train.py`):**
    - Trains the U-Net model on provided image data.
    - Creates datasets by randomly cropping subvolumes from larger stacks.
    - Supports downscaling/upscaling for resolution differences between axes.
- **Evaluation (`eval.py`):**
    - Evaluates the trained model's performance.
    - Includes utilities for importing TIFF sequences.
    - Provides Jaccard Index calculation for binary segmentation assessment.
    - Implements morphing between slices using Euclidean Distance Transform (EDT) for creating interpolated stacks.
- **Application (`apply.py`):**
    - Applies a trained U-Net model to segment large 3D image stacks.
    - Processes slice by slice with spatial tiling to handle large inputs.
    - Supports upscaling of image height based on a resolution factor.
- **TIFF Image Support:**
    - Utilities for importing and exporting TIFF image sequences.

## File Structure

- `Unet.py`: Contains the PyTorch definitions for the `UNet` and `UNet2` model architectures.
- `train.py`: Script for training the U-Net model.
- `apply.py`: Script for applying a trained model to segment new images.
- `eval.py`: Script for evaluating model performance and performing image stack manipulations like morphing.
- `README.md`: This file, providing an overview and instructions for the project.

## Dependencies

The project relies on the following Python libraries:

- PyTorch
- NumPy
- scikit-image
- Pillow (PIL)

You can typically install these using pip:
```bash
pip install torch torchvision torchaudio numpy scikit-image Pillow
```

## Usage

The scripts are primarily designed to be run by configuring parameters within the files themselves (e.g., file paths, training settings).

### Training (`train.py`)

1.  **Prepare your data:** The input data (e.g., a 3D TIFF stack, referred to as `RAW_STACK` in `eval.py`) should be prepared. The `train_execution_process` function in `train.py` expects this data as a NumPy array argument.
2.  **Configure parameters:** Open `train.py` and modify constants such as:
    - `TRAIN_XY_SIZE`, `TRAIN_Z_SIZE`: Dimensions for training crops.
    - `TRAIN_NB`: Number of training samples to generate.
    - `RESOLUTION_FACTOR`: Factor for downscaling/upscaling.
    - `BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`: Training hyperparameters.
    - `RANDOM_STATE`: For reproducibility.
3.  **Run the script:** The training process is typically initiated from `eval.py`, which calls `train_execution_process` from `train.py`.

### Applying the Model (`apply.py`)

1.  **Trained Model:** The `eval.py` script uses the model directly returned by the `train_execution_process` function. If you intend to save a trained model and load it separately for `apply.py`, you would need to implement standard PyTorch model saving (e.g., `torch.save()`) after training and loading (e.g., `torch.load()`) before application.
2.  **Input Data:** The large image stack you want to segment.
3.  **Configure parameters:** The `apply_model_to_stack_tiled_slice_by_slice` function in `apply.py` is called by `eval.py`. Key parameters include:
    - `model`: The loaded PyTorch model.
    - `STACK`: The input NumPy array (image stack).
    - `train_depth`: The depth of input volumes the model expects (corresponds to `TRAIN_Z_SIZE` in training).
    - `RESOLUTION_FACTOR`: As used in training/data preparation.
    - `TRAIN_XY_SIZE`: Tile size for processing.
    - `batch_size`: Batch size for inference.
4.  **Run:** Typically, `eval.py` would load a model and then call the application script.

### Evaluation (`eval.py`)

`eval.py` serves as a primary script to orchestrate training, application, and evaluation.

1.  **Configure Paths and Parameters:**
    - `INPUT_IMAGES_DIRECTORY`: Directory containing the raw TIFF sequence for training/evaluation.
    - `OUTPUT_PREDICTION_OUTDIRECTORY`: Where to save segmented output TIFFs.
    - `OUTPUT_INTERPOLATED_DIRECTORY`: Where to save morphed/interpolated TIFFs.
    - Other parameters like `XY_RESOLUTION`, `Z_RESOLUTION`, `TRAIN_XY_SIZE`, etc., should be set according to your data and model.
2.  **Run the script:**
    ```bash
    python eval.py
    ```
    This script will typically:
    - Load the input TIFF sequence.
    - Downscale the stack based on `RESOLUTION_FACTOR`.
    - Train a new U-Net model using `train_execution_process`.
    - Apply the trained model to the downscaled stack using `apply_model_to_stack_tiled_slice_by_slice`.
    - Perform morphing on the downscaled stack.
    - Export the predicted segmentation and morphed stack as TIFF sequences.
    - Calculate and print Jaccard indices comparing the original raw stack with the morphed stack and the predicted segmentation.

## Model Details

### `UNet`
A basic U-Net architecture with a standard encoder-decoder structure using double convolution blocks and max pooling for downsampling, and transpose convolutions for upsampling. Skip connections concatenate features from the encoder to the decoder. The final output uses a sigmoid activation for binary segmentation.

### `UNet2`
An enhanced version of `UNet` that incorporates:
-   **Batch Normalization:** Added after each convolution (before ReLU) to stabilize training and potentially speed up convergence. `bias=False` is used in convolutional layers when Batch Normalization follows.
-   **Dropout:** A dropout layer is added in the bottleneck of the network to help prevent overfitting. The dropout rate is configurable.

Both models are designed for image segmentation and expect input tensors in the format (Batch, Channels, Depth, Height, Width) when processing 3D subvolumes for training, or (Batch, Channels, Height, Width) for 2D slice processing during application.

## Contributing

Contributions to this project are welcome. Please consider the following if you wish to contribute:
-   Follow the existing coding style.
-   Ensure new features or changes are well-documented.
-   Add tests for new functionality where applicable.
-   Open an issue to discuss significant changes before starting work.

## License

This project is currently not licensed. Please specify a license if you intend for others to use, modify, or distribute this code. Consider common open-source licenses like MIT, Apache 2.0, or GPL.
