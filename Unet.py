import torch
import torch.nn as nn
#import torch.optim as optim
#import torch.nn.functional as F



# --- Model Definition ---
class UNet(nn.Module):
    """Lightweight U-Net model for image segmentation."""
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc_conv1 = double_conv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv2 = double_conv(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv3 = double_conv(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv4 = double_conv(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck_conv = double_conv(256, 512)

        # Decoder
        self.upconv6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv6 = double_conv(512, 256)

        self.upconv7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv7 = double_conv(256, 128)

        self.upconv8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv8 = double_conv(128, 64)

        self.upconv9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv9 = double_conv(64, 32)

        # Output layer
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc_conv1(x)
        pool1 = self.pool1(enc1)

        enc2 = self.enc_conv2(pool1)
        pool2 = self.pool2(enc2)

        enc3 = self.enc_conv3(pool2)
        pool3 = self.pool3(enc3)

        enc4 = self.enc_conv4(pool3)
        pool4 = self.pool4(enc4)

        # Bottleneck
        bottleneck = self.bottleneck_conv(pool4)

        # Decoder
        up6 = self.upconv6(bottleneck)
        dec6 = self.dec_conv6(torch.cat([up6, enc4], dim=1))

        up7 = self.upconv7(dec6)
        dec7 = self.dec_conv7(torch.cat([up7, enc3], dim=1))

        up8 = self.upconv8(dec7)
        dec8 = self.dec_conv8(torch.cat([up8, enc2], dim=1))

        up9 = self.upconv9(dec8)
        dec9 = self.dec_conv9(torch.cat([up9, enc1], dim=1))

        # Output
        output = torch.sigmoid(self.out_conv(dec9))
        return output

class UNet2(nn.Module):
    """
    Improved U-Net model for image segmentation with Batch Normalization and Dropout.

    Key improvements:
    - Batch Normalization: Added after each convolution layer to stabilize training
      and accelerate convergence.
    - Dropout: Introduced in the bottleneck and potentially other deeper layers
      to prevent overfitting.
    - Flexible double_conv: Modified to include Batch Normalization.
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        """
        Initializes the UNet model.

        Args:
            in_channels (int): Number of input image channels (e.g., 3 for RGB, 1 for grayscale).
            out_channels (int): Number of output segmentation classes/channels.
            dropout_rate (float): The probability of an element to be zeroed.
                                  Used in Dropout layers. Set to 0 to disable.
        """
        # Changed: Using the simpler and more robust Python 3 super() call
        super().__init__()

        # Helper function for a double convolution block with ReLU and Batch Normalization
        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), # Bias=False when using BN
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), # Bias=False when using BN
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Encoder (Downsampling Path)
        # Each block consists of two convolutional layers followed by a max pooling layer.
        self.enc_conv1 = double_conv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv2 = double_conv(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv3 = double_conv(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv4 = double_conv(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck (Deepest part of the network)
        self.bottleneck_conv = double_conv(256, 512)
        # Add dropout in the bottleneck to prevent overfitting
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()


        # Decoder (Upsampling Path)
        # Each block consists of a transpose convolution (upsampling)
        # followed by concatenation with the corresponding encoder feature map,
        # and then two convolutional layers.

        # Up-convolution for bottleneck features
        self.upconv6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # The input channels for dec_conv6 are 512 (256 from upconv + 256 from enc4)
        self.dec_conv6 = double_conv(512, 256)

        self.upconv7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # The input channels for dec_conv7 are 256 (128 from upconv + 128 from enc3)
        self.dec_conv7 = double_conv(256, 128)

        self.upconv8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # The input channels for dec_conv8 are 128 (64 from upconv + 64 from enc2)
        self.dec_conv8 = double_conv(128, 64)

        self.upconv9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        # The input channels for dec_conv9 are 64 (32 from upconv + 32 from enc1)
        self.dec_conv9 = double_conv(64, 32)

        # Output layer
        # A 1x1 convolution to map the final feature maps to the desired number of output channels.
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Defines the forward pass of the UNet model.

        Args:
            x (torch.Tensor): Input tensor (image).

        Returns:
            torch.Tensor: Output segmentation mask.
        """
        # Encoder
        enc1 = self.enc_conv1(x)
        pool1 = self.pool1(enc1)

        enc2 = self.enc_conv2(pool1)
        pool2 = self.pool2(enc2)

        enc3 = self.enc_conv3(pool2)
        pool3 = self.pool3(enc3)

        enc4 = self.enc_conv4(pool3)
        pool4 = self.pool4(enc4)

        # Bottleneck
        bottleneck = self.bottleneck_conv(pool4)
        bottleneck = self.dropout(bottleneck) # Apply dropout here

        # Decoder
        # Upsample and concatenate with corresponding encoder feature map
        up6 = self.upconv6(bottleneck)
        # Ensure spatial dimensions match before concatenation if needed (F.interpolate can help)
        # If the spatial dimensions of up6 and enc4 don't perfectly match due to pooling/stride,
        # you might need to resize one of them. For typical U-Net, they should match.
        dec6 = self.dec_conv6(torch.cat([up6, enc4], dim=1))

        up7 = self.upconv7(dec6)
        dec7 = self.dec_conv7(torch.cat([up7, enc3], dim=1))

        up8 = self.upconv8(dec7)
        dec8 = self.dec_conv8(torch.cat([up8, enc2], dim=1))

        up9 = self.upconv9(dec8)
        dec9 = self.dec_conv9(torch.cat([up9, enc1], dim=1))

        # Output layer with sigmoid activation for binary segmentation
        # For multi-class segmentation, you would typically use nn.Softmax(dim=1)
        # or apply nn.LogSoftmax and use NLLLoss.
        output = torch.sigmoid(self.out_conv(dec9))
        return output


