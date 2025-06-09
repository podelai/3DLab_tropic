import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A residual block with two convolutional layers, batch normalization, and a skip connection.
    This helps in training deeper networks by preventing the vanishing gradient problem.
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to handle changes in channel dimensions
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # Add the original input (identity)
        out = self.relu(out)
        
        return out

class AttentionBlock(nn.Module):
    """
    Attention Gate (AG) to be used in U-Net skip connections.
    It learns to suppress irrelevant regions in an input feature map while highlighting salient features.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        # Convolution for the feature map from the previous layer (g)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # Convolution for the feature map from the skip connection (x)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # Final convolution to create the attention mask
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # Apply the attention mask to the skip connection features
        return x * psi


class AttentionResUNet(nn.Module):
    """
    A U-Net architecture that integrates Residual Blocks and Attention Gates.
    This model is robust and well-suited for complex image-to-image tasks.
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        
        # Encoder (Downsampling Path)
        self.enc1 = ResidualBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = ResidualBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock(512, 1024)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Decoder (Upsampling Path) with Attention Gates
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.attn4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.dec4 = ResidualBlock(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.attn3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.dec3 = ResidualBlock(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.attn2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.dec2 = ResidualBlock(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.attn1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.dec1 = ResidualBlock(128, 64)

        # Final Output Layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder Path
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        b = self.dropout(b)

        # Decoder Path with Attention
        d4 = self.upconv4(b)
        e4_attn = self.attn4(g=d4, x=e4)
        d4 = torch.cat((e4_attn, d4), dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        e3_attn = self.attn3(g=d3, x=e3)
        d3 = torch.cat((e3_attn, d3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        e2_attn = self.attn2(g=d2, x=e2)
        d2 = torch.cat((e2_attn, d2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        e1_attn = self.attn1(g=d1, x=e1)
        d1 = torch.cat((e1_attn, d1), dim=1)
        d1 = self.dec1(d1)
        
        # Apply sigmoid for binary segmentation output
        output = torch.sigmoid(self.out_conv(d1))
        
        return output
