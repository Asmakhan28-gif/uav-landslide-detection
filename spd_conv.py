"""
SPD-Conv Module Implementation
Spatial Pyramid Dilated Convolution for enhanced feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPDConv(nn.Module):
    """
    Space-to-Depth Convolution Module
    Reduces spatial dimensions while preserving feature information
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, scale=2):
        super(SPDConv, self).__init__()
        self.scale = scale
        
        # Calculate the new channel dimension after space-to-depth operation
        spd_channels = in_channels * (scale ** 2)
        
        # Non-strided convolution after space-to-depth
        self.conv = nn.Conv2d(
            spd_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def space_to_depth(self, x):
        """
        Space-to-depth transformation
        Reshapes spatial information to channel dimension
        """
        N, C, H, W = x.shape
        scale = self.scale
        
        # Ensure dimensions are divisible by scale
        assert H % scale == 0 and W % scale == 0, \
            f"Height ({H}) and Width ({W}) must be divisible by scale ({scale})"
        
        # Reshape and permute to move spatial info to channels
        x = x.view(N, C, H // scale, scale, W // scale, scale)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(N, C * scale * scale, H // scale, W // scale)
        
        return x
    
    def forward(self, x):
        # Apply space-to-depth transformation
        x = self.space_to_depth(x)
        
        # Apply convolution, batch norm, and activation
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        
        return x


class CBS(nn.Module):
    """
    CBS Module: Convolution + Batch Normalization + SiLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CBS, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


if __name__ == "__main__":
    # Test SPD-Conv module
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test input
    x = torch.randn(1, 64, 160, 160).to(device)
    
    # Initialize SPD-Conv
    spd_conv = SPDConv(64, 128, scale=2).to(device)
    
    # Forward pass
    output = spd_conv(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("SPD-Conv test passed!")
