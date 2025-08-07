"""
SPANet (Spatial Path Aggregation Network) Implementation
Enhanced feature fusion network for multi-scale context awareness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from spd_conv import CBS


class C2F(nn.Module):
    """
    C2F Module: CSP Bottleneck with 2 convolutions
    Lightweight module inspired by ELAN design
    """
    def __init__(self, in_channels, out_channels, n=1, shortcut=False):
        super(C2F, self).__init__()
        self.c = int(out_channels * 0.5)  # hidden channels
        self.cv1 = CBS(in_channels, 2 * self.c, 1, 1, 0)
        self.cv2 = CBS((2 + n) * self.c, out_channels, 1, 1, 0)
        self.m = nn.ModuleList([Bottleneck(self.c, self.c, shortcut) for _ in range(n)])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Bottleneck(nn.Module):
    """Standard bottleneck block"""
    def __init__(self, in_channels, out_channels, shortcut=True):
        super(Bottleneck, self).__init__()
        self.cv1 = CBS(in_channels, out_channels, 1, 1, 0)
        self.cv2 = CBS(out_channels, out_channels, 3, 1, 1)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SPANet(nn.Module):
    """
    Spatial Path Aggregation Network
    Enhanced PANet with shallow feature integration (P2 layer)
    """
    def __init__(self, channels_list=[256, 512, 1024]):  # P2, P3, P4 channels
        super(SPANet, self).__init__()
        
        # Channels for different feature levels
        self.p2_channels = channels_list[0]  # 256
        self.p3_channels = channels_list[1]  # 512
        self.p4_channels = channels_list[2]  # 1024
        
        # Top-down pathway
        self.p4_to_p3 = CBS(self.p4_channels, self.p3_channels, 1, 1, 0)
        self.p3_to_p2 = CBS(self.p3_channels, self.p2_channels, 1, 1, 0)
        
        # Feature fusion modules
        self.p3_fusion = C2F(self.p3_channels * 2, self.p3_channels)
        self.p2_fusion = C2F(self.p2_channels * 2, self.p2_channels)
        
        # Bottom-up pathway
        self.p2_to_p3_down = CBS(self.p2_channels, self.p3_channels, 3, 2, 1)
        self.p3_to_p4_down = CBS(self.p3_channels, self.p4_channels, 3, 2, 1)
        
        # Final fusion modules
        self.p3_final_fusion = C2F(self.p3_channels * 2, self.p3_channels)
        self.p4_final_fusion = C2F(self.p4_channels * 2, self.p4_channels)
        
    def forward(self, features):
        """
        Args:
            features: List of feature maps [P2, P3, P4] from backbone
        Returns:
            List of enhanced feature maps [P2', P3'', P4'']
        """
        p2, p3, p4 = features
        
        # Top-down pathway
        # P4 -> P3
        p4_up = F.interpolate(self.p4_to_p3(p4), size=p3.shape[2:], mode='bilinear', align_corners=False)
        p3_fused = self.p3_fusion(torch.cat([p3, p4_up], dim=1))
        
        # P3 -> P2  
        p3_up = F.interpolate(self.p3_to_p2(p3_fused), size=p2.shape[2:], mode='bilinear', align_corners=False)
        p2_fused = self.p2_fusion(torch.cat([p2, p3_up], dim=1))
        
        # Bottom-up pathway
        # P2 -> P3
        p2_down = self.p2_to_p3_down(p2_fused)
        p3_final = self.p3_final_fusion(torch.cat([p3_fused, p2_down], dim=1))
        
        # P3 -> P4
        p3_down = self.p3_to_p4_down(p3_final)
        p4_final = self.p4_final_fusion(torch.cat([p4, p3_down], dim=1))
        
        return [p2_fused, p3_final, p4_final]


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) layer
    """
    def __init__(self, in_channels, out_channels, k=5):
        super(SPPF, self).__init__()
        c_ = in_channels // 2
        self.cv1 = CBS(in_channels, c_, 1, 1, 0)
        self.cv2 = CBS(c_ * 4, out_channels, 1, 1, 0)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


if __name__ == "__main__":
    # Test SPANet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test feature maps
    p2 = torch.randn(1, 256, 160, 160).to(device)
    p3 = torch.randn(1, 512, 80, 80).to(device)
    p4 = torch.randn(1, 1024, 40, 40).to(device)
    
    # Initialize SPANet
    spanet = SPANet([256, 512, 1024]).to(device)
    
    # Forward pass
    features = [p2, p3, p4]
    output_features = spanet(features)
    
    print("SPANet Test:")
    for i, feat in enumerate(output_features):
        print(f"P{i+2} output shape: {feat.shape}")
    
    print("SPANet test passed!")
