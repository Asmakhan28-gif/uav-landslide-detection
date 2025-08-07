"""
LSDM (Landslide Detection Model) Implementation
Complete end-to-end landslide detection framework based on enhanced YOLOv8s
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from spd_conv import SPDConv, CBS
from spanet import SPANet, C2F, SPPF


class Detect(nn.Module):
    """YOLOv8 detection head"""
    
    def __init__(self, num_classes=1, channels=[256, 512, 1024]):
        super(Detect, self).__init__()
        self.nc = num_classes  # number of classes (landslide = 1)
        self.nl = len(channels)  # number of detection layers
        self.reg_max = 16  # DFL channels
        self.no = num_classes + self.reg_max * 4  # number of outputs per anchor
        
        # Detection heads for each scale
        self.cv2 = nn.ModuleList([
            nn.Sequential(
                CBS(ch, ch, 3, 1, 1),
                CBS(ch, ch, 3, 1, 1),
                nn.Conv2d(ch, 4 * self.reg_max, 1)
            ) for ch in channels
        ])
        
        self.cv3 = nn.ModuleList([
            nn.Sequential(
                CBS(ch, ch, 3, 1, 1),
                CBS(ch, ch, 3, 1, 1),
                nn.Conv2d(ch, self.nc, 1)
            ) for ch in channels
        ])
        
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        
    def forward(self, x):
        """Forward pass through detection heads"""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        return x


class DFL(nn.Module):
    """Distribution Focal Loss (DFL) for bounding box regression"""
    
    def __init__(self, c1=16):
        super(DFL, self).__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class LSGDBackbone(nn.Module):
    """
    Enhanced YOLOv8s backbone with SPD-Conv modules
    """
    
    def __init__(self, in_channels=3, base_channels=64):
        super(LSGDBackbone, self).__init__()
        
        # Stem
        self.stem = CBS(in_channels, base_channels, 3, 2, 1)  # 1/2
        
        # Stage 1
        self.stage1 = nn.Sequential(
            CBS(base_channels, base_channels * 2, 3, 2, 1),  # 1/4
            C2F(base_channels * 2, base_channels * 2, 1)
        )
        
        # Stage 2 - Replace with SPD-Conv
        self.stage2 = nn.Sequential(
            SPDConv(base_channels * 2, base_channels * 4, 3, 1, 1, scale=2),  # 1/8, SPD-Conv at layer 3
            C2F(base_channels * 4, base_channels * 4, 2)
        )
        
        # Stage 3 - Replace with SPD-Conv  
        self.stage3 = nn.Sequential(
            SPDConv(base_channels * 4, base_channels * 8, 3, 1, 1, scale=2),  # 1/16, SPD-Conv at layer 6
            C2F(base_channels * 8, base_channels * 8, 2)
        )
        
        # Stage 4 - Replace with SPD-Conv
        self.stage4 = nn.Sequential(
            SPDConv(base_channels * 8, base_channels * 16, 3, 1, 1, scale=2),  # 1/32, SPD-Conv at layer 9
            C2F(base_channels * 16, base_channels * 16, 1),
            SPPF(base_channels * 16, base_channels * 16)
        )
        
    def forward(self, x):
        # Extract multi-scale features
        x = self.stem(x)
        
        # Stage outputs
        p1 = self.stage1(x)      # 1/4, 128 channels
        p2 = self.stage2(p1)     # 1/8, 256 channels  
        p3 = self.stage3(p2)     # 1/16, 512 channels
        p4 = self.stage4(p3)     # 1/32, 1024 channels
        
        return [p2, p3, p4]  # Return P2, P3, P4 for SPANet


class LSDM(nn.Module):
    """
    Complete LSDM (Landslide Detection Model) framework
    Enhanced YOLOv8s with SPD-Conv and SPANet
    """
    
    def __init__(self, num_classes=1, input_channels=3):
        super(LSDM, self).__init__()
        
        self.nc = num_classes
        
        # Enhanced backbone with SPD-Conv
        self.backbone = LSGDBackbone(input_channels)
        
        # Enhanced neck with SPANet
        self.neck = SPANet([256, 512, 1024])
        
        # Detection head
        self.head = Detect(num_classes, [256, 512, 1024])
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x):
        # Backbone feature extraction
        features = self.backbone(x)
        
        # Neck feature fusion
        enhanced_features = self.neck(features)
        
        # Detection head
        predictions = self.head(enhanced_features)
        
        return predictions
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        }


def create_lsdm_model(num_classes=1, pretrained=False):
    """
    Create LSDM model
    
    Args:
        num_classes: Number of classes (default: 1 for landslide detection)
        pretrained: Whether to load pretrained weights
    
    Returns:
        LSDM model
    """
    model = LSDM(num_classes=num_classes)
    
    if pretrained:
        # Load pretrained weights if available
        # This would be implemented when pretrained weights are available
        print("Pretrained weights not implemented yet")
    
    return model


if __name__ == "__main__":
    # Test LSDM model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_lsdm_model(num_classes=1).to(device)
    
    # Test input
    x = torch.randn(1, 3, 640, 640).to(device)
    
    # Forward pass
    with torch.no_grad():
        predictions = model(x)
    
    # Print results
    print("LSDM Model Test:")
    print(f"Input shape: {x.shape}")
    print(f"Number of detection scales: {len(predictions)}")
    
    for i, pred in enumerate(predictions):
        print(f"Scale {i+1} prediction shape: {pred.shape}")
    
    # Model information
    info = model.get_model_info()
    print(f"\nModel Information:")
    print(f"Total parameters: {info['total_parameters']:,}")
    print(f"Trainable parameters: {info['trainable_parameters']:,}")
    print(f"Model size: {info['model_size_mb']:.2f} MB")
    
    print("\nLSDM model test passed!")
