# LSDM: Landslide Detection Model

This repository contains the implementation of LSDM (Landslide Detection Model), an efficient end-to-end framework for UAV-based landslide detection using enhanced YOLOv8s architecture with Spatial Pyramid Dilated Convolution (SPD-Conv) and Spatial Path Aggregation Network (SPANet).

## Features

- **Enhanced YOLOv8s Architecture**: Modified backbone with SPD-Conv modules for better small object detection
- **SPD-Conv Module**: Space-to-depth convolution that preserves fine-grained features while reducing information loss
- **SPANet**: Enhanced feature fusion network with shallow feature integration (P2 layer)
- **Real-time Performance**: Optimized for UAV deployment and edge devices
- **Lightweight Design**: ~11M parameters for efficient inference

## Model Architecture

### Key Components

1. **SPD-Conv Module** (`spd_conv.py`):
   - Replaces traditional strided convolutions
   - Uses space-to-depth transformation to preserve spatial information
   - Maintains discriminative features for small landslide detection

2. **SPANet** (`spanet.py`):
   - Enhanced Path Aggregation Network
   - Integrates shallow P2 features for better multi-scale fusion
   - Improves contextual awareness across diverse terrains

3. **LSDM Model** (`lsdm_model.py`):
   - Complete end-to-end framework
   - Enhanced YOLOv8s backbone with SPD-Conv integration
   - SPANet neck for improved feature fusion
   - YOLOv8 detection head with DFL (Distribution Focal Loss)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/LSDM-Landslide-Detection.git
cd LSDM-Landslide-Detection

# Install dependencies
pip install -r requirements.txt
```

## Dataset Format

The model expects YOLO format annotations:

```
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── val/
│       ├── val1.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img1.txt
    │   ├── img2.txt
    │   └── ...
    └── val/
        ├── val1.txt
        └── ...
```

Label format (YOLO): `class_id x_center y_center width height` (normalized coordinates)

## Training

```bash
python train.py \
    --train-images ./dataset/images/train \
    --train-labels ./dataset/labels/train \
    --val-images ./dataset/images/val \
    --val-labels ./dataset/labels/val \
    --epochs 150 \
    --batch-size 16 \
    --img-size 640 \
    --output-dir ./runs/train
```

### Training Parameters

- `--epochs`: Number of training epochs (default: 150)
- `--batch-size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 0.001)
- `--img-size`: Input image size (default: 640)
- `--num-classes`: Number of classes (default: 1 for landslide)

## Inference

### Single Image

```bash
python inference.py \
    --model ./runs/train/best_model.pth \
    --source ./test_image.jpg \
    --output ./runs/inference \
    --conf-thresh 0.25
```

### Directory of Images

```bash
python inference.py \
    --model ./runs/train/best_model.pth \
    --source ./test_images/ \
    --output ./runs/inference \
    --conf-thresh 0.25
```


## Usage Examples

### Quick Test

```python
from lsdm_model import create_lsdm_model
import torch

# Create model
model = create_lsdm_model(num_classes=1)

# Test forward pass
x = torch.randn(1, 3, 640, 640)
predictions = model(x)

print(f"Model loaded successfully!")
print(f"Prediction shapes: {[p.shape for p in predictions]}")
```

### Custom Training Loop

```python
from lsdm_model import create_lsdm_model
from train import LandslideDataset, YOLOLoss
import torch

# Initialize model, dataset, and loss
model = create_lsdm_model(num_classes=1)
dataset = LandslideDataset('./images', './labels')
criterion = YOLOLoss(num_classes=1)

# Your training loop here...
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{khan2025lsdm,
  title={UAV-Based Landslide Detection Using Context-Aware Framework for Complex Terrain Mapping},
  author={Khan, Asma and Nasr, Kashif and Khan, Samee Ullah},
  journal={Submitted to Elsevier},
  year={2025}
}
```

## Requirements

- Python 3.7+
- PyTorch 1.11.0+
- OpenCV 4.5.0+
- NumPy 1.21.0+
- See `requirements.txt` for full dependencies

## Hardware Requirements

- **Training**: NVIDIA GPU with 8GB+ VRAM recommended
- **Inference**: CPU or GPU (optimized for edge devices)
- **Memory**: 4GB+ RAM for training, 2GB+ for inference

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on YOLOv8 architecture by Ultralytics
- SPD-Conv implementation inspired by the original paper
- Dataset collected using DJI Mavic Air 2 in California landslide-prone regions



## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or image size
2. **Model Loading Error**: Ensure model path is correct and model file exists
3. **Dataset Loading**: Check image and label paths match expected format

### Performance Tips

1. Use GPU for training and inference when available
2. Adjust batch size based on available memory
3. Use mixed precision training for faster convergence
4. Consider data augmentation for better generalization
