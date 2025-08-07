# LSDM: Landslide Detection Model

This repository contains the implementation of **LSDM (Landslide Detection Model)** — an efficient end-to-end framework for UAV-based landslide detection using an enhanced YOLOv8s architecture with **Spatial Pyramid Dilated Convolution (SPD-Conv)** and **Spatial Path Aggregation Network (SPANet)**.

---

## 🚀 Features

- 🔍 **Enhanced YOLOv8s Architecture** with SPD-Conv backbone
- 🧠 **SPD-Conv Module**: Preserves fine-grained spatial features
- 🕸️ **SPANet**: Shallow-deep feature fusion via P2-P4 layers
- ⚡ **Real-time Inference**: Optimized for UAV and edge devices
- 📦 **Lightweight**: ~11M parameters, efficient on GPU/CPU

---

## 🏗️ Model Architecture

### Key Components

1. **SPD-Conv Module** (`spd_conv.py`):
   - Replaces strided convolutions
   - Uses space-to-depth transformation
   - Retains high-resolution features for small landslide detection

2. **SPANet** (`spanet.py`):
   - Enhanced Path Aggregation Network
   - Combines shallow P2 and deeper P3/P4 layers
   - Improves context awareness across terrain scales

3. **LSDM Model** (`lsdm_model.py`):
   - YOLOv8-inspired end-to-end detection model
   - Combines SPD-Conv + SPANet + DFL detection head

---

## ⚠️ Implementation Notice

> **Important Note:**  
> This repository includes placeholder logic in some components to demonstrate the architecture. These include:

- 🔧 **Loss Function**: The current `YOLOLoss` class is a simplified placeholder and **does not implement full YOLO-style loss** (no CIoU/DFL/objectness).
- 🔍 **Inference Postprocessing**: The `inference_script.py` uses **dummy bounding box coordinates** (randomly generated) instead of actual box decoding + NMS.

✅ The **LSDM architecture itself is fully implemented and trainable**, and this repo serves as a **baseline** for researchers to build upon.

📌 For practical deployment or evaluation, users should:
- Implement proper loss functions with target assignment
- Decode predicted boxes and apply Non-Maximum Suppression (NMS)
- Evaluate with mAP, IoU, recall, etc.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/LSDM-Landslide-Detection.git
cd LSDM-Landslide-Detection

# Install dependencies
pip install -r requirements.txt
```

---

## 📁 Dataset Format

The model expects **YOLO format annotations**:

```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Each label file (`.txt`) contains:
```
class_id x_center y_center width height
```
(normalized coordinates)

---

## 🏋️ Training

```bash
python train_lsdm.py   --train-images ./dataset/images/train   --train-labels ./dataset/labels/train   --val-images ./dataset/images/val   --val-labels ./dataset/labels/val   --epochs 150   --batch-size 16   --img-size 640   --output-dir ./runs/train
```

### Training Parameters

| Argument       | Description                      | Default |
|----------------|----------------------------------|---------|
| `--epochs`     | Number of training epochs        | 150     |
| `--batch-size` | Batch size                       | 16      |
| `--img-size`   | Input image size                 | 640     |
| `--num-classes`| Number of object classes         | 1       |

---

## 🔎 Inference

### For a Single Image

```bash
python inference_script.py   --model ./runs/train/best_model.pth   --source ./test_image.jpg   --output ./runs/inference   --conf-thresh 0.25
```

### For a Folder of Images

```bash
python inference_script.py   --model ./runs/train/best_model.pth   --source ./test_images/   --output ./runs/inference   --conf-thresh 0.25
```

---

## 🧪 Usage Examples

### Quick Model Test

```python
from lsdm_model import create_lsdm_model
import torch

model = create_lsdm_model(num_classes=1)
x = torch.randn(1, 3, 640, 640)
predictions = model(x)

print("Prediction shapes:", [p.shape for p in predictions])
```

### Custom Training Setup

```python
from lsdm_model import create_lsdm_model
from train_lsdm import LandslideDataset, YOLOLoss

model = create_lsdm_model(num_classes=1)
dataset = LandslideDataset('./images', './labels')
criterion = YOLOLoss(num_classes=1)
```

---

## 📦 Requirements

- Python 3.7+
- PyTorch 1.11.0+
- OpenCV 4.5.0+
- NumPy 1.21.0+
- See `requirements.txt` for complete list

---

## 🖥️ Hardware Requirements

| Task       | Recommendation                     |
|------------|------------------------------------|
| Training   | NVIDIA GPU (8GB+ VRAM)             |
| Inference  | CPU or GPU (Optimized for edge AI) |
| RAM        | 4GB+ for training, 2GB+ for inference |

---

## 🤝 Contributing

1. Fork the repo
2. Create a branch (`git checkout -b feature/XYZ`)
3. Commit your changes
4. Push and create a pull request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

---

## 🙏 Acknowledgments

- Architecture inspired by **YOLOv8 (Ultralytics)**
- SPD-Conv design based on academic literature
- UAV dataset collected using **DJI Mavic Air 2** in landslide-prone regions

---

## 🛠 Troubleshooting

### Common Issues

| Problem                     | Solution                                  |
|----------------------------|-------------------------------------------|
| CUDA Out of Memory         | Reduce batch size or image resolution     |
| Dataset not loading        | Check folder paths and label formatting   |
| Model checkpoint not found | Verify path to `.pth` file in inference   |

---

## 💡 Performance Tips

- Use GPU + mixed precision (`torch.cuda.amp`) for faster training
- Add data augmentation (Albumentations) to generalize better
- Evaluate using mAP50/mAP50-95 for best model selection

---
