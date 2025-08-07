"""
Inference script for LSDM (Landslide Detection Model)
"""

import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
import time

from lsdm_model import create_lsdm_model


class LSDMInference:
    """LSDM inference class"""
    
    def __init__(self, model_path, device='auto', img_size=640, conf_thresh=0.25):
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = create_lsdm_model(num_classes=1)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
    
    def preprocess(self, image):
        """Preprocess image for inference"""
        # Resize image
        h, w = image.shape[:2]
        image_resized = cv2.resize(image, (self.img_size, self.img_size))
        
        # Normalize
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device), (h, w)
    
    def postprocess(self, predictions, original_shape):
        """Post-process predictions"""
        # Simplified post-processing
        # In a complete implementation, this would include:
        # - NMS (Non-Maximum Suppression)
        # - Coordinate transformation back to original image size
        # - Confidence filtering
        
        detections = []
        h_orig, w_orig = original_shape
        
        # Process each scale
        for pred in predictions:
            batch_size, channels, height, width = pred.shape
            
            # Reshape predictions
            pred = pred.view(batch_size, -1, channels // (height * width))
            
            # Extract boxes and confidences (simplified)
            for i in range(pred.shape[1]):
                # Dummy detection for demonstration
                # Replace with proper post-processing
                conf = torch.sigmoid(pred[0, i, -1]).item()
                
                if conf > self.conf_thresh:
                    # Dummy coordinates (replace with actual box decoding)
                    x1 = int(np.random.uniform(0, w_orig * 0.3))
                    y1 = int(np.random.uniform(0, h_orig * 0.3))
                    x2 = int(np.random.uniform(w_orig * 0.7, w_orig))
                    y2 = int(np.random.uniform(h_orig * 0.7, h_orig))
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class': 0  # Landslide class
                    })
        
        return detections[:3]  # Return maximum 3 detections for demo
    
    def predict(self, image):
        """Run inference on image"""
        with torch.no_grad():
            # Preprocess
            image_tensor, original_shape = self.preprocess(image)
            
            # Inference
            start_time = time.time()
            predictions = self.model(image_tensor)
            inference_time = time.time() - start_time
            
            # Post-process
            detections = self.postprocess(predictions, original_shape)
            
            return detections, inference_time
    
    def predict_image_path(self, image_path):
        """Run inference on image from path"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.predict(image)


def draw_detections(image, detections):
    """Draw detection results on image"""
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"Landslide {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image


def main():
    parser = argparse.ArgumentParser(description='LSDM Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--source', type=str, required=True,
                       help='Source image or directory')
    parser.add_argument('--output', type=str, default='./runs/inference',
                       help='Output directory')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--conf-thresh', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda:0, etc.)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize inference
    inference = LSDMInference(
        model_path=args.model,
        device=args.device,
        img_size=args.img_size,
        conf_thresh=args.conf_thresh
    )
    
    # Process source
    source_path = Path(args.source)
    
    if source_path.is_file():
        # Single image
        image_files = [source_path]
    elif source_path.is_dir():
        # Directory of images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(source_path.glob(ext))
    else:
        raise ValueError(f"Source path not found: {args.source}")
    
    # Process images
    total_time = 0
    for img_path in image_files:
        print(f"Processing: {img_path.name}")
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Could not load image: {img_path}")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference
        detections, inference_time = inference.predict(image_rgb)
        total_time += inference_time
        
        print(f"  Detections: {len(detections)}")
        print(f"  Inference time: {inference_time:.3f}s")
        
        # Draw results
        image_result = draw_detections(image.copy(), detections)
        
        # Save result
        output_path = output_dir / f"result_{img_path.name}"
        cv2.imwrite(str(output_path), image_result)
        print(f"  Saved: {output_path}")
    
    # Print summary
    if image_files:
        avg_time = total_time / len(image_files)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        print(f"\nProcessed {len(image_files)} images")
        print(f"Average inference time: {avg_time:.3f}s")
        print(f"Average FPS: {fps:.1f}")


if __name__ == '__main__':
    main()
