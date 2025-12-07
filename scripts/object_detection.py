import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import json
import os
import numpy as np
from pathlib import Path
import cv2

from resnet_train import create_resnet18, TrainConfig

# ===========================
# OBJECT DETECTION STYLE VISUALIZATION
# ===========================

class ObjectDetectionVisualizer:
    """Visualize predictions with bounding boxes drawn directly on images"""
    
    def __init__(self, checkpoint_path, class_info_path=None):
        """
        Initialize predictor for object detection visualization
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            class_info_path: Path to class_info.json from preprocessing
        """
        self.device = torch.device("mps" if torch.backends.mps.is_available() 
                                  else "cuda" if torch.cuda.is_available() 
                                  else "cpu")
        print(f"Using device: {self.device}")
        
        # Load class information
        if class_info_path and os.path.exists(class_info_path):
            with open(class_info_path, 'r') as f:
                class_info = json.load(f)
            self.classes = class_info['classes']
            self.idx_to_class = {int(k): v for k, v in class_info['idx_to_class'].items()}
            self.class_to_idx = {v: int(k) for k, v in class_info['idx_to_class'].items()}
        else:
            # Default classes
            self.classes = ['Pedestrian', 'Cyclist', 'Car']
            self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.num_classes = len(self.classes)
        
        # Create model
        self.model = create_resnet18(
            num_classes=self.num_classes,
            pretrained=False,
            device=self.device
        )
        
        # Load trained weights
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Classes: {self.classes}\n")
        
        # Define transforms (same as validation)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Color palette for each class (BGR format for OpenCV)
        self.colors_bgr = {
            'Pedestrian': (52, 76, 231),   # Red
            'Cyclist': (219, 152, 52),      # Blue
            'Car': (113, 204, 46),          # Green
        }
        
        # RGB format for PIL
        self.colors_rgb = {
            'Pedestrian': (231, 76, 52),    # Red
            'Cyclist': (52, 152, 219),      # Blue
            'Car': (46, 204, 113),          # Green
        }
    
    def predict_cropped_object(self, image, bbox):
        """
        Predict class for a cropped region of the image
        
        Args:
            image: PIL Image
            bbox: Bounding box [left, top, right, bottom]
        
        Returns:
            predicted_class: Predicted class name
            confidence: Confidence score
            probabilities: Dictionary of class probabilities
        """
        # Crop the bounding box region
        left, top, right, bottom = bbox
        cropped = image.crop((left, top, right, bottom))
        
        # Preprocess
        input_tensor = self.transform(cropped).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class = self.idx_to_class[predicted_idx.item()]
        confidence_score = confidence.item()
        
        probs_dict = {
            self.idx_to_class[i]: probabilities[0][i].item() 
            for i in range(self.num_classes)
        }
        
        return predicted_class, confidence_score, probs_dict
    
    def draw_bounding_boxes_pil(self, image_path, annotation_dict, 
                                save_path, show_ground_truth=True,
                                box_thickness=3, font_size=20):
        """
        Draw bounding boxes directly on image copy using PIL
        
        Args:
            image_path: Path to image
            annotation_dict: Dictionary with 'objects' containing bbox and class info
            save_path: Path to save output image
            show_ground_truth: Whether to show ground truth comparison
            box_thickness: Thickness of bounding box lines
            font_size: Size of text labels
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size - 4)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
                font_small = ImageFont.truetype("arial.ttf", font_size - 4)
            except:
                font = ImageFont.load_default()
                font_small = ImageFont.load_default()
        
        if 'objects' not in annotation_dict or len(annotation_dict['objects']) == 0:
            # Add "No objects" text
            draw.text((10, 10), "No objects detected", fill=(255, 0, 0), font=font)
            image.save(save_path)
            print(f"✓ Saved (no objects): {save_path}")
            return []
        
        # Process each object
        predictions_summary = []
        
        for idx, obj in enumerate(annotation_dict['objects']):
            bbox = obj['bbox']  # [left, top, right, bottom]
            ground_truth_class = obj['class']
            
            # Get prediction for this object
            pred_class, confidence, probs = self.predict_cropped_object(image, bbox)
            
            predictions_summary.append({
                'ground_truth': ground_truth_class,
                'prediction': pred_class,
                'confidence': confidence,
                'correct': pred_class == ground_truth_class
            })
            
            # Bounding box coordinates
            left, top, right, bottom = [int(x) for x in bbox]
            
            # Choose color based on prediction
            color = self.colors_rgb.get(pred_class, (149, 165, 166))
            
            # Determine if prediction is correct
            is_correct = pred_class == ground_truth_class
            
            # Draw bounding box (thicker if correct)
            thickness = box_thickness if is_correct else box_thickness - 1
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=color,
                    width=1
                )
            
            # If incorrect, add dashed effect by drawing shorter segments
            if not is_correct:
                dash_length = 10
                gap_length = 5
                
                # Top edge
                x = left
                while x < right:
                    draw.line([(x, top), (min(x + dash_length, right), top)], 
                             fill=color, width=2)
                    x += dash_length + gap_length
                
                # Bottom edge
                x = left
                while x < right:
                    draw.line([(x, bottom), (min(x + dash_length, right), bottom)], 
                             fill=color, width=2)
                    x += dash_length + gap_length
            
            # Create label text
            if show_ground_truth:
                if is_correct:
                    label_text = f"{pred_class} {confidence:.0%}"
                    status_icon = "✓"
                else:
                    label_text = f"P:{pred_class} {confidence:.0%}"
                    status_icon = f"GT:{ground_truth_class}"
            else:
                label_text = f"{pred_class} {confidence:.0%}"
                status_icon = ""
            
            # Calculate label position (above bbox)
            label_y = top - font_size - 10
            if label_y < 0:  # If too close to top, put inside
                label_y = top + 5
            
            # Get text bounding box for background
            bbox_text = draw.textbbox((left, label_y), label_text, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            
            # Draw label background
            padding = 4
            draw.rectangle(
                [left - padding, label_y - padding, 
                 left + text_width + padding, label_y + text_height + padding],
                fill=color,
                outline=(255, 255, 255),
                width=2
            )
            
            # Draw text
            draw.text((left, label_y), label_text, fill=(255, 255, 255), font=font)
            
            # Add status icon if needed
            if status_icon:
                icon_y = label_y + text_height + 2
                if not is_correct:
                    draw.text((left, icon_y), status_icon, fill=(255, 255, 255), font=font_small)
        
        # Add overall statistics at bottom of image
        total = len(predictions_summary)
        correct = sum(1 for p in predictions_summary if p['correct'])
        accuracy = correct / total if total > 0 else 0
        
        stats_text = f"Objects: {total} | Correct: {correct}/{total} ({accuracy:.0%})"
        
        # Draw stats background at bottom
        img_width, img_height = image.size
        stats_y = img_height - font_size - 20
        
        bbox_stats = draw.textbbox((10, stats_y), stats_text, font=font)
        stats_width = bbox_stats[2] - bbox_stats[0]
        stats_height = bbox_stats[3] - bbox_stats[1]
        
        draw.rectangle(
            [5, stats_y - 5, stats_width + 15, stats_y + stats_height + 5],
            fill=(0, 0, 0),
            outline=(255, 255, 255),
            width=2
        )
        
        draw.text((10, stats_y), stats_text, fill=(255, 255, 255), font=font)
        
        # Save image
        image.save(save_path, quality=95)
        print(f"✓ Saved: {save_path}")
        
        return predictions_summary
    
    def draw_bounding_boxes_opencv(self, image_path, annotation_dict, 
                                   save_path, show_ground_truth=True,
                                   box_thickness=3):
        """
        Draw bounding boxes directly on image copy using OpenCV
        (Alternative to PIL method - produces slightly different style)
        
        Args:
            image_path: Path to image
            annotation_dict: Dictionary with 'objects' containing bbox and class info
            save_path: Path to save output image
            show_ground_truth: Whether to show ground truth comparison
            box_thickness: Thickness of bounding box lines
        """
        # Load image with OpenCV
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL for prediction (needed for cropping)
        pil_image = Image.fromarray(image)
        
        # Convert back to BGR for OpenCV drawing
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if 'objects' not in annotation_dict or len(annotation_dict['objects']) == 0:
            cv2.putText(image, "No objects detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(save_path, image)
            print(f"✓ Saved (no objects): {save_path}")
            return []
        
        # Process each object
        predictions_summary = []
        
        for idx, obj in enumerate(annotation_dict['objects']):
            bbox = obj['bbox']  # [left, top, right, bottom]
            ground_truth_class = obj['class']
            
            # Get prediction for this object
            pred_class, confidence, probs = self.predict_cropped_object(pil_image, bbox)
            
            predictions_summary.append({
                'ground_truth': ground_truth_class,
                'prediction': pred_class,
                'confidence': confidence,
                'correct': pred_class == ground_truth_class
            })
            
            # Bounding box coordinates
            left, top, right, bottom = [int(x) for x in bbox]
            
            # Choose color based on prediction (BGR for OpenCV)
            color = self.colors_bgr.get(pred_class, (166, 165, 149))
            
            # Determine if prediction is correct
            is_correct = pred_class == ground_truth_class
            
            # Draw bounding box
            if is_correct:
                # Solid line for correct
                cv2.rectangle(image, (left, top), (right, bottom), color, box_thickness)
            else:
                # Dashed line for incorrect (approximate with line segments)
                dash_length = 10
                
                # Top
                for x in range(left, right, dash_length * 2):
                    cv2.line(image, (x, top), (min(x + dash_length, right), top), color, 2)
                # Bottom
                for x in range(left, right, dash_length * 2):
                    cv2.line(image, (x, bottom), (min(x + dash_length, right), bottom), color, 2)
                # Left
                for y in range(top, bottom, dash_length * 2):
                    cv2.line(image, (left, y), (left, min(y + dash_length, bottom)), color, 2)
                # Right
                for y in range(top, bottom, dash_length * 2):
                    cv2.line(image, (right, y), (right, min(y + dash_length, bottom)), color, 2)
            
            # Create label text
            if show_ground_truth:
                if is_correct:
                    label_text = f"{pred_class} {confidence:.0%}"
                else:
                    label_text = f"P:{pred_class} {confidence:.0%} GT:{ground_truth_class}"
            else:
                label_text = f"{pred_class} {confidence:.0%}"
            
            # Calculate label position
            label_y = top - 10
            if label_y < 25:
                label_y = top + 25
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw label background
            cv2.rectangle(
                image,
                (left - 2, label_y - text_height - 8),
                (left + text_width + 2, label_y + 2),
                color,
                -1
            )
            
            # Draw white border around label
            cv2.rectangle(
                image,
                (left - 2, label_y - text_height - 8),
                (left + text_width + 2, label_y + 2),
                (255, 255, 255),
                2
            )
            
            # Draw text
            cv2.putText(
                image, label_text,
                (left, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        # Add overall statistics
        total = len(predictions_summary)
        correct = sum(1 for p in predictions_summary if p['correct'])
        accuracy = correct / total if total > 0 else 0
        
        stats_text = f"Objects: {total} | Correct: {correct}/{total} ({accuracy:.0%})"
        
        # Draw stats at bottom
        img_height = image.shape[0]
        (stats_width, stats_height), _ = cv2.getTextSize(
            stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        
        stats_y = img_height - 20
        cv2.rectangle(image, (5, stats_y - stats_height - 10), 
                     (stats_width + 15, stats_y + 5), (0, 0, 0), -1)
        cv2.rectangle(image, (5, stats_y - stats_height - 10), 
                     (stats_width + 15, stats_y + 5), (255, 255, 255), 2)
        cv2.putText(image, stats_text, (10, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Save image
        cv2.imwrite(save_path, image)
        print(f"✓ Saved: {save_path}")
        
        return predictions_summary
    
    def process_batch(self, image_annotation_pairs, output_dir, 
                     method='pil', show_ground_truth=True):
        """
        Process multiple images and save with bounding boxes
        
        Args:
            image_annotation_pairs: List of (image_path, annotation_dict) tuples
            output_dir: Directory to save visualizations
            method: 'pil' or 'opencv' for drawing method
            show_ground_truth: Whether to show ground truth labels
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_predictions = []
        
        print(f"\nProcessing {len(image_annotation_pairs)} images...\n")
        
        for idx, (image_path, annotation) in enumerate(image_annotation_pairs, 1):
            image_name = os.path.basename(image_path)
            save_path = os.path.join(output_dir, f"detection_{idx:03d}_{image_name}")
            
            print(f"[{idx}/{len(image_annotation_pairs)}] {image_name}...")
            
            if method == 'opencv':
                predictions = self.draw_bounding_boxes_opencv(
                    image_path, annotation, save_path, show_ground_truth
                )
            else:  # PIL (default)
                predictions = self.draw_bounding_boxes_pil(
                    image_path, annotation, save_path, show_ground_truth
                )
            
            all_predictions.append({
                'image': image_name,
                'predictions': predictions
            })
        
        # Create summary report
        self._create_summary_report(all_predictions, output_dir)
        
        print(f"\n✓ All visualizations saved to {output_dir}")
    
    def _create_summary_report(self, all_predictions, output_dir):
        """Create a summary report of all predictions"""
        
        report_path = os.path.join(output_dir, "detection_summary.txt")
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("OBJECT DETECTION SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            total_objects = 0
            total_correct = 0
            class_stats = {cls: {'total': 0, 'correct': 0} for cls in self.classes}
            
            for img_data in all_predictions:
                f.write(f"\nImage: {img_data['image']}\n")
                f.write("-" * 80 + "\n")
                
                for obj_pred in img_data['predictions']:
                    total_objects += 1
                    gt_class = obj_pred['ground_truth']
                    pred_class = obj_pred['prediction']
                    confidence = obj_pred['confidence']
                    is_correct = obj_pred['correct']
                    
                    if is_correct:
                        total_correct += 1
                    
                    class_stats[gt_class]['total'] += 1
                    if is_correct:
                        class_stats[gt_class]['correct'] += 1
                    
                    status = "✓" if is_correct else "✗"
                    f.write(f"  {status} GT: {gt_class:12} | Pred: {pred_class:12} | Conf: {confidence:.1%}\n")
            
            # Overall statistics
            f.write("\n" + "="*80 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("="*80 + "\n\n")
            
            overall_acc = total_correct / total_objects if total_objects > 0 else 0
            f.write(f"Total Objects: {total_objects}\n")
            f.write(f"Correct Predictions: {total_correct}\n")
            f.write(f"Overall Accuracy: {overall_acc:.2%}\n\n")
            
            f.write("Per-Class Performance:\n")
            f.write("-" * 80 + "\n")
            
            for cls in self.classes:
                stats = class_stats[cls]
                if stats['total'] > 0:
                    acc = stats['correct'] / stats['total']
                    f.write(f"  {cls:12}: {stats['correct']:3}/{stats['total']:3} ({acc:.1%})\n")
        
        print(f"✓ Summary report saved to {report_path}")


# ===========================
# MAIN DEMO
# ===========================

def visualize_from_test_set(num_images=10, method='pil'):
    """
    Visualize predictions with bboxes from the test set
    
    Args:
        num_images: Number of images to process
        method: 'pil' or 'opencv' for drawing method
    """
    
    import json
    
    # Paths
    checkpoint_path = "./resnet18_output/checkpoints/best_model.pth"
    class_info_path = "./preprocessed_vod/class_info.json"
    test_annotations_path = "./preprocessed_vod/test_annotations.json"
    output_dir = "./detection_visualizations"
    image_dir = "data/View_of_Delft_dataset_PUBLIC/view_of_delft_PUBLIC/lidar/training/image_2/00001.jpg"

    if not os.path.exists(image_dir):
        print(f"\nERROR: Directory not found: {image_dir}")
        return
    
    # Load test annotations
    with open(test_annotations_path, 'r') as f:
        test_data = json.load(f)
    
    # Prepare image-annotation pairs
    pairs = []
    for i in range(min(num_images, len(test_data['images']))):
        image_name = test_data['images'][i]
        annotation = test_data['annotations'][i]
        image_path = os.path.join(image_dir, image_name)
        
        if os.path.exists(image_path):
            pairs.append((image_path, annotation))
    
    print(f"\nLoaded {len(pairs)} images with annotations")
    
    # Initialize visualizer
    visualizer = ObjectDetectionVisualizer(checkpoint_path, class_info_path)
    
    # Process images
    visualizer.process_batch(pairs, output_dir, method=method, show_ground_truth=True)
    
    print(f"\n{'='*80}")
    print(f"✓ Detection visualizations saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    import sys
    
    print("="*80)
    print("ResNet18 Object Detection Visualization")
    print("Bounding boxes drawn directly on image copies")
    print("="*80)
    print()
    
    # Ask how many images to process
    try:
        num_images = int(input("How many images to visualize? (default 10): ") or "10")
    except ValueError:
        num_images = 10
    
    # Ask which drawing method
    print("\nDrawing method:")
    print("  1. PIL (clean, smooth)")
    print("  2. OpenCV (alternative style)")
    method_choice = input("Choose (1 or 2, default 1): ").strip() or "1"
    method = 'opencv' if method_choice == '2' else 'pil'
    
    visualize_from_test_set(num_images=num_images, method=method)
