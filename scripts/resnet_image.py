import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

from resnet_train import create_resnet18, TrainConfig

# ===========================
# INFERENCE AND VISUALIZATION
# ===========================

class ResNet18Predictor:
    """Class for making predictions with trained ResNet18 and visualizing results"""
    
    def __init__(self, checkpoint_path, class_info_path=None):
        """
        Initialize predictor
        
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
        else:
            # Default classes
            self.classes = ['Pedestrian', 'Cyclist', 'Car']
            self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}
        
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
    
    def predict_image(self, image_path, return_probabilities=False):
        """
        Predict class for a single image
        
        Args:
            image_path: Path to image file
            return_probabilities: If True, return class probabilities
        
        Returns:
            prediction: Predicted class name
            confidence: Confidence score (0-1)
            probabilities: Class probabilities (if return_probabilities=True)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Get class name
        predicted_class = self.idx_to_class[predicted_idx.item()]
        confidence_score = confidence.item()
        
        if return_probabilities:
            probs_dict = {
                self.idx_to_class[i]: probabilities[0][i].item() 
                for i in range(self.num_classes)
            }
            return predicted_class, confidence_score, probs_dict
        
        return predicted_class, confidence_score
    
    def visualize_single_prediction(self, image_path, save_path=None, show=True):
        """
        Visualize prediction for a single image with probabilities
        
        Args:
            image_path: Path to image
            save_path: Path to save visualization (optional)
            show: Whether to display the plot
        """
        # Get prediction with probabilities
        pred_class, confidence, probs = self.predict_image(
            image_path, return_probabilities=True
        )
        
        # Load original image
        image = Image.open(image_path).convert('RGB')
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # ===== LEFT: Show image with prediction =====
        ax1.imshow(image)
        ax1.axis('off')
        
        # Add title with prediction
        title_color = 'green' if confidence > 0.7 else 'orange' if confidence > 0.5 else 'red'
        ax1.set_title(
            f'Prediction: {pred_class}\nConfidence: {confidence:.1%}',
            fontsize=16, fontweight='bold', color=title_color, pad=20
        )
        
        # ===== RIGHT: Show probability bar chart =====
        classes = list(probs.keys())
        probabilities = list(probs.values())
        
        # Color bars - highlight predicted class in green
        colors = ['#2ecc71' if cls == pred_class else '#95a5a6' for cls in classes]
        
        bars = ax2.barh(classes, probabilities, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Probability', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Class', fontsize=13, fontweight='bold')
        ax2.set_title('Class Probabilities', fontsize=14, fontweight='bold')
        ax2.set_xlim([0, 1])
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add probability values on bars
        for i, (cls, prob) in enumerate(zip(classes, probabilities)):
            ax2.text(prob + 0.02, i, f'{prob:.1%}', 
                    va='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_batch_predictions(self, image_paths, save_dir=None, num_cols=3):
        """
        Visualize predictions for multiple images in a grid
        
        Args:
            image_paths: List of image file paths
            save_dir: Directory to save visualizations
            num_cols: Number of columns in grid
        """
        num_images = len(image_paths)
        num_rows = (num_images + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 6*num_rows))
        
        # Handle single row case
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.reshape(num_rows, num_cols)
        
        for idx, image_path in enumerate(image_paths):
            row = idx // num_cols
            col = idx % num_cols
            ax = axes[row, col]
            
            # Get prediction
            pred_class, confidence, probs = self.predict_image(
                image_path, return_probabilities=True
            )
            
            # Load and display image
            image = Image.open(image_path).convert('RGB')
            ax.imshow(image)
            ax.axis('off')
            
            # Add title with prediction
            title_color = 'green' if confidence > 0.7 else 'orange' if confidence > 0.5 else 'red'
            ax.set_title(
                f'{pred_class}\n{confidence:.1%}',
                fontsize=12, fontweight='bold', color=title_color
            )
        
        # Hide empty subplots
        for idx in range(num_images, num_rows * num_cols):
            row = idx // num_cols
            col = idx % num_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "batch_predictions.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Batch visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_with_annotations(self, image_path, annotation_dict=None, save_path=None):
        """
        Visualize image with prediction and bounding boxes
        
        Args:
            image_path: Path to image
            annotation_dict: Dictionary with 'objects' containing bbox info
            save_path: Path to save visualization
        """
        # Get prediction
        pred_class, confidence, probs = self.predict_image(
            image_path, return_probabilities=True
        )
        
        # Load original image
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.imshow(image_array)
        
        # Draw bounding boxes if provided
        if annotation_dict and 'objects' in annotation_dict:
            for obj in annotation_dict['objects']:
                bbox = obj['bbox']  # [left, top, right, bottom]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                # Create rectangle
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), width, height,
                    linewidth=2, edgecolor='blue', facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label
                label = obj['class']
                ax.text(bbox[0], bbox[1]-5, label, 
                       fontsize=10, color='blue', fontweight='bold',
                       bbox=dict(facecolor='white', alpha=0.7))
        
        # Add prediction info
        info_text = f"Model Prediction: {pred_class} ({confidence:.1%})"
        ax.text(0.5, -0.05, info_text, transform=ax.transAxes,
               fontsize=14, ha='center', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Annotated visualization saved to {save_path}")
        
        plt.show()
    
    def create_comparison_report(self, image_paths, output_dir, num_per_row=4):
        """
        Create a detailed report with all predictions
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save report
            num_per_row: Number of images per row
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Make predictions for all images
        predictions = []
        print(f"\nMaking predictions for {len(image_paths)} images...")
        
        for image_path in image_paths:
            pred_class, confidence, probs = self.predict_image(
                image_path, return_probabilities=True
            )
            predictions.append({
                'image': image_path,
                'prediction': pred_class,
                'confidence': confidence,
                'probabilities': probs
            })
        
        # Create visualization grid
        num_images = len(image_paths)
        num_rows = (num_images + num_per_row - 1) // num_per_row
        
        fig, axes = plt.subplots(num_rows, num_per_row, figsize=(20, 5*num_rows))
        
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.reshape(num_rows, num_per_row)
        
        for idx, (image_path, pred_info) in enumerate(zip(image_paths, predictions)):
            row = idx // num_per_row
            col = idx % num_per_row
            ax = axes[row, col]
            
            # Load and display image
            image = Image.open(image_path).convert('RGB')
            ax.imshow(image)
            ax.axis('off')
            
            # Add prediction info
            pred_class = pred_info['prediction']
            confidence = pred_info['confidence']
            
            title_color = 'green' if confidence > 0.7 else 'orange' if confidence > 0.5 else 'red'
            ax.set_title(
                f'{pred_class}\n{confidence:.1%}',
                fontsize=11, fontweight='bold', color=title_color
            )
        
        # Hide empty subplots
        for idx in range(num_images, num_rows * num_per_row):
            row = idx // num_per_row
            col = idx % num_per_row
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(output_dir, "predictions_grid.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"✓ Grid saved to {fig_path}")
        plt.close()
        
        # Save detailed report
        report_path = os.path.join(output_dir, "predictions_report.txt")
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("RESNET18 PREDICTION REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Summary statistics
            total_correct_by_class = {cls: 0 for cls in self.classes}
            confidence_by_class = {cls: [] for cls in self.classes}
            
            f.write("INDIVIDUAL PREDICTIONS:\n")
            f.write("-"*70 + "\n")
            
            for i, pred_info in enumerate(predictions, 1):
                pred_class = pred_info['prediction']
                confidence = pred_info['confidence']
                image_name = os.path.basename(pred_info['image'])
                
                confidence_by_class[pred_class].append(confidence)
                
                f.write(f"\n{i}. {image_name}\n")
                f.write(f"   Prediction: {pred_class}\n")
                f.write(f"   Confidence: {confidence:.1%}\n")
                f.write(f"   Probabilities:\n")
                
                for cls, prob in pred_info['probabilities'].items():
                    f.write(f"     - {cls}: {prob:.1%}\n")
            
            # Class statistics
            f.write("\n" + "="*70 + "\n")
            f.write("CLASS STATISTICS:\n")
            f.write("="*70 + "\n\n")
            
            for cls in self.classes:
                if confidence_by_class[cls]:
                    avg_conf = np.mean(confidence_by_class[cls])
                    count = len(confidence_by_class[cls])
                    f.write(f"{cls}:\n")
                    f.write(f"  Count: {count}\n")
                    f.write(f"  Average Confidence: {avg_conf:.1%}\n")
                    f.write(f"  Min Confidence: {min(confidence_by_class[cls]):.1%}\n")
                    f.write(f"  Max Confidence: {max(confidence_by_class[cls]):.1%}\n\n")
        
        print(f"✓ Report saved to {report_path}")
        
        return predictions

# ===========================
# DEMO USAGE
# ===========================

def demo_single_image():
    """Demo: Visualize a single image with prediction"""
    
    # Paths
    checkpoint_path = "./resnet18_output/checkpoints/best_model.pth"
    class_info_path = "./preprocessed_vod/class_info.json"
    test_image = "data/View_of_Delft_dataset_PUBLIC/view_of_delft_PUBLIC/lidar/training/image_2/00001.jpg"
    output_dir = "./prediction_visualizations"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize predictor
    predictor = ResNet18Predictor(checkpoint_path, class_info_path)
    
    # Visualize single image
    print(f"\nVisualizing: {test_image}\n")
    save_path = os.path.join(output_dir, "single_prediction.png")
    predictor.visualize_single_prediction(test_image, save_path=save_path, show=True)

def demo_batch_images():
    """Demo: Visualize multiple images in a grid"""
    
    # Paths
    checkpoint_path = "./resnet18_output/checkpoints/best_model.pth"
    class_info_path = "./preprocessed_vod/class_info.json"
    image_dir = "data/View_of_Delft_dataset_PUBLIC/view_of_delft_PUBLIC/lidar/training/image_2"  # CHANGE THIS
    output_dir = "./prediction_visualizations"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize predictor
    predictor = ResNet18Predictor(checkpoint_path, class_info_path)
    
    # Get some test images
    image_files = sorted([os.path.join(image_dir, f) 
                         for f in os.listdir(image_dir)[:12]
                         if f.endswith('.png')])
    
    print(f"\nVisualizing {len(image_files)} images...\n")
    predictor.visualize_batch_predictions(image_files, save_dir=output_dir, num_cols=3)

def demo_detailed_report():
    """Demo: Create detailed prediction report"""
    
    # Paths
    checkpoint_path = "./resnet18_output/checkpoints/best_model.pth"
    class_info_path = "./preprocessed_vod/class_info.json"
    image_dir = "/path/to/View-of-Delft-Dataset/lidar/training/image_2"  # CHANGE THIS
    output_dir = "./prediction_report"
    
    # Initialize predictor
    predictor = ResNet18Predictor(checkpoint_path, class_info_path)
    
    # Get test images
    image_files = sorted([os.path.join(image_dir, f) 
                         for f in os.listdir(image_dir)[:20]
                         if f.endswith('.png')])
    
    print(f"\nCreating report for {len(image_files)} images...\n")
    predictions = predictor.create_comparison_report(image_files, output_dir, num_per_row=5)
    
    print(f"\n✓ Report and visualizations saved to {output_dir}")

if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("ResNet18 Inference & Visualization")
    print("="*70)
    
    # Choose which demo to run
    if len(sys.argv) > 1:
        demo_type = sys.argv[1]
    else:
        demo_type = "batch"  # Default to batch
    
    if demo_type == "single":
        demo_single_image()
    elif demo_type == "batch":
        demo_batch_images()
    elif demo_type == "report":
        demo_detailed_report()
    else:
        print(f"Unknown demo type: {demo_type}")
        print("Usage: python visualize_predictions.py [single|batch|report]")
