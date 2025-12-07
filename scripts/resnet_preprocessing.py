import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import cv2

# ===========================
# CONFIGURATION
# ===========================

class Config:
    """Configuration for View of Delft dataset preprocessing"""
    
    # Dataset paths - MODIFY THESE TO YOUR ACTUAL PATHS
    DATASET_ROOT = "data/View_of_Delft_dataset_PUBLIC/view_of_delft_PUBLIC"
    IMAGE_DIR = os.path.join(DATASET_ROOT, "lidar/training/image_2")
    LABEL_DIR = os.path.join(DATASET_ROOT, "lidar/training/label_2")
    CALIB_DIR = os.path.join(DATASET_ROOT, "lidar/training/calib")
    
    # Output directories
    OUTPUT_DIR = "./preprocessed_vod"
    TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
    VAL_DIR = os.path.join(OUTPUT_DIR, "val")
    TEST_DIR = os.path.join(OUTPUT_DIR, "test")
    
    # Dataset split ratios (as per project requirements: 59% train, 15% val, 26% test)
    TRAIN_SPLIT = 0.59
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.26
    
    # Classes for detection (from VoD dataset)
    CLASSES = ['Pedestrian', 'Cyclist', 'Car', 'Truck', 'Tram', 
               'Van', 'Misc', 'Person_sitting', 'DontCare']
    
    # Target classes for your first objective
    TARGET_CLASSES = ['Pedestrian', 'Cyclist', 'Car']
    
    # Class to index mapping
    CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}
    IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}
    
    # Image preprocessing parameters for ResNet18
    IMG_SIZE = (224, 224)  # ResNet18 input size
    IMG_MEAN = [0.485, 0.456, 0.406]  # ImageNet means
    IMG_STD = [0.229, 0.224, 0.225]   # ImageNet stds
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # Minimum object size threshold (to filter very small objects)
    MIN_BBOX_AREA = 100  # pixels

# ===========================
# DATA PARSING FUNCTIONS
# ===========================

def parse_kitti_label(label_file: str) -> List[Dict]:
    """
    Parse KITTI-format label file from VoD dataset
    
    KITTI format per line:
    type truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom 
    dimensions_h dimensions_w dimensions_l location_x location_y location_z rotation_y
    """
    objects = []
    
    if not os.path.exists(label_file):
        return objects
    
    with open(label_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(' ')
            
            if len(parts) < 15:
                continue
            
            obj = {
                'type': parts[0],
                'truncated': float(parts[1]),
                'occluded': int(parts[2]),
                'alpha': float(parts[3]),
                'bbox': [float(parts[4]), float(parts[5]), 
                        float(parts[6]), float(parts[7])],  # [left, top, right, bottom]
                'dimensions': [float(parts[8]), float(parts[9]), float(parts[10])],  # h, w, l
                'location': [float(parts[11]), float(parts[12]), float(parts[13])],  # x, y, z
                'rotation_y': float(parts[14])
            }
            
            objects.append(obj)
    
    return objects

def filter_objects(objects: List[Dict], target_classes: List[str], 
                   min_area: int = 100) -> List[Dict]:
    """Filter objects by class and minimum bounding box area"""
    filtered = []
    
    for obj in objects:
        # Check if object is in target classes
        if obj['type'] not in target_classes:
            continue
        
        # Calculate bbox area
        bbox = obj['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        
        # Filter by minimum area
        if area < min_area:
            continue
        
        filtered.append(obj)
    
    return filtered

def create_annotation_dict(image_id: str, objects: List[Dict], 
                          class_to_idx: Dict) -> Dict:
    """Create annotation dictionary for an image"""
    annotation = {
        'image_id': image_id,
        'objects': []
    }
    
    for obj in objects:
        obj_annotation = {
            'class': obj['type'],
            'class_idx': class_to_idx[obj['type']],
            'bbox': obj['bbox'],  # [left, top, right, bottom]
            'truncated': obj['truncated'],
            'occluded': obj['occluded']
        }
        annotation['objects'].append(obj_annotation)
    
    return annotation

# ===========================
# DATASET CREATION
# ===========================

def create_dataset_splits(image_dir: str, label_dir: str, config: Config):
    """
    Create train/val/test splits and save metadata
    """
    print("Creating dataset splits...")
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.endswith('.png') or f.endswith('.jpg')])
    
    print(f"Found {len(image_files)} images")
    
    # Parse all annotations
    annotations = []
    valid_images = []
    
    for img_file in image_files:
        image_id = os.path.splitext(img_file)[0]
        label_file = os.path.join(label_dir, f"{image_id}.txt")
        
        # Parse labels
        objects = parse_kitti_label(label_file)
        
        # Filter to target classes
        objects = filter_objects(objects, config.TARGET_CLASSES, config.MIN_BBOX_AREA)
        
        # Skip images with no relevant objects
        if len(objects) == 0:
            continue
        
        # Create annotation
        annotation = create_annotation_dict(image_id, objects, config.CLASS_TO_IDX)
        annotations.append(annotation)
        valid_images.append(img_file)
    
    print(f"Found {len(valid_images)} images with target objects")
    
    # Create splits
    num_images = len(valid_images)
    indices = np.random.permutation(num_images)
    
    train_end = int(config.TRAIN_SPLIT * num_images)
    val_end = train_end + int(config.VAL_SPLIT * num_images)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    splits = {
        'train': [(valid_images[i], annotations[i]) for i in train_indices],
        'val': [(valid_images[i], annotations[i]) for i in val_indices],
        'test': [(valid_images[i], annotations[i]) for i in test_indices]
    }
    
    print(f"Split sizes - Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    
    # Save splits to JSON files
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    for split_name, split_data in splits.items():
        split_file = os.path.join(config.OUTPUT_DIR, f"{split_name}_annotations.json")
        
        split_dict = {
            'images': [img for img, _ in split_data],
            'annotations': [ann for _, ann in split_data]
        }
        
        with open(split_file, 'w') as f:
            json.dump(split_dict, f, indent=2)
        
        print(f"Saved {split_name} split to {split_file}")
    
    # Save class information
    class_info = {
        'classes': config.TARGET_CLASSES,
        'class_to_idx': config.CLASS_TO_IDX,
        'idx_to_class': config.IDX_TO_CLASS,
        'num_classes': len(config.TARGET_CLASSES)
    }
    
    class_file = os.path.join(config.OUTPUT_DIR, "class_info.json")
    with open(class_file, 'w') as f:
        json.dump(class_info, f, indent=2)
    
    print(f"Saved class information to {class_file}")
    
    return splits

# ===========================
# PYTORCH DATASET CLASS
# ===========================

class VoDDataset(Dataset):
    """
    PyTorch Dataset for View of Delft
    Suitable for training ResNet18 for object classification
    """
    
    def __init__(self, image_dir: str, annotations_file: str, 
                 transform=None, target_transform=None):
        """
        Args:
            image_dir: Directory containing images
            annotations_file: JSON file with annotations
            transform: Image transformations
            target_transform: Target transformations
        """
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        self.images = data['images']
        self.annotations = data['annotations']
        
        print(f"Loaded {len(self.images)} images from {annotations_file}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Get annotation
        annotation = self.annotations[idx]
        
        # For classification: use the most prominent object (largest bbox)
        # Or you can aggregate multiple objects
        objects = annotation['objects']
        
        if len(objects) == 0:
            # Shouldn't happen due to filtering, but handle gracefully
            label = 0
        else:
            # Take the largest object as primary label
            largest_obj = max(objects, key=lambda obj: 
                            (obj['bbox'][2] - obj['bbox'][0]) * 
                            (obj['bbox'][3] - obj['bbox'][1]))
            label = largest_obj['class_idx']
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label, annotation

# ===========================
# DATA TRANSFORMATIONS
# ===========================

def get_transforms(config: Config, augment: bool = True):
    """
    Get image transforms for training/validation
    
    Args:
        config: Configuration object
        augment: Whether to apply data augmentation (for training)
    """
    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                 saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD)
        ])
    else:
        # Validation/test transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD)
        ])
    
    return transform

# ===========================
# CUSTOM COLLATE FUNCTION
# ===========================

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized annotations
    
    Args:
        batch: List of (image, label, annotation) tuples
    
    Returns:
        images: Batched images tensor
        labels: Batched labels tensor
        annotations: List of annotations (not batched)
    """
    images = []
    labels = []
    annotations = []
    
    for item in batch:
        images.append(item[0])
        labels.append(item[1])
        annotations.append(item[2])
    
    # Stack images and labels into tensors
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels)
    
    # Keep annotations as list (variable size)
    return images, labels, annotations


# ===========================
# DATA LOADERS
# ===========================

def create_dataloaders(config: Config):
    """Create PyTorch DataLoaders for train/val/test sets"""
    
    # Get transforms
    train_transform = get_transforms(config, augment=True)
    val_transform = get_transforms(config, augment=False)
    
    # Create datasets
    train_dataset = VoDDataset(
        image_dir=config.IMAGE_DIR,
        annotations_file=os.path.join(config.OUTPUT_DIR, "train_annotations.json"),
        transform=train_transform
    )
    
    val_dataset = VoDDataset(
        image_dir=config.IMAGE_DIR,
        annotations_file=os.path.join(config.OUTPUT_DIR, "val_annotations.json"),
        transform=val_transform
    )
    
    test_dataset = VoDDataset(
        image_dir=config.IMAGE_DIR,
        annotations_file=os.path.join(config.OUTPUT_DIR, "test_annotations.json"),
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    print(f"Created dataloaders:")
    print(f"  Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} images, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} images, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

# ===========================
# DATASET STATISTICS
# ===========================

def compute_dataset_statistics(annotations_file: str, class_to_idx: Dict):
    """Compute and display dataset statistics"""
    
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    annotations = data['annotations']
    
    # Count objects per class
    class_counts = {cls: 0 for cls in class_to_idx.keys()}
    total_objects = 0
    
    bbox_sizes = []
    objects_per_image = []
    
    for annotation in annotations:
        num_objects = len(annotation['objects'])
        objects_per_image.append(num_objects)
        
        for obj in annotation['objects']:
            class_counts[obj['class']] += 1
            total_objects += 1
            
            # Compute bbox size
            bbox = obj['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            bbox_sizes.append((width, height))
    
    # Print statistics
    print(f"\nDataset Statistics for {annotations_file}:")
    print(f"  Total images: {len(annotations)}")
    print(f"  Total objects: {total_objects}")
    print(f"  Avg objects per image: {np.mean(objects_per_image):.2f}")
    print(f"\nClass distribution:")
    for cls, count in class_counts.items():
        percentage = (count / total_objects) * 100 if total_objects > 0 else 0
        print(f"  {cls}: {count} ({percentage:.1f}%)")
    
    # Bbox statistics
    widths = [w for w, h in bbox_sizes]
    heights = [h for w, h in bbox_sizes]
    
    print(f"\nBounding box statistics:")
    print(f"  Width - mean: {np.mean(widths):.1f}, std: {np.std(widths):.1f}")
    print(f"  Height - mean: {np.mean(heights):.1f}, std: {np.std(heights):.1f}")
    
    return class_counts, bbox_sizes

# ===========================
# VISUALIZATION
# ===========================

def visualize_samples(dataloader, num_samples=4, config=Config()):
    """Visualize sample images with annotations"""
    
    # Get a batch
    images, labels, annotations = next(iter(dataloader))
    
    # Denormalize images for visualization
    mean = torch.tensor(config.IMG_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMG_STD).view(3, 1, 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(images))):
        # Denormalize
        img = images[i] * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        # Display
        axes[i].imshow(img)
        axes[i].set_title(f"Class: {config.IDX_TO_CLASS[labels[i].item()]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "sample_images.png"))
    print(f"Saved sample visualization to {config.OUTPUT_DIR}/sample_images.png")
    plt.close()

# ===========================
# MAIN PREPROCESSING PIPELINE
# ===========================

def main():
    """Main preprocessing pipeline"""
    
    print("="*60)
    print("View of Delft Dataset Preprocessing for ResNet18")
    print("="*60)
    
    # Initialize config
    config = Config()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Step 1: Create dataset splits
    print("\nStep 1: Creating dataset splits...")
    splits = create_dataset_splits(config.IMAGE_DIR, config.LABEL_DIR, config)
    
    # Step 2: Compute statistics for each split
    print("\nStep 2: Computing dataset statistics...")
    for split_name in ['train', 'val', 'test']:
        annotations_file = os.path.join(config.OUTPUT_DIR, f"{split_name}_annotations.json")
        compute_dataset_statistics(annotations_file, config.CLASS_TO_IDX)
    
    # Step 3: Create dataloaders
    print("\nStep 3: Creating PyTorch DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Step 4: Visualize samples
    # print("\nStep 4: Visualizing sample images...")
    # visualize_samples(train_loader, num_samples=4, config=config)
    
    # Step 5: Save preprocessing summary
    print("\nStep 5: Saving preprocessing summary...")
    summary = {
        'dataset_root': config.DATASET_ROOT,
        'target_classes': config.TARGET_CLASSES,
        'num_classes': len(config.TARGET_CLASSES),
        'train_size': len(train_loader.dataset),
        'val_size': len(val_loader.dataset),
        'test_size': len(test_loader.dataset),
        'image_size': config.IMG_SIZE,
        'batch_size': config.BATCH_SIZE,
        'preprocessing': {
            'normalization_mean': config.IMG_MEAN,
            'normalization_std': config.IMG_STD,
            'min_bbox_area': config.MIN_BBOX_AREA
        }
    }
    
    summary_file = os.path.join(config.OUTPUT_DIR, "preprocessing_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved preprocessing summary to {summary_file}")
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)
    print(f"\nOutput files saved to: {config.OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Review the preprocessing_summary.json file")
    print("2. Check sample visualizations in sample_images.png")
    print("3. Use the created dataloaders for training ResNet18")
    print("4. See the example training script (train_resnet18.py)")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Run preprocessing
    train_loader, val_loader, test_loader = main()
