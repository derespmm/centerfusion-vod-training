import os
import time
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# Import from preprocessing script
from resnet_preprocessing import Config, create_dataloaders

# ===========================
# TRAINING CONFIGURATION
# ===========================

class TrainConfig:
    """Training configuration"""
    
    # Paths
    DATASET_CONFIG = Config()  # Use preprocessing config
    OUTPUT_DIR = "./resnet18_output"
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    
    # Model parameters
    NUM_CLASSES = len(DATASET_CONFIG.TARGET_CLASSES)
    PRETRAINED = True  # Use ImageNet pretrained weights
    
    # Training hyperparameters
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    
    # Learning rate scheduler
    SCHEDULER_STEP_SIZE = 7  # Decay LR every 7 epochs
    SCHEDULER_GAMMA = 0.1    # Multiply LR by 0.1
    
    # Early stopping
    PATIENCE = 10  # Stop if no improvement for 10 epochs
    
    # Device
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")  # Mac M-series GPU
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")  # NVIDIA GPU
        print("Using CUDA GPU")
    else:
        DEVICE = torch.device("cpu")  # CPU fallback
        print("Using CPU")
    
    # Logging
    PRINT_FREQ = 10  # Print every N batches
    SAVE_FREQ = 5    # Save checkpoint every N epochs

# ===========================
# MODEL SETUP
# ===========================

def create_resnet18(num_classes, pretrained=True, device='cpu'):
    """
    Create ResNet18 model for object classification
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        device: Device to load model on
    
    Returns:
        model: ResNet18 model
    """
    print(f"Creating ResNet18 model...")
    print(f"  Pretrained: {pretrained}")
    print(f"  Number of classes: {num_classes}")
    
    # Load pretrained ResNet18
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        print("  Loaded ImageNet pretrained weights")
    else:
        model = models.resnet18(weights=None)
        print("  Initialized from scratch")
    
    # Modify final fully connected layer for our number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    print(f"  Modified final FC layer: {num_features} -> {num_classes}")
    
    # Move to device
    model = model.to(device)
    print(f"  Model moved to {device}")
    
    return model

def count_parameters(model):
    """Count trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    return total_params, trainable_params

# ===========================
# TRAINING FUNCTIONS
# ===========================

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, config):
    """
    Train for one epoch
    
    Returns:
        avg_loss: Average loss for the epoch
        accuracy: Accuracy for the epoch
    """
    model.train()
    
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]")
    
    for batch_idx, (inputs, labels, _) in enumerate(pbar):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        batch_size = inputs.size(0)
        total_samples += batch_size
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Statistics - FIX: Use .item() instead of .double()
        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds == labels.data).item()  # Changed here
        
        # Update progress bar
        if (batch_idx + 1) % config.PRINT_FREQ == 0:
            batch_loss = running_loss / total_samples
            batch_acc = running_corrects / total_samples  # Changed here
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'acc': f'{batch_acc:.4f}'
            })
    
    # Epoch statistics
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples  # Changed here
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, epoch, config, phase='Val'):
    """
    Validate/test the model
    
    Returns:
        avg_loss: Average loss
        accuracy: Accuracy
        all_preds: All predictions
        all_labels: All true labels
    """
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    all_preds = []
    all_labels = []
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [{phase}]")
    
    with torch.no_grad():
        for inputs, labels, _ in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            batch_size = inputs.size(0)
            total_samples += batch_size
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Statistics - FIX: Use .item() instead of .double()
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels.data).item()  # Changed here
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            batch_loss = running_loss / total_samples
            batch_acc = running_corrects / total_samples  # Changed here
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'acc': f'{batch_acc:.4f}'
            })
    
    # Epoch statistics
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples  # Changed here
    
    return epoch_loss, epoch_acc, all_preds, all_labels  # Changed here (removed .item())


# ===========================
# TRAINING LOOP
# ===========================

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler, config, resume_from=None):
    """
    Main training loop with validation
    
    Args:
        model: PyTorch model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Training configuration
        resume_from: Path to checkpoint to resume from
    
    Returns:
        model: Trained model
        history: Training history
    """
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    # Create output directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    # Best model tracking
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    
    # Early stopping
    epochs_no_improve = 0
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        print(f"\nResuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        history = checkpoint['history']
        print(f"Resumed from epoch {start_epoch}, best acc: {best_acc:.4f}")
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print("-" * 40)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, 
            config.DEVICE, epoch, config
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, 
            config.DEVICE, epoch, config, phase='Val'
        )
        
        # Step scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Check if best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            
            # Save best model
            best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'history': history
            }, best_model_path)
            print(f"  ✓ New best model! Saved to {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s)")
        
        # Save periodic checkpoint
        if (epoch + 1) % config.SAVE_FREQ == 0:
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR, 
                f"checkpoint_epoch_{epoch+1}.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'history': history
            }, checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")
        
        # Early stopping check
        if epochs_no_improve >= config.PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"No improvement for {config.PATIENCE} consecutive epochs")
            break
    
    # Training complete
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch+1}")
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

# ===========================
# EVALUATION
# ===========================

def evaluate_model(model, test_loader, criterion, config):
    """
    Evaluate model on test set
    
    Returns:
        test_loss: Test loss
        test_acc: Test accuracy
        predictions: All predictions
        true_labels: All true labels
    """
    print("\n" + "="*60)
    print("Evaluating on Test Set")
    print("="*60)
    
    test_loss, test_acc, predictions, true_labels = validate(
        model, test_loader, criterion, 
        config.DEVICE, 0, config, phase='Test'
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    
    return test_loss, test_acc, predictions, true_labels

def compute_metrics(true_labels, predictions, class_names):
    """Compute detailed classification metrics"""
    
    # Classification report
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    report = classification_report(
        true_labels, predictions, 
        target_names=class_names, 
        digits=4
    )
    print(report)
    
    # Save report
    return report

def plot_confusion_matrix(true_labels, predictions, class_names, save_path):
    """Plot and save confusion matrix"""
    
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

def plot_training_history(history, save_path):
    """Plot training history"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate
    axes[1, 0].plot(history['lr'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)
    
    # Best epoch marker
    best_epoch = np.argmax(history['val_acc'])
    best_acc = history['val_acc'][best_epoch]
    
    axes[1, 1].plot(history['val_acc'])
    axes[1, 1].axvline(x=best_epoch, color='r', linestyle='--', 
                       label=f'Best Epoch: {best_epoch+1}')
    axes[1, 1].scatter([best_epoch], [best_acc], color='r', s=100, zorder=5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Validation Accuracy')
    axes[1, 1].set_title(f'Best Val Acc: {best_acc:.4f} at Epoch {best_epoch+1}')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()

# ===========================
# MAIN TRAINING PIPELINE
# ===========================

def main():
    """Main training pipeline"""
    
    print("="*60)
    print("ResNet18 Training for View of Delft Object Classification")
    print("="*60)
    
    # Initialize config
    config = TrainConfig()
    
    print(f"\nDevice: {config.DEVICE}")
    print(f"Number of classes: {config.NUM_CLASSES}")
    print(f"Classes: {config.DATASET_CONFIG.TARGET_CLASSES}")
    
    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(config.DATASET_CONFIG)
    
    # Create model
    print("\nCreating model...")
    model = create_resnet18(
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED,
        device=config.DEVICE
    )
    
    # Count parameters
    count_parameters(model)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    print(f"\nLoss function: CrossEntropyLoss")
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )
    print(f"Optimizer: SGD")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Momentum: {config.MOMENTUM}")
    print(f"  Weight decay: {config.WEIGHT_DECAY}")
    
    # Learning rate scheduler
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=config.SCHEDULER_STEP_SIZE,
        gamma=config.SCHEDULER_GAMMA
    )
    print(f"LR Scheduler: StepLR")
    print(f"  Step size: {config.SCHEDULER_STEP_SIZE}")
    print(f"  Gamma: {config.SCHEDULER_GAMMA}")
    
    # Train model
    model, history = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, scheduler, config
    )
    
    # Plot training history
    history_plot_path = os.path.join(config.OUTPUT_DIR, "training_history.png")
    plot_training_history(history, history_plot_path)
    
    # Evaluate on test set
    test_loss, test_acc, predictions, true_labels = evaluate_model(
        model, test_loader, criterion, config
    )
    
    # Compute detailed metrics
    report = compute_metrics(
        true_labels, predictions, 
        config.DATASET_CONFIG.TARGET_CLASSES
    )
    
    # Plot confusion matrix
    cm_path = os.path.join(config.OUTPUT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(
        true_labels, predictions,
        config.DATASET_CONFIG.TARGET_CLASSES,
        cm_path
    )
    
    # Save final results
    results = {
        'model': 'ResNet18',
        'num_classes': config.NUM_CLASSES,
        'classes': config.DATASET_CONFIG.TARGET_CLASSES,
        'pretrained': config.PRETRAINED,
        'num_epochs_trained': len(history['train_loss']),
        'best_val_acc': float(max(history['val_acc'])),
        'best_epoch': int(np.argmax(history['val_acc']) + 1),
        'test_loss': float(test_loss),
        'test_acc': float(test_acc),
        'hyperparameters': {
            'learning_rate': config.LEARNING_RATE,
            'momentum': config.MOMENTUM,
            'weight_decay': config.WEIGHT_DECAY,
            'batch_size': config.DATASET_CONFIG.BATCH_SIZE,
            'optimizer': 'SGD',
            'scheduler': 'StepLR',
            'scheduler_step_size': config.SCHEDULER_STEP_SIZE,
            'scheduler_gamma': config.SCHEDULER_GAMMA
        },
        'training_time_minutes': None  # Will be filled during training
    }
    
    results_path = os.path.join(config.OUTPUT_DIR, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Save classification report
    report_path = os.path.join(config.OUTPUT_DIR, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Classification report saved to {report_path}")
    
    print("\n" + "="*60)
    print("Training pipeline complete!")
    print("="*60)
    print(f"\nAll outputs saved to: {config.OUTPUT_DIR}")
    print("\nGenerated files:")
    print(f"  - best_model.pth (in {config.CHECKPOINT_DIR})")
    print(f"  - training_history.png")
    print(f"  - confusion_matrix.png")
    print(f"  - training_results.json")
    print(f"  - classification_report.txt")
    
    return model, history, results

if __name__ == "__main__":
    # Run training
    model, history, results = main()
