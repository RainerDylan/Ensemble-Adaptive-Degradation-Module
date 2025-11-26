"""
Training script for audio deepfake detection
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import create_dataloaders
from models.simple_cnn import SimpleCNN


def calculate_eer(labels, scores):
    """
    Calculate Equal Error Rate (EER)
    
    Args:
        labels: Ground truth labels (0=bonafide, 1=spoof)
        scores: Prediction scores for spoof class
        
    Returns:
        eer: Equal Error Rate as percentage
        threshold: Threshold at EER
    """
    # Sort by scores
    sorted_indices = np.argsort(scores)
    labels = np.array(labels)[sorted_indices]
    scores = np.array(scores)[sorted_indices]
    
    # Calculate FAR and FRR for each threshold
    n_spoof = np.sum(labels == 1)
    n_bonafide = np.sum(labels == 0)
    
    far = []
    frr = []
    
    for i, threshold in enumerate(scores):
        # False Accept Rate: spoof accepted as bonafide
        false_accepts = np.sum((labels[:i] == 1))
        far.append(false_accepts / n_spoof if n_spoof > 0 else 0)
        
        # False Reject Rate: bonafide rejected as spoof
        false_rejects = np.sum((labels[i:] == 0))
        frr.append(false_rejects / n_bonafide if n_bonafide > 0 else 0)
    
    far = np.array(far)
    frr = np.array(frr)
    
    # Find EER (where FAR = FRR)
    abs_diff = np.abs(far - frr)
    min_index = np.argmin(abs_diff)
    eer = (far[min_index] + frr[min_index]) / 2
    threshold = scores[min_index]
    
    return eer * 100, threshold


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_labels = []
    all_scores = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for spectrograms, labels, _ in pbar:
        # Move to device
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits, _ = model(spectrograms)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Store for EER calculation
        probs = torch.softmax(logits, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(probs[:, 1].detach().cpu().numpy())  # Spoof probability
        
        # Update running loss
        running_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    epoch_eer, _ = calculate_eer(all_labels, all_scores)
    
    return epoch_loss, epoch_acc, epoch_eer


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for spectrograms, labels, _ in tqdm(val_loader, desc='Validating'):
            # Move to device
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits, _ = model(spectrograms)
            loss = criterion(logits, labels)
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store for EER calculation
            probs = torch.softmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())
            
            # Update running loss
            running_loss += loss.item()
    
    # Calculate validation metrics
    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    val_eer, _ = calculate_eer(all_labels, all_scores)
    
    return val_loss, val_acc, val_eer


def train(config):
    """Main training function"""
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, class_weights = create_dataloaders(
        train_dir=config['train_dir'],
        val_dir=config['val_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Create model
    print("\nInitializing model...")
    model = SimpleCNN(dropout_rate=config['dropout_rate'])
    model = model.to(device)
    
    total_params, trainable_params = model.count_parameters()
    print(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total")
    
    # Loss function with class weights
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # TensorBoard writer
    log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)
    
    # Training loop
    print(f"\nStarting training for {config['num_epochs']} epochs...")
    print("=" * 70)
    
    best_val_eer = float('inf')
    best_epoch = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_eer': [],
        'val_loss': [],
        'val_acc': [],
        'val_eer': [],
        'learning_rate': []
    }
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc, train_eer = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        if val_loader:
            val_loss, val_acc, val_eer = validate(
                model, val_loader, criterion, device
            )
        else:
            val_loss, val_acc, val_eer = 0, 0, 0
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"\nTrain - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, EER: {train_eer:.2f}%")
        if val_loader:
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, EER: {val_eer:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('EER/train', train_eer, epoch)
        writer.add_scalar('EER/val', val_eer, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_eer'].append(train_eer)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_eer'].append(val_eer)
        history['learning_rate'].append(current_lr)
        
        # Learning rate scheduling
        if val_loader:
            scheduler.step(val_eer)
        
        # Save best model
        if val_loader and val_eer < best_val_eer:
            best_val_eer = val_eer
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_eer': val_eer,
                'val_acc': val_acc,
                'config': config
            }
            
            checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_model.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"\n✓ Best model saved (EER: {val_eer:.2f}%)")
        
        # Save checkpoint every N epochs
        if epoch % config['save_every'] == 0:
            checkpoint_path = os.path.join(
                config['checkpoint_dir'], 
                f'checkpoint_epoch_{epoch}.pt'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, checkpoint_path)
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best validation EER: {best_val_eer:.2f}% (Epoch {best_epoch})")
    
    # Save training history
    history_path = os.path.join(config['checkpoint_dir'], 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    writer.close()
    
    return model, history


def main():
    """Main function"""
    
    # Configuration
    config = {
        # Data paths - UPDATE THESE
        'train_dir': r'D:\SAMPOERNA\Semester 7\Capstone\2019\LA\preprocessed\train',
        'val_dir': None,  # Set to validation directory if you have one
        
        # Model hyperparameters
        'dropout_rate': 0.3,
        
        # Training hyperparameters
        'batch_size': 32,
        'num_epochs': 20,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        
        # Data loading
        'num_workers': 4,
        
        # Checkpointing
        'checkpoint_dir': 'checkpoints',
        'save_every': 5,
        
        # Reproducibility
        'seed': 42
    }
    
    print("Configuration:")
    print(json.dumps(config, indent=2))
    
    # Train model
    model, history = train(config)
    
    print("\n✓ Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
