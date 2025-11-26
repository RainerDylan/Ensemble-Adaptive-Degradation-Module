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

# Add src to path to correctly find modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import create_dataloaders
from src.models.simple_cnn import SimpleCNN

def calculate_eer(labels, scores):
    """Calculate Equal Error Rate (EER)"""
    if len(labels) == 0: return 0, 0
    
    sorted_indices = np.argsort(scores)
    labels = np.array(labels)[sorted_indices]
    scores = np.array(scores)[sorted_indices]
    
    n_spoof = np.sum(labels == 1)
    n_bonafide = np.sum(labels == 0)
    
    if n_spoof == 0 or n_bonafide == 0: return 0, 0
    
    far = []
    frr = []
    
    for i in range(len(scores)):
        false_accepts = np.sum((labels[:i] == 1))
        far.append(false_accepts / n_spoof)
        
        false_rejects = np.sum((labels[i:] == 0))
        frr.append(false_rejects / n_bonafide)
    
    far = np.array(far)
    frr = np.array(frr)
    
    abs_diff = np.abs(far - frr)
    min_index = np.argmin(abs_diff)
    eer = (far[min_index] + frr[min_index]) / 2
    threshold = scores[min_index]
    
    return eer * 100, threshold

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_scores = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for spectrograms, labels, _ in pbar:
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(spectrograms)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        probs = torch.softmax(logits, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(probs[:, 1].detach().cpu().numpy())
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * correct / total:.2f}%'})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    epoch_eer, _ = calculate_eer(all_labels, all_scores)
    
    return epoch_loss, epoch_acc, epoch_eer

def train(config):
    # Seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    print("\nCreating dataloaders...")
    try:
        train_loader, val_loader, class_weights = create_dataloaders(
            train_dir=config['train_dir'],
            val_dir=config['val_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
    except FileNotFoundError as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("Check if the path in config['train_dir'] exists and contains processed data.")
        return None, None

    print("\nInitializing model...")
    model = SimpleCNN(dropout_rate=config['dropout_rate']).to(device)
    
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    print(f"\nStarting training for {config['num_epochs']} epochs...")
    history = {'train_loss': [], 'train_acc': []}
    
    for epoch in range(1, config['num_epochs'] + 1):
        train_loss, train_acc, train_eer = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        print(f"Epoch {epoch} - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, EER: {train_eer:.2f}%")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Save checkpoint every 5 epochs or last epoch
        if epoch % config['save_every'] == 0 or epoch == config['num_epochs']:
            checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    return model, history

def main():
    # --- CONFIGURATION WITH YOUR PATHS ---
    config = {
        # Updated to match your project structure
        'train_dir': r'D:\SAMPOERNA\Semester 7\Capstone\Ensemble-Adaptive-Degradation-Module\data\processed\train',
        'val_dir': None,
        'dropout_rate': 0.3,
        'batch_size': 16, # Reduced batch size to be safe
        'num_epochs': 5,  # Reduced epochs for quick test
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'num_workers': 0, # Set to 0 for Windows compatibility to avoid multiprocessing errors
        'checkpoint_dir': 'checkpoints',
        'save_every': 5,
        'seed': 42
    }
    
    print("Configuration loaded.")
    model, history = train(config)
    
    if model:
        print("\n✓ Training pipeline completed successfully!")

if __name__ == "__main__":
    main()