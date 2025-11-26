import torch
import torch.nn as nn
import torch.optim as optim
from src.data.dataset import create_dataloaders
from src.models.aasist import AASIST
import os
from tqdm import tqdm
import sys
import time

def main():
    # CONFIG
    TRAIN_DIR = r"D:\SAMPOERNA\Semester 7\Capstone\Ensemble-Adaptive-Degradation-Module\data\processed\train_raw"
    CHECKPOINT_DIR = "checkpoints"
    MAX_BATCHES_PER_EPOCH = 10 # Limit for preliminary testing
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print("\n" + "="*60)
    print("          AASIST TRAINING PIPELINE (PRELIMINARY)")
    print("="*60)

    # GPU CHECK
    print(" [*] Checking Hardware Acceleration...")
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        print(f" [✓] GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device('cpu')
        print(" [!] WARNING: GPU not found. Using CPU (Slow).")
        print("     Run: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    
    # DATA LOADING
    print("-" * 60)
    print(" [*] Loading Dataset...")
    if not os.path.exists(TRAIN_DIR):
        print(" [!] ERROR: Data directory missing. Run src/data/preprocess_raw.py first.")
        return
        
    train_loader, _, weights = create_dataloaders(TRAIN_DIR, batch_size=8)
    print(f" [*] Dataset Loaded. Total Batches: {len(train_loader)}")
    print(f" [*] Limiting to {MAX_BATCHES_PER_EPOCH} batches per epoch.")

    # MODEL SETUP
    print(" [*] Initializing AASIST Model...")
    model = AASIST().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
    
    # TRAINING LOOP
    print("="*60)
    print(" STARTING TRAINING")
    print("="*60)
    
    for epoch in range(1, 6):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Progress Bar
        pbar = tqdm(enumerate(train_loader), total=MAX_BATCHES_PER_EPOCH, desc=f"Epoch {epoch}/5", unit="batch")
        
        for i, (inputs, labels, _) in pbar:
            if i >= MAX_BATCHES_PER_EPOCH: break # Stop early
            
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{100*correct/total:.1f}%"})
            
        # End Epoch Stats
        avg_loss = total_loss / (i+1)
        epoch_acc = 100 * correct / total if total > 0 else 0
        print(f" [Summary] Epoch {epoch}: Avg Loss = {avg_loss:.4f} | Accuracy = {epoch_acc:.2f}%")
        
        # Save
        save_path = os.path.join(CHECKPOINT_DIR, f"aasist_epoch_{epoch}.pt")
        torch.save(model.state_dict(), save_path)
        
    print("\n" + "="*60)
    print(" [✓] TRAINING COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()