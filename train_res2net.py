import torch
import torch.nn as nn
import torch.optim as optim
from src.data.dataset import create_dataloaders
from src.models.res2net import Res2Net
import os

def main():
    # Point to SPECTROGRAM data folder
    TRAIN_DIR = r"D:\SAMPOERNA\Semester 7\Capstone\Ensemble-Adaptive-Degradation-Module\data\processed\train_spect"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training Res2Net on {DEVICE}...")
    
    if not os.path.exists(TRAIN_DIR):
        print("ERROR: Run src/data/preprocess_spect.py first!")
        return

    train_loader, _, weights = create_dataloaders(TRAIN_DIR, batch_size=16)
    model = Res2Net().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
    
    for epoch in range(1, 6):
        model.train()
        total_loss = 0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch} Loss: {total_loss/len(train_loader):.4f}")
        torch.save(model.state_dict(), f"checkpoints/res2net_epoch_{epoch}.pt")

if __name__ == "__main__":
    main()