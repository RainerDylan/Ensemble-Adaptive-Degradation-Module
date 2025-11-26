import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import sys

class ASVspoofDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        meta_path = os.path.join(data_dir, 'metadata.pkl')
        
        if not os.path.exists(meta_path):
            print(f" [ERROR] Metadata not found at {meta_path}")
            print(f"         Did you run the correct preprocessing script?")
            sys.exit(1)
            
        with open(meta_path, 'rb') as f:
            self.metadata = pickle.load(f)
            
        self.file_ids = [f for f in self.metadata.keys() 
                         if os.path.exists(os.path.join(data_dir, f"{f}.npy"))]

        # Class Weight Calculation
        labels = [self.metadata[f] for f in self.file_ids]
        n_spoof = sum(labels)
        n_bonafide = len(labels) - n_spoof
        
        w_bona = len(labels)/(2*n_bonafide) if n_bonafide > 0 else 1.0
        w_spoof = len(labels)/(2*n_spoof) if n_spoof > 0 else 1.0
        self.weights = torch.FloatTensor([w_bona, w_spoof])

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        label = self.metadata[file_id]
        data = np.load(os.path.join(self.data_dir, f"{file_id}.npy"))
        
        # Add Channel Dimension: (N, ...) -> (1, N, ...)
        data_tensor = torch.FloatTensor(data).unsqueeze(0)
        
        return data_tensor, torch.LongTensor([label])[0], file_id

def create_dataloaders(train_dir, val_dir=None, batch_size=16, num_workers=0):
    train_ds = ASVspoofDataset(train_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    val_loader = None
    if val_dir and os.path.exists(val_dir):
        val_ds = ASVspoofDataset(val_dir)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
    return train_loader, val_loader, train_ds.weights