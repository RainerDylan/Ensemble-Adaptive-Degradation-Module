import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

class ASVspoofDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        meta_path = os.path.join(data_dir, 'metadata.pkl')
        
        if not os.path.exists(meta_path):
            # Try looking one level up if not found (sometimes happens with structure changes)
            raise FileNotFoundError(f"Metadata not found in {data_dir}")
            
        with open(meta_path, 'rb') as f:
            self.metadata = pickle.load(f)
            
        # Only keep files that actually exist on disk
        self.file_ids = [f for f in self.metadata.keys() 
                         if os.path.exists(os.path.join(data_dir, f"{f}.npy"))]

        if len(self.file_ids) == 0:
            print(f"WARNING: No .npy files found in {data_dir}")

        # Calculate class weights for imbalance handling
        labels = [self.metadata[f] for f in self.file_ids]
        n_spoof = sum(labels)
        n_bonafide = len(labels) - n_spoof
        
        # Prevent division by zero
        w_bonafide = len(labels)/(2*n_bonafide) if n_bonafide > 0 else 1.0
        w_spoof = len(labels)/(2*n_spoof) if n_spoof > 0 else 1.0
        
        self.weights = torch.FloatTensor([w_bonafide, w_spoof])

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        label = self.metadata[file_id]
        
        data_path = os.path.join(self.data_dir, f"{file_id}.npy")
        data = np.load(data_path)
        
        # UNIVERSAL SHAPE FIX
        # If 1D (Raw): (64600,) -> (1, 64600)
        # If 2D (Spect): (500, 80) -> (1, 500, 80)
        data_tensor = torch.FloatTensor(data).unsqueeze(0)
        
        # FIX: Return 3 values (Data, Label, ID) to match train.py expectation
        return data_tensor, torch.LongTensor([label])[0], file_id

def create_dataloaders(train_dir, val_dir=None, batch_size=16, num_workers=0):
    # 1. Create Train Loader
    train_dataset = ASVspoofDataset(train_dir)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    
    # 2. Create Val Loader (Optional)
    val_loader = None
    if val_dir:
        try:
            val_dataset = ASVspoofDataset(val_dir)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=num_workers
            )
        except FileNotFoundError:
            print(f"Warning: Validation dir {val_dir} not found or empty. Skipping validation.")

    # 3. Return exactly 3 values as train.py expects
    return train_loader, val_loader, train_dataset.weights