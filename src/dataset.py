"""
PyTorch Dataset for ASVspoof 2019 preprocessed spectrograms
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle

class ASVspoofDataset(Dataset):
    """
    Dataset class for loading preprocessed Mel-spectrograms
    Input: Preprocessed directory with .npy files and metadata.pkl
    Output: (spectrogram, label, file_id)
    """
    
    def __init__(self, preprocessed_dir, transform=None):
        self.preprocessed_dir = preprocessed_dir
        self.transform = transform
        
        # Load metadata (file_id -> label mapping)
        metadata_path = os.path.join(preprocessed_dir, 'metadata.pkl')
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}. Did you run preprocessing.py?")

        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Get list of available preprocessed files
        self.file_ids = []
        for file_id in self.metadata.keys():
            spec_path = os.path.join(preprocessed_dir, f"{file_id}.npy")
            if os.path.exists(spec_path):
                self.file_ids.append(file_id)
        
        # Count bonafide vs spoof
        self.labels = [self.metadata[fid] for fid in self.file_ids]
        self.n_bonafide = sum(1 for l in self.labels if l == 0)
        self.n_spoof = sum(1 for l in self.labels if l == 1)
        
        print(f"Loaded ASVspoofDataset from {preprocessed_dir}")
        print(f"  Total samples: {len(self.file_ids)}")
        print(f"  Bonafide: {self.n_bonafide}")
        print(f"  Spoof: {self.n_spoof}")
        if self.n_bonafide > 0:
            print(f"  Class ratio (spoof/bonafide): {self.n_spoof/self.n_bonafide:.2f}")
    
    def __len__(self):
        return len(self.file_ids)
    
    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        label = self.metadata[file_id]
        
        # Load spectrogram
        spec_path = os.path.join(self.preprocessed_dir, f"{file_id}.npy")
        spectrogram = np.load(spec_path)  # Shape: (500, 80)
        
        # Add channel dimension: (500, 80) -> (1, 500, 80)
        spectrogram = np.expand_dims(spectrogram, axis=0)
        
        # Convert to torch tensor
        spectrogram = torch.FloatTensor(spectrogram)
        label = torch.LongTensor([label])[0]  # Scalar tensor
        
        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        return spectrogram, label, file_id
    
    def get_class_weights(self):
        total = len(self.file_ids)
        if self.n_bonafide == 0 or self.n_spoof == 0:
            return torch.FloatTensor([1.0, 1.0])
            
        weight_bonafide = total / (2 * self.n_bonafide)
        weight_spoof = total / (2 * self.n_spoof)
        
        weights = torch.FloatTensor([weight_bonafide, weight_spoof])
        return weights

def create_dataloaders(train_dir, val_dir=None, batch_size=64, num_workers=4, pin_memory=True):
    # Create training dataset
    train_dataset = ASVspoofDataset(train_dir)
    
    # Create training dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    class_weights = train_dataset.get_class_weights()
    
    val_loader = None
    if val_dir:
        val_dataset = ASVspoofDataset(val_dir)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
    
    print(f"\nDataLoaders created:")
    print(f"  Training batches: {len(train_loader)}")
    if val_loader:
        print(f"  Validation batches: {len(val_loader)}")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader, class_weights

def test_dataset():
    print("Testing ASVspoofDataset...")
    
    # Updated path to match your project structure
    preprocessed_dir = r"D:\SAMPOERNA\Semester 7\Capstone\Ensemble-Adaptive-Degradation-Module\data\processed\train"
    
    if not os.path.exists(preprocessed_dir):
        print(f"Error: Directory not found: {preprocessed_dir}")
        print("Please run src/preprocessing.py first.")
        return

    dataset = ASVspoofDataset(preprocessed_dir)
    
    # Test loading a single sample
    spec, label, file_id = dataset[0]
    
    print(f"\nSample loaded successfully!")
    print(f"  File ID: {file_id}")
    print(f"  Spectrogram shape: {spec.shape}")
    print(f"  Label: {label.item()}")
    
    print("\n✓ Dataset test passed!")

if __name__ == "__main__":
    test_dataset()