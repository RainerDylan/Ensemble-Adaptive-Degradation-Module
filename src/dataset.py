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
    
    Returns:
        spectrogram: Tensor of shape (1, 500, 80) - (channels, time, frequency)
        label: 0 for bonafide, 1 for spoof
        file_id: String identifier for the audio file
    """
    
    def __init__(self, preprocessed_dir, transform=None):
        """
        Args:
            preprocessed_dir: Directory with .npy spectrogram files and metadata.pkl
            transform: Optional transform to apply to spectrograms
        """
        self.preprocessed_dir = preprocessed_dir
        self.transform = transform
        
        # Load metadata (file_id -> label mapping)
        metadata_path = os.path.join(preprocessed_dir, 'metadata.pkl')
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
        print(f"  Class ratio (spoof/bonafide): {self.n_spoof/self.n_bonafide:.2f}")
    
    def __len__(self):
        return len(self.file_ids)
    
    def __getitem__(self, idx):
        """
        Load and return a single sample
        """
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
        
        # Apply transforms if any
        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        return spectrogram, label, file_id
    
    def get_class_weights(self):
        """
        Calculate class weights for handling imbalanced dataset
        Useful for weighted loss functions
        
        Returns:
            weights: Tensor of shape (2,) with weights for [bonafide, spoof]
        """
        total = len(self.file_ids)
        weight_bonafide = total / (2 * self.n_bonafide)
        weight_spoof = total / (2 * self.n_spoof)
        
        weights = torch.FloatTensor([weight_bonafide, weight_spoof])
        return weights


def create_dataloaders(train_dir, 
                       val_dir=None,
                       batch_size=64,
                       num_workers=4,
                       pin_memory=True):
    """
    Create PyTorch DataLoaders for training and validation
    
    Args:
        train_dir: Directory with preprocessed training data
        val_dir: Directory with preprocessed validation data (optional)
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation (None if val_dir not provided)
        class_weights: Weights for handling class imbalance
    """
    # Create training dataset
    train_dataset = ASVspoofDataset(train_dir)
    
    # Create training dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches
    )
    
    # Get class weights from training data
    class_weights = train_dataset.get_class_weights()
    
    # Create validation dataloader if validation directory provided
    val_loader = None
    if val_dir:
        val_dataset = ASVspoofDataset(val_dir)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle validation data
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
    
    print(f"\nDataLoaders created:")
    print(f"  Training batches: {len(train_loader)}")
    if val_loader:
        print(f"  Validation batches: {len(val_loader)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Class weights: {class_weights}")
    
    return train_loader, val_loader, class_weights


def test_dataset():
    """
    Test function to verify dataset is working correctly
    """
    print("Testing ASVspoofDataset...")
    
    # Path to preprocessed data
    preprocessed_dir = r"D:\SAMPOERNA\Semester 7\Capstone\2019\LA\preprocessed\train"
    
    # Create dataset
    dataset = ASVspoofDataset(preprocessed_dir)
    
    # Test loading a single sample
    spec, label, file_id = dataset[0]
    
    print(f"\nSample loaded successfully!")
    print(f"  File ID: {file_id}")
    print(f"  Spectrogram shape: {spec.shape}")  # Should be (1, 500, 80)
    print(f"  Label: {label.item()} ({'bonafide' if label == 0 else 'spoof'})")
    print(f"  Spectrogram dtype: {spec.dtype}")
    print(f"  Spectrogram range: [{spec.min():.2f}, {spec.max():.2f}]")
    
    # Test dataloader
    print("\nTesting DataLoader...")
    train_loader, _, class_weights = create_dataloaders(
        preprocessed_dir,
        batch_size=8,
        num_workers=0  # Use 0 for testing to avoid multiprocessing issues
    )
    
    # Get one batch
    batch_spec, batch_labels, batch_ids = next(iter(train_loader))
    
    print(f"\nBatch loaded successfully!")
    print(f"  Batch spectrogram shape: {batch_spec.shape}")  # Should be (8, 1, 500, 80)
    print(f"  Batch labels shape: {batch_labels.shape}")  # Should be (8,)
    print(f"  Batch labels: {batch_labels.tolist()}")
    print(f"  Sample IDs: {batch_ids[:3]}...")
    
    print("\n✓ Dataset test passed!")


if __name__ == "__main__":
    test_dataset()
