# Complete Setup Guide for Audio Deepfake Detection

## Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional but recommended)
- At least 16GB RAM
- ASVspoof 2019 LA dataset downloaded

## Step-by-Step Instructions

### 1. Check Your Data Structure

First, verify you have the ASVspoof 2019 dataset with this structure:

```
D:\SAMPOERNA\Semester 7\Capstone\2019\LA\
├── ASVspoof2019_LA_train\
│   └── flac\                          # Your training audio files
│       ├── LA_T_1000137.flac
│       ├── LA_T_1000265.flac
│       └── ... (more files)
├── ASVspoof2019_LA_cm_protocols\
│   ├── ASVspoof2019.LA.cm.train.trn.txt
│   ├── ASVspoof2019.LA.cm.dev.trl.txt
│   └── ASVspoof2019.LA.cm.eval.trl.txt
└── ASVspoof2019_LA_dev\               # Optional: development set
    └── flac\
```

**Protocol File Format** (example line):
```
LA_0003 LA_T_1138614 - bonafide
LA_0003 LA_T_1377600 A07 spoof
```
Format: `speaker_id file_id attack_type label`

---

### 2. Environment Setup

#### A. Create Virtual Environment
```bash
# Navigate to your project directory
cd path\to\Ensemble-Adaptive-Degradation-Module

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # Mac/Linux
```

#### B. Install Dependencies
```bash
# Install PyTorch (choose appropriate version for your system)
# For CUDA 11.8:
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

#### C. Verify PyTorch Installation
```python
# Run in Python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

### 3. Create Project Structure

Create these directories in your repository:

```bash
mkdir -p data/processed/train
mkdir -p data/processed/dev
mkdir -p checkpoints
mkdir -p logs
mkdir -p src/models
```

---

### 4. Data Preprocessing (CRITICAL STEP)

#### A. Update Paths in `src/preprocessing.py`

Open `src/preprocessing.py` and update the `main()` function with your actual paths:

```python
def main():
    preprocessor = AudioPreprocessor()
    
    # UPDATE THESE PATHS
    audio_dir = r"D:\SAMPOERNA\Semester 7\Capstone\2019\LA\ASVspoof2019_LA_train\flac"
    protocol_file = r"D:\SAMPOERNA\Semester 7\Capstone\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
    output_dir = r"D:\SAMPOERNA\Semester 7\Capstone\2019\LA\preprocessed\train"
    
    # Start with 100 samples for testing
    metadata = preprocessor.preprocess_dataset(
        audio_dir=audio_dir,
        protocol_file=protocol_file,
        output_dir=output_dir,
        max_samples=100  # Remove this to process all data
    )
```

#### B. Run Preprocessing

```bash
# Test with 100 samples first
python src/preprocessing.py
```

**Expected Output:**
```
Initialized AudioPreprocessor:
  Sample Rate: 16000 Hz
  FFT Size: 512
  Hop Length: 160 (10.0ms)
  Mel Bands: 80
  Frequency Range: 20-8000 Hz
  Max Frames: 500 (5.0s)

Reading protocol file: ...
Found 25380 files in protocol
Limited to 100 samples for testing

Preprocessing audio files...
100%|████████████| 100/100 [00:45<00:00,  2.21it/s]

Preprocessing complete:
  Successful: 100
  Failed: 0
Metadata saved to ...

✓ Preprocessing complete!
```

#### C. Verify Preprocessed Data

```python
# Run this in Python to verify
import numpy as np
import os

output_dir = r"D:\SAMPOERNA\Semester 7\Capstone\2019\LA\preprocessed\train"

# Check files exist
npy_files = [f for f in os.listdir(output_dir) if f.endswith('.npy')]
print(f"Found {len(npy_files)} .npy files")

# Load one sample
sample = np.load(os.path.join(output_dir, npy_files[0]))
print(f"Sample shape: {sample.shape}")  # Should be (500, 80)
print(f"Sample range: [{sample.min():.2f}, {sample.max():.2f}]")
```

---

### 5. Test Dataset Loading

```bash
python src/dataset.py
```

**Expected Output:**
```
Testing ASVspoofDataset...
Loaded ASVspoofDataset from ...
  Total samples: 100
  Bonafide: 8
  Spoof: 92
  Class ratio (spoof/bonafide): 11.50

Sample loaded successfully!
  File ID: LA_T_1000137
  Spectrogram shape: torch.Size([1, 500, 80])
  Label: 1 (spoof)
  Spectrogram dtype: torch.float32
  Spectrogram range: [-2.85, 3.12]

Testing DataLoader...
DataLoaders created:
  Training batches: 12
  Batch size: 8
  Class weights: tensor([6.2500, 0.5435])

Batch loaded successfully!
  Batch spectrogram shape: torch.Size([8, 1, 500, 80])
  Batch labels shape: torch.Size([8])
  Batch labels: [1, 1, 1, 1, 1, 1, 1, 1]

✓ Dataset test passed!
```

---

### 6. Test Model Architecture

```bash
python src/models/simple_cnn.py
```

**Expected Output:**
```
Testing SimpleCNN model...

Model Parameters:
  Total: 121,410
  Trainable: 121,410

Input shape: torch.Size([4, 1, 500, 80])

Output shapes:
  Logits: torch.Size([4, 2])
  Embedding: torch.Size([4, 128])

Sample predictions:
  Logits: [-0.0234, 0.0156]
  Probabilities: [0.4902, 0.5098]
  Predicted class: 1

✓ Model test passed!
```

---

### 7. Train the Model

#### A. Update Paths in `src/train.py`

Open `src/train.py` and update paths in the `main()` function:

```python
config = {
    # UPDATE THIS PATH
    'train_dir': r'D:\SAMPOERNA\Semester 7\Capstone\2019\LA\preprocessed\train',
    'val_dir': None,  # Set if you have validation data
    
    # Model hyperparameters
    'dropout_rate': 0.3,
    
    # Training hyperparameters  
    'batch_size': 32,       # Reduce if out of memory
    'num_epochs': 20,       # Start small for testing
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    
    # Data loading
    'num_workers': 4,       # Set to 0 if you have issues
    
    # Checkpointing
    'checkpoint_dir': 'checkpoints',
    'save_every': 5,
    
    # Reproducibility
    'seed': 42
}
```

#### B. Start Training

```bash
python src/train.py
```

**Expected Output:**
```
Configuration:
{
  "train_dir": "...",
  "batch_size": 32,
  ...
}

Using device: cuda
GPU: NVIDIA GeForce RTX 3060

Creating dataloaders...
Loaded ASVspoofDataset from ...
  Total samples: 100
  Bonafide: 8
  Spoof: 92

DataLoaders created:
  Training batches: 3
  Batch size: 32
  Class weights: tensor([6.2500, 0.5435])

Initializing model...
Model parameters: 121,410 trainable / 121,410 total

Starting training for 20 epochs...
======================================================================

Epoch 1/20
----------------------------------------------------------------------
Epoch 1: 100%|████| 3/3 [00:02<00:00,  1.21it/s, loss=0.6234, acc=91.67%]

Train - Loss: 0.6234, Acc: 91.67%, EER: 25.00%
Learning Rate: 0.001000

✓ Best model saved (EER: 25.00%)
...
```

---

### 8. Monitor Training (Optional)

#### Using TensorBoard:
```bash
tensorboard --logdir=logs
```
Then open browser to: http://localhost:6006

#### Check Saved Checkpoints:
```
checkpoints/
├── best_model.pt          # Best model based on validation EER
├── checkpoint_epoch_5.pt
├── checkpoint_epoch_10.pt
└── history.json          # Training metrics
```

---

### 9. Evaluate the Model

Create `src/evaluate.py`:

```python
import torch
import numpy as np
from models.simple_cnn import SimpleCNN
from dataset import ASVspoofDataset

def evaluate_model(checkpoint_path, test_dir, device='cuda'):
    """Evaluate trained model"""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = SimpleCNN(dropout_rate=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load test data
    test_dataset = ASVspoofDataset(test_dir)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )
    
    all_labels = []
    all_scores = []
    all_preds = []
    
    print("Evaluating...")
    with torch.no_grad():
        for specs, labels, _ in test_loader:
            specs = specs.to(device)
            logits, _ = model(specs)
            probs = torch.softmax(logits, dim=1)
            
            all_labels.extend(labels.numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
    
    # Calculate metrics
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Bonafide detected: {np.sum(np.array(all_preds) == 0)}")
    print(f"  Spoof detected: {np.sum(np.array(all_preds) == 1)}")

if __name__ == "__main__":
    evaluate_model(
        checkpoint_path='checkpoints/best_model.pt',
        test_dir=r'D:\SAMPOERNA\Semester 7\Capstone\2019\LA\preprocessed\train'
    )
```

Run:
```bash
python src/evaluate.py
```

---

## Troubleshooting

### Problem: Out of Memory Error
**Solution:**
- Reduce `batch_size` to 16 or 8
- Set `num_workers=0`
- Close other applications

### Problem: CUDA Not Available
**Solution:**
- Check CUDA installation: `nvidia-smi`
- Reinstall PyTorch with correct CUDA version
- Train on CPU (slower but works)

### Problem: Files Not Found
**Solution:**
- Verify paths use raw strings: `r"D:\path\..."`
- Check protocol file format matches exactly
- Ensure FLAC files exist in specified directory

### Problem: Slow Training
**Solution:**
- Reduce `max_samples` during testing
- Use GPU if available
- Increase `num_workers` (but not more than CPU cores)

---

## Next Steps

Once this pipeline works:

1. **Preprocess Full Dataset**
   - Remove `max_samples` parameter
   - Process training, dev, and eval sets

2. **Implement LCNN**
   - Replace SimpleCNN with proper LCNN architecture
   - Add MFM activation function

3. **Add Res2Net and AASIST**
   - Implement full architectures from thesis
   - Create ensemble fusion

4. **Implement ADM**
   - Add Monte Carlo Dropout
   - Implement uncertainty-guided degradation
   - Add adaptive severity scaling

---

## Project Timeline

- **Week 1-2**: Get this baseline working
- **Week 3-4**: Implement full architectures (LCNN, Res2Net, AASIST)
- **Week 5-6**: Implement ADM and ensemble
- **Week 7-8**: Experiments and thesis writing

---

## Quick Test Checklist

- [ ] Environment activated
- [ ] Dependencies installed
- [ ] PyTorch with CUDA working
- [ ] Dataset structure verified
- [ ] Preprocessing completed (100 samples)
- [ ] Dataset test passed
- [ ] Model test passed
- [ ] Training started successfully
- [ ] Checkpoints saving correctly

---

## Support

If you encounter issues:
1. Check error messages carefully
2. Verify all paths are correct
3. Ensure data format matches expected structure
4. Try with smaller batch size or fewer samples first
5. Check GPU memory: `nvidia-smi`

Good luck with your thesis! 🎓
