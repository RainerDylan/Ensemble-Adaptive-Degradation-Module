"""
Evaluation script for the trained model using the Test/Eval dataset.
"""
import torch
import torch.nn as nn
import numpy as np
import os
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import ASVspoofDataset
from models.simple_cnn import SimpleCNN

def calculate_eer(labels, scores):
    """Calculate Equal Error Rate (EER)"""
    if len(labels) == 0: return 0, 0
    
    # Sort scores and labels together
    sorted_indices = np.argsort(scores)
    labels = np.array(labels)[sorted_indices]
    scores = np.array(scores)[sorted_indices]
    
    n_spoof = np.sum(labels == 1)
    n_bonafide = np.sum(labels == 0)
    
    if n_spoof == 0 or n_bonafide == 0: 
        print("Warning: Only one class present in dataset. Cannot calculate EER.")
        return 0, 0
    
    far = []
    frr = []
    
    # Efficiently calculate FAR/FRR
    # (Simplified implementation for clarity)
    for i in range(0, len(scores), max(1, len(scores)//1000)): # Sample thresholds
        threshold = scores[i]
        false_accepts = np.sum((labels[:i] == 1)) # Spoofs below threshold classified as Bonafide (if score is 'bonafide confidence')
        # Note: Our model outputs logits. 
        # If index 1 is spoof, higher score = more likely spoof.
        # Let's assume scores = Spoof Probability.
        
        # Threshold logic: Score > Threshold => Spoof
        # FAR: Bonafide classified as Spoof (Score > Threshold)
        # FRR: Spoof classified as Bonafide (Score <= Threshold)
        
        # Let's use standard logic:
        # scores are Spoof Probabilities.
        # FAR = Bonafide samples with Score > Threshold
        # FRR = Spoof samples with Score <= Threshold
        
        # Current loop iterates sorted scores. 
        # scores[i] is the threshold.
        # Samples below i have Score <= Threshold.
        # Samples above i have Score > Threshold.
        
        current_far = np.sum(labels[i:] == 0) / n_bonafide
        current_frr = np.sum(labels[:i] == 1) / n_spoof
        
        far.append(current_far)
        frr.append(current_frr)
    
    far = np.array(far)
    frr = np.array(frr)
    
    # Find point where FAR approx equals FRR
    abs_diff = np.abs(far - frr)
    min_index = np.argmin(abs_diff)
    eer = (far[min_index] + frr[min_index]) / 2
    
    return eer * 100

def evaluate(model_path, eval_data_dir):
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Load Dataset
    print(f"Loading evaluation data from: {eval_data_dir}")
    if not os.path.exists(os.path.join(eval_data_dir, 'metadata.pkl')):
        print("Error: Metadata not found. Did you run src/preprocess_eval.py?")
        return

    dataset = ASVspoofDataset(eval_data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"Found {len(dataset)} samples.")

    # 3. Load Model
    print(f"Loading model from: {model_path}")
    model = SimpleCNN(dropout_rate=0.3).to(device)
    
    if not os.path.exists(model_path):
        print("Error: Checkpoint not found. Train the model first!")
        return

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 4. Run Inference
    all_labels = []
    all_spoof_scores = []
    correct = 0
    total = 0
    
    print("Running evaluation...")
    with torch.no_grad():
        for spectrograms, labels, _ in tqdm(dataloader):
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            
            logits, _ = model(spectrograms)
            probs = torch.softmax(logits, dim=1)
            
            # Predictions for accuracy (Class 1 is Spoof)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Collect scores for EER (Probability of being Spoof)
            # Class 1 = Spoof
            spoof_probs = probs[:, 1].cpu().numpy()
            true_labels = labels.cpu().numpy()
            
            all_spoof_scores.extend(spoof_probs)
            all_labels.extend(true_labels)

    # 5. Calculate Metrics
    accuracy = 100 * correct / total
    eer = calculate_eer(all_labels, all_spoof_scores)
    
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"EER:      {eer:.2f}%")
    print("-" * 40)
    print(f"Total Samples: {total}")
    print("="*40 + "\n")

if __name__ == "__main__":
    # Configuration
    MODEL_CHECKPOINT = "checkpoints/best_model.pt"
    EVAL_DATA_DIR = r"D:\SAMPOERNA\Semester 7\Capstone\Ensemble-Adaptive-Degradation-Module\data\processed\eval"
    
    evaluate(MODEL_CHECKPOINT, EVAL_DATA_DIR)