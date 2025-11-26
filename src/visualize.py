import numpy as np
import matplotlib.pyplot as plt
import os
import random

def view_spectrogram(data_dir):
    # 1. Find all .npy files
    files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    if not files:
        print(f"No .npy files found in {data_dir}")
        return

    # 2. Pick a random file to show
    random_file = random.choice(files)
    file_path = os.path.join(data_dir, random_file)
    
    # 3. Load the data
    # Shape is usually (500, 80) -> (Time, Frequency)
    spectrogram = np.load(file_path)
    
    print(f"Displaying: {random_file}")
    print(f"Shape: {spectrogram.shape}")
    print(f"Min value: {spectrogram.min():.2f}")
    print(f"Max value: {spectrogram.max():.2f}")

    # 4. Plot it
    plt.figure(figsize=(10, 4))
    # We transpose (.T) so Time is on X-axis and Frequency on Y-axis
    plt.imshow(spectrogram.T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Mel-Spectrogram: {random_file}")
    plt.ylabel("Mel Frequency Bands")
    plt.xlabel("Time Frames")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Update this path to match where preprocessing saved your files
    processed_dir = r"D:\SAMPOERNA\Semester 7\Capstone\Ensemble-Adaptive-Degradation-Module\data\processed\train"
    
    if os.path.exists(processed_dir):
        view_spectrogram(processed_dir)
    else:
        print(f"Error: Directory not found at {processed_dir}")