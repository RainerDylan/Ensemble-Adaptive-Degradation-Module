import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def view_all(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    if not files: return print("No files found.")
    
    print(f"Viewing {len(files)} files. Close window for next.")
    
    for i, f in enumerate(files):
        data = np.load(f)
        plt.figure(figsize=(10, 4))
        if len(data.shape) == 2:
            plt.imshow(data.T, aspect='auto', origin='lower', cmap='magma')
            plt.title(f"Spectrogram: {os.path.basename(f)}")
        else:
            plt.plot(data)
            plt.title(f"Raw Waveform: {os.path.basename(f)}")
            plt.ylim(-1, 1)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Point to your data folder here
    path = r"D:\SAMPOERNA\Semester 7\Capstone\Ensemble-Adaptive-Degradation-Module\data\processed\train_spect"
    view_all(path)