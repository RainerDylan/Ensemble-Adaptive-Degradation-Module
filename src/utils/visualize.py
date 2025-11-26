import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def view_all_data(data_dir):
    # Find all .npy files recursively
    files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    
    if not files:
        print(f"No .npy files found in {data_dir}")
        return

    print(f"\nFound {len(files)} files. Loading viewer...")
    print("Controls: [Close Window] = Next Image, [Ctrl+C] in terminal = Quit")

    for i, file_path in enumerate(files):
        try:
            data = np.load(file_path)
            
            plt.figure(figsize=(12, 6))
            filename = os.path.basename(file_path)
            
            # Check if Spectrogram (2D) or Raw Audio (1D)
            if len(data.shape) == 2: 
                # Spectrogram: (Time, Freq)
                plt.imshow(data.T, aspect='auto', origin='lower', cmap='magma')
                plt.colorbar(format='%+2.0f dB')
                plt.title(f"[{i+1}/{len(files)}] Spectrogram: {filename}")
                plt.ylabel("Frequency")
                plt.xlabel("Time")
            elif len(data.shape) == 1:
                # Raw Audio: (Time,)
                plt.plot(data, linewidth=0.5, color='blue')
                plt.title(f"[{i+1}/{len(files)}] Raw Waveform: {filename}")
                plt.xlabel("Samples")
                plt.ylabel("Amplitude")
                plt.ylim(-1.1, 1.1)
            else:
                print(f"Skipping {filename}: Unknown shape {data.shape}")
                continue

            plt.tight_layout()
            print(f"Displaying {filename}...")
            plt.show() # Script pauses here until you close the window
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    # UPDATE THIS PATH to point to your processed data
    # Example: "data/processed/train" (Spectrograms) or "data/processed/train_raw" (Raw)
    processed_dir = r"D:\SAMPOERNA\Semester 7\Capstone\Ensemble-Adaptive-Degradation-Module\data\processed\train"
    
    view_all_data(processed_dir)