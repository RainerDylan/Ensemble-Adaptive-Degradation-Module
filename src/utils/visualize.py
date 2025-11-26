import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys

def view_all(data_dir):
    # Find .npy files
    files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    
    if not files:
        print(f" [!] No .npy files found in {data_dir}")
        return
    
    print("\n" + "="*60)
    print(f" VIEWING DATA: {os.path.basename(data_dir)}")
    print(f" Found {len(files)} files.")
    print(" CONTROLS: Close the image window to see the next file.")
    print("           Press Ctrl+C in terminal to Exit.")
    print("="*60)
    
    for i, f in enumerate(files):
        try:
            data = np.load(f)
            filename = os.path.basename(f)
            
            plt.figure(figsize=(12, 5))
            
            # Case 1: Spectrogram (2D Array: Time x Freq)
            if len(data.shape) == 2:
                # Transpose for visualization so Time is X-axis
                plt.imshow(data.T, aspect='auto', origin='lower', cmap='magma')
                plt.colorbar(format='%+2.0f dB')
                plt.title(f"Spectrogram [{i+1}/{len(files)}]: {filename}")
                plt.ylabel("Frequency (Mel Bins)")
                plt.xlabel("Time Frames")
            
            # Case 2: Raw Waveform (1D Array or 2D with 1 channel)
            else:
                # Ensure it's flat for plotting
                wave_data = data.flatten()
                plt.plot(wave_data, linewidth=0.5, color='blue')
                plt.title(f"Raw Waveform [{i+1}/{len(files)}]: {filename}")
                plt.ylabel("Amplitude (Normalized)")
                plt.xlabel("Samples")
                plt.ylim(-1.1, 1.1) # Fixed Y-axis to show normalization
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            print(f" [>] Displaying {filename}...")
            plt.show() # Script pauses here until you close window
            
        except KeyboardInterrupt:
            print("\n [!] Exiting viewer...")
            sys.exit()
        except Exception as e:
            print(f" [!] Error reading {f}: {e}")

def main():
    # 1. Find all processed folders
    base_dir = r"data/processed"
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} does not exist. Run preprocessing first!")
        return

    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    if not folders:
        print("No data folders found.")
        return

    # 2. Ask User to Select
    print("\nSelect data to visualize:")
    for idx, folder in enumerate(folders):
        print(f" [{idx + 1}] {folder}")

    try:
        choice = int(input("\nEnter number: ")) - 1
        if 0 <= choice < len(folders):
            selected_folder = os.path.join(base_dir, folders[choice])
            view_all(selected_folder)
        else:
            print("Invalid selection.")
    except ValueError:
        print("Please enter a number.")

if __name__ == "__main__":
    main()