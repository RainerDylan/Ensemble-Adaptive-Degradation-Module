"""
Preprocessing script specifically for ASVspoof 2019 LA EVALUATION data
"""
import os
import sys

# Add src to path to find modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import AudioPreprocessor

def main():
    preprocessor = AudioPreprocessor()
    
    # --- EVALUATION PATHS ---
    # 1. Eval Audio Source
    # Based on your input: D:\SAMPOERNA\Semester 7\Capstone\2019\LA\ASVspoof2019_LA_eval
    # Standard structure usually has a 'flac' folder inside.
    audio_dir = r"D:\SAMPOERNA\Semester 7\Capstone\2019\LA\ASVspoof2019_LA_eval\flac"
    
    # 2. Eval Protocol File
    # Note: Eval protocols usually end in .trl.txt (not .trn.txt)
    protocol_file = r"D:\SAMPOERNA\Semester 7\Capstone\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.eval.trl.txt"
    
    # 3. Output for Processed Eval Data
    output_dir = r"D:\SAMPOERNA\Semester 7\Capstone\Ensemble-Adaptive-Degradation-Module\data\processed\eval"
    
    # ------------------------
    
    print("Starting EVALUATION preprocessing...")
    print(f"Audio: {audio_dir}")
    print(f"Protocol: {protocol_file}")
    print(f"Output: {output_dir}")

    if not os.path.exists(audio_dir):
        print(f"ERROR: Audio directory not found: {audio_dir}")
        return
    if not os.path.exists(protocol_file):
        print(f"ERROR: Protocol file not found: {protocol_file}")
        return

    # Process the dataset
    # We remove 'max_samples' to process ALL data, or keep it small (e.g. 500) for a quick test
    metadata = preprocessor.preprocess_dataset(
        audio_dir=audio_dir,
        protocol_file=protocol_file,
        output_dir=output_dir,
        max_samples=500  # Change to None to process EVERYTHING (takes time!)
    )
    
    print("\n✓ Eval preprocessing complete!")
    print(f"Saved to: {output_dir}")

if __name__ == "__main__":
    main()