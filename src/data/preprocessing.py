"""
Audio Preprocessing for ASVspoof 2019 LA Dataset
Converts FLAC audio to Mel-spectrograms following thesis methodology
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import pickle

class AudioPreprocessor:
    """
    Preprocesses audio files into Mel-spectrograms
    Following the methodology in Section 3.2.3
    """
    
    def __init__(self, 
                 target_sr=16000,
                 n_fft=512,
                 hop_length=160,
                 n_mels=80,
                 win_length=400,
                 fmin=20,
                 fmax=8000,
                 max_frames=500):
        self.target_sr = target_sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.win_length = win_length
        self.fmin = fmin
        self.fmax = fmax
        self.max_frames = max_frames
        
        print(f"Initialized AudioPreprocessor:")
        print(f"  Sample Rate: {self.target_sr} Hz")
        print(f"  FFT Size: {self.n_fft}")
        print(f"  Hop Length: {self.hop_length} ({self.hop_length/self.target_sr*1000:.1f}ms)")
        print(f"  Mel Bands: {self.n_mels}")
        print(f"  Frequency Range: {self.fmin}-{self.fmax} Hz")
        print(f"  Max Frames: {self.max_frames} ({self.max_frames*self.hop_length/self.target_sr:.1f}s)")
    
    def load_and_resample(self, audio_path):
        try:
            waveform, sr = librosa.load(audio_path, sr=None)
            if sr != self.target_sr:
                waveform = librosa.resample(waveform, 
                                           orig_sr=sr, 
                                           target_sr=self.target_sr,
                                           res_type='polyphase')
            return waveform, self.target_sr
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None, None
    
    def trim_silence(self, waveform, top_db=30):
        waveform_trimmed, _ = librosa.effects.trim(
            waveform, 
            top_db=top_db,
            frame_length=int(self.target_sr * 0.02),
            hop_length=int(self.target_sr * 0.01)
        )
        return waveform_trimmed
    
    def normalize_amplitude(self, waveform, epsilon=1e-6):
        l2_norm = np.sqrt(np.sum(waveform ** 2))
        normalized = waveform / (l2_norm + epsilon)
        return normalized
    
    def compute_melspectrogram(self, waveform):
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.target_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window='hamming',
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=None)
        mel_spec_db = mel_spec_db.T
        return mel_spec_db
    
    def normalize_per_channel(self, mel_spec):
        mean = np.mean(mel_spec, axis=0, keepdims=True)
        std = np.std(mel_spec, axis=0, keepdims=True)
        normalized = (mel_spec - mean) / (std + 1e-6)
        return normalized
    
    def pad_or_truncate(self, mel_spec):
        current_frames = mel_spec.shape[0]
        if current_frames > self.max_frames:
            fixed_spec = mel_spec[:self.max_frames, :]
        elif current_frames < self.max_frames:
            pad_amount = self.max_frames - current_frames
            fixed_spec = np.pad(
                mel_spec,
                ((0, pad_amount), (0, 0)),
                mode='reflect'
            )
        else:
            fixed_spec = mel_spec
        return fixed_spec
    
    def preprocess_single_file(self, audio_path):
        waveform, sr = self.load_and_resample(audio_path)
        if waveform is None:
            return None
        waveform = self.trim_silence(waveform)
        waveform = self.normalize_amplitude(waveform)
        mel_spec = self.compute_melspectrogram(waveform)
        mel_spec = self.normalize_per_channel(mel_spec)
        mel_spec = self.pad_or_truncate(mel_spec)
        return mel_spec
    
    def preprocess_dataset(self, audio_dir, protocol_file, output_dir, max_samples=None):
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nReading protocol file: {protocol_file}")
        metadata = {}
        
        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    file_id = parts[1]
                    label = parts[-1]
                    metadata[file_id] = 0 if label == 'bonafide' else 1
        
        print(f"Found {len(metadata)} files in protocol")
        
        if max_samples:
            file_ids = list(metadata.keys())[:max_samples]
            metadata = {k: metadata[k] for k in file_ids}
            print(f"Limited to {max_samples} samples for testing")
        
        successful = 0
        failed = 0
        
        print(f"\nPreprocessing audio files...")
        for file_id in tqdm(metadata.keys()):
            audio_path = os.path.join(audio_dir, f"{file_id}.flac")
            if not os.path.exists(audio_path):
                # Try checking if it's just missing .flac extension in path construction
                audio_path_alt = os.path.join(audio_dir, file_id) 
                if os.path.exists(audio_path_alt):
                     audio_path = audio_path_alt
                else:
                    # print(f"Warning: {audio_path} not found") # Commented out to reduce noise
                    failed += 1
                    continue
            
            spec = self.preprocess_single_file(audio_path)
            if spec is not None:
                output_path = os.path.join(output_dir, f"{file_id}.npy")
                np.save(output_path, spec)
                successful += 1
            else:
                failed += 1
        
        print(f"\nPreprocessing complete:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        
        metadata_path = os.path.join(output_dir, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Metadata saved to {metadata_path}")
        return metadata

def main():
    """
    Main execution point with YOUR specific paths
    """
    preprocessor = AudioPreprocessor()
    
    # --- YOUR SPECIFIC PATHS ---
    # 1. Where your raw audio files are located
    audio_dir = r"D:\SAMPOERNA\Semester 7\Capstone\2019\LA\ASVspoof2019_LA_train\flac"
    
    # 2. Where your protocol text file is located
    protocol_file = r"D:\SAMPOERNA\Semester 7\Capstone\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
    
    # 3. Where to save the processed files (inside your project folder)
    output_dir = r"D:\SAMPOERNA\Semester 7\Capstone\Ensemble-Adaptive-Degradation-Module\data\processed\train"
    
    # ---------------------------
    
    print("Starting preprocessing...")
    print(f"Audio Source: {audio_dir}")
    print(f"Protocol: {protocol_file}")
    print(f"Output: {output_dir}")

    # Using 100 samples for the test run.
    # Once this works, you can delete 'max_samples=100' to process everything.
    metadata = preprocessor.preprocess_dataset(
        audio_dir=audio_dir,
        protocol_file=protocol_file,
        output_dir=output_dir,
        max_samples=100 
    )
    
    print("\n✓ Preprocessing complete!")
    print(f"Processed spectrograms saved to: {output_dir}")

if __name__ == "__main__":
    main()