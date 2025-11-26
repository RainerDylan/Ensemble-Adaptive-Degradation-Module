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
        """
        Initialize preprocessor with parameters from thesis
        
        Args:
            target_sr: Target sampling rate (16kHz)
            n_fft: FFT size (512 points)
            hop_length: Hop length in samples (10ms = 160 samples at 16kHz)
            n_mels: Number of mel filters (80)
            win_length: Window length (25ms = 400 samples at 16kHz)
            fmin: Minimum frequency (20 Hz)
            fmax: Maximum frequency (8000 Hz)
            max_frames: Fixed temporal dimension (500 frames = 5 seconds)
        """
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
        """
        Step 1: Load audio and resample to 16kHz
        Uses polyphase resampling to minimize aliasing
        
        Args:
            audio_path: Path to FLAC file
            
        Returns:
            waveform: numpy array of audio samples
            sr: Sampling rate (should be 16000)
        """
        try:
            # Load audio file
            waveform, sr = librosa.load(audio_path, sr=None)
            
            # Resample if needed using polyphase filter
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
        """
        Step 2: Remove leading/trailing silence
        Uses energy-based threshold at 30dB below peak
        
        Args:
            waveform: Audio samples
            top_db: Threshold in dB below peak amplitude
            
        Returns:
            trimmed_waveform: Audio with silence removed
        """
        # Trim silence
        waveform_trimmed, _ = librosa.effects.trim(
            waveform, 
            top_db=top_db,
            frame_length=int(self.target_sr * 0.02),  # 20ms window
            hop_length=int(self.target_sr * 0.01)     # 10ms hop
        )
        
        return waveform_trimmed
    
    def normalize_amplitude(self, waveform, epsilon=1e-6):
        """
        Step 3: L2 normalization to unit energy
        Prevents models from learning level-based cues
        
        Args:
            waveform: Audio samples
            epsilon: Small constant to prevent division by zero
            
        Returns:
            normalized_waveform: L2-normalized audio
        """
        # Calculate L2 norm
        l2_norm = np.sqrt(np.sum(waveform ** 2))
        
        # Normalize with epsilon for numerical stability
        normalized = waveform / (l2_norm + epsilon)
        
        return normalized
    
    def compute_melspectrogram(self, waveform):
        """
        Step 4: Convert to Mel-spectrogram
        
        Args:
            waveform: Normalized audio samples
            
        Returns:
            mel_spec: Mel-spectrogram in dB scale (time x frequency)
        """
        # Compute mel-spectrogram
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
            power=2.0  # Power spectrum
        )
        
        # Convert to log scale (dB)
        # Add small epsilon before log to avoid log(0)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=None)
        
        # Transpose to (time, frequency) format
        mel_spec_db = mel_spec_db.T
        
        return mel_spec_db
    
    def normalize_per_channel(self, mel_spec):
        """
        Step 5: Z-normalization per frequency channel
        Removes frequency-dependent energy variations
        
        Args:
            mel_spec: Mel-spectrogram (time x frequency)
            
        Returns:
            normalized_spec: Z-normalized spectrogram
        """
        # Calculate mean and std per frequency channel (across time)
        mean = np.mean(mel_spec, axis=0, keepdims=True)
        std = np.std(mel_spec, axis=0, keepdims=True)
        
        # Normalize with small epsilon to prevent division by zero
        normalized = (mel_spec - mean) / (std + 1e-6)
        
        return normalized
    
    def pad_or_truncate(self, mel_spec):
        """
        Step 6: Fixed-length padding/truncation to 500 frames
        Uses reflection padding to preserve spectral continuity
        
        Args:
            mel_spec: Variable-length mel-spectrogram (time x frequency)
            
        Returns:
            fixed_spec: Fixed-length spectrogram (500 x 80)
        """
        current_frames = mel_spec.shape[0]
        
        if current_frames > self.max_frames:
            # Truncate: take first 500 frames
            fixed_spec = mel_spec[:self.max_frames, :]
        elif current_frames < self.max_frames:
            # Pad: use reflection padding
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
        """
        Complete preprocessing pipeline for one audio file
        
        Args:
            audio_path: Path to FLAC file
            
        Returns:
            spectrogram: Preprocessed mel-spectrogram (500 x 80)
                        or None if preprocessing failed
        """
        # Step 1: Load and resample
        waveform, sr = self.load_and_resample(audio_path)
        if waveform is None:
            return None
        
        # Step 2: Trim silence
        waveform = self.trim_silence(waveform)
        
        # Step 3: Normalize amplitude
        waveform = self.normalize_amplitude(waveform)
        
        # Step 4: Compute mel-spectrogram
        mel_spec = self.compute_melspectrogram(waveform)
        
        # Step 5: Per-channel normalization
        mel_spec = self.normalize_per_channel(mel_spec)
        
        # Step 6: Pad or truncate to fixed length
        mel_spec = self.pad_or_truncate(mel_spec)
        
        return mel_spec
    
    def preprocess_dataset(self, 
                          audio_dir, 
                          protocol_file, 
                          output_dir,
                          max_samples=None):
        """
        Preprocess entire dataset
        
        Args:
            audio_dir: Directory containing FLAC files
            protocol_file: Path to protocol file (CM_protocol_train.trn)
            output_dir: Directory to save preprocessed spectrograms
            max_samples: Optional limit on number of samples (for testing)
            
        Returns:
            metadata: Dictionary with file_id -> label mapping
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Read protocol file to get labels
        print(f"\nReading protocol file: {protocol_file}")
        metadata = {}
        
        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Protocol format: speaker_id audio_file_id attack_type label
                if len(parts) >= 4:
                    file_id = parts[1]
                    label = parts[-1]  # 'bonafide' or 'spoof'
                    metadata[file_id] = 0 if label == 'bonafide' else 1
        
        print(f"Found {len(metadata)} files in protocol")
        
        # Limit samples if specified (for quick testing)
        if max_samples:
            file_ids = list(metadata.keys())[:max_samples]
            metadata = {k: metadata[k] for k in file_ids}
            print(f"Limited to {max_samples} samples for testing")
        
        # Preprocess each file
        successful = 0
        failed = 0
        
        print(f"\nPreprocessing audio files...")
        for file_id in tqdm(metadata.keys()):
            audio_path = os.path.join(audio_dir, f"{file_id}.flac")
            
            if not os.path.exists(audio_path):
                print(f"Warning: {audio_path} not found")
                failed += 1
                continue
            
            # Preprocess
            spec = self.preprocess_single_file(audio_path)
            
            if spec is not None:
                # Save as numpy file
                output_path = os.path.join(output_dir, f"{file_id}.npy")
                np.save(output_path, spec)
                successful += 1
            else:
                failed += 1
        
        print(f"\nPreprocessing complete:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        
        # Save metadata
        metadata_path = os.path.join(output_dir, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Metadata saved to {metadata_path}")
        
        return metadata


def main():
    """
    Example usage for preprocessing ASVspoof 2019 LA training data
    """
    # Initialize preprocessor
    preprocessor = AudioPreprocessor()
    
    # Paths - UPDATE THESE TO YOUR ACTUAL PATHS
    audio_dir = r"D:\SAMPOERNA\Semester 7\Capstone\2019\LA\ASVspoof2019_LA_train\flac"
    protocol_file = r"D:\SAMPOERNA\Semester 7\Capstone\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
    output_dir = r"D:\SAMPOERNA\Semester 7\Capstone\2019\LA\preprocessed\train"
    
    # For initial testing, process only 100 samples
    # Remove max_samples parameter to process full dataset
    metadata = preprocessor.preprocess_dataset(
        audio_dir=audio_dir,
        protocol_file=protocol_file,
        output_dir=output_dir,
        max_samples=100  # Start small for testing
    )
    
    print("\n✓ Preprocessing complete!")
    print(f"Processed spectrograms saved to: {output_dir}")


if __name__ == "__main__":
    main()
