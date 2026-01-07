import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from pathlib import Path
import random

def safe_load_audio(file_path, target_sr=16000, target_len=None, force_mono=True):
    """
    Safely loads, resamples, and adjusts an audio file.
    
    Args:
        file_path (str/Path): Path to the audio file.
        target_sr (int): Desired sample rate.
        target_len (int, optional): Fixed length in samples. If None, returns original length.
        force_mono (bool): If True, averages channels to return [1, T].
        
    Returns:
        torch.Tensor: Processed audio [Channels, Time] or None on failure.
    """
    try:
        path = str(file_path)
        # Fast load (backend-agnostic)
        waveform, sr = torchaudio.load(path)
        
        # 1. Resampling
        if sr != target_sr:
            resampler = T.Resample(sr, target_sr)
            waveform = resampler(waveform)
            
        # 2. Mono / Stereo Fix
        if force_mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif not force_mono and waveform.shape[0] == 1:
            # If stereo requested but input is mono, duplicate
            waveform = waveform.repeat(2, 1)
            
        # 3. Padding / Cropping (if a target length is specified)
        if target_len is not None:
            current_len = waveform.shape[1]
            
            if current_len > target_len:
                # Random crop (for training)
                start = random.randint(0, current_len - target_len)
                waveform = waveform[:, start:start + target_len]
            elif current_len < target_len:
                # Pad with zeros (or reflect)
                padding = target_len - current_len
                # Right-pad
                waveform = F.pad(waveform, (0, padding), "constant", 0)
                
        return waveform

    except Exception as e:
        print(f"Error {file_path}: {e}")
        return None

def scan_audio_files(directory, extensions=['.wav', '.flac', '.mp3']):
    """
    Recursively scans a directory for valid audio files.
    """
    files = []
    path = Path(directory)
    if not path.exists():
        return []
        
    for ext in extensions:
        # Case-insensitive search would be ideal, but glob is simple
        files.extend(list(path.rglob(f"*{ext}")))
        files.extend(list(path.rglob(f"*{ext.upper()}")))
        
    return sorted([str(f) for f in files])

def db_to_linear(db):
    """Converts decibels to linear scale."""
    return 10 ** (db / 20.0)

def linear_to_db(scale):
    """Converts linear scale to decibels."""
    return 20.0 * torch.log10(scale + 1e-9)