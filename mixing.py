import torch
import math

def calculate_rms(tensor):
    """Computes a safe Root Mean Square (RMS)."""
    return tensor.norm(p=2) / (math.sqrt(tensor.numel()) + 1e-9)

def mix_signals(clean, noise, snr_db):
    """
    Mixes two signals to achieve a target SNR in dB.
    Args:
        clean (Tensor): Clean audio [C, T] or [T]
        noise (Tensor): Noise [C, T] or [T]
        snr_db (float): Desired Signal-to-Noise Ratio (e.g., 10.0, -5.0)
    """
    clean_rms = calculate_rms(clean)
    noise_rms = calculate_rms(noise)
    
    if noise_rms < 1e-9: 
        return clean
    
    # Formula: target_noise = clean / 10^(snr/20)
    target_noise_rms = clean_rms / (10**(snr_db/20))
    scale = target_noise_rms / (noise_rms + 1e-9)
    
    return clean + noise * scale