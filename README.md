# Titan Data: Engineered for Chaos
**Titan Data** is a robust, procedural audio augmentation library designed to train State-of-the-Art (SOTA) denoising and restoration models. Unlike static datasets, Titan generates infinite variations of acoustic conditions, forcing neural networks to learn structural reconstruction rather than memorization.

## The dataset
Create an infinite data loader that mixes clean speech with real noise, reverb, and digital degradation on the fly.

```bash
from titan_data import TitanAudioDataset, DataLoader

# Initialize robust dataset
dataset = TitanAudioDataset(
    clean_path="./data/LibriSpeech",
    noise_path="./data/ESC-50",
    sample_rate=16000,
    duration=3.0
)

# Optional: Set Curriculum Learning (Easy -> Hard)
dataset.set_curriculum(snr_range=(10, 30), prob_real_noise=0.2)

loader = DataLoader(dataset, batch_size=32, num_workers=4)

for noisy, clean in loader:
    # noisy: [B, 1, 48000] - Distorted, Noisy, Reverb
    # clean: [B, 1, 48000] - Pristine Reference
    pass
```
## The Newtonian Loss
Titan Data includes NewtonianLoss, a mathematically stable objective function designed for high-fidelity audio reconstruction.
```bash
from titan_data import NewtonianLoss

# Initialize
criterion = NewtonianLoss()

# Training loop
loss = criterion(model_output, clean_target)
loss.backward()
```
## Features
- Procedural Noise Synthesis: Generates Pink, Brown, and Mains Hum (50/60Hz) noise mathematically.

- Room Simulator: Convolves audio with random Impulse Responses (RT60) to simulate acoustic spaces.

- Signal Brutalizer: Simulates clipping, cheap microphones, and bad EQ.

- Precise SNR Mixing: Mixes clean and noise signals based on exact dB calculations, not linear amplitude.
