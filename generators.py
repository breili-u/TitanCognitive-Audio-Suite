import torch
import math
import random

class NoiseSynth:
    def __init__(self, sr=16000):
        self.sr = sr

    def colored_noise(self, length, color='pink'):
        """Generates Pink (1/f) or Brown (1/f^2) noise."""
        alpha = 1.0 if color == 'pink' else 2.0
        uneven = length % 2
        X = torch.randn(length // 2 + 1 + uneven) + 1j * torch.randn(length // 2 + 1 + uneven)
        S = torch.arange(len(X)) + 1
        X = X / (S ** (alpha / 2))
        noise = torch.fft.irfft(X)
        if uneven: noise = noise[:-1]
        return noise.unsqueeze(0)

    def mains_hum(self, length, freq=50, harmonics=True):
        """Generates electrical mains hum (50/60Hz)."""
        t = torch.linspace(0, length/self.sr, length)
        hum = torch.sin(2*math.pi*freq*t)
        if harmonics:
            hum += 0.5 * torch.sin(2*math.pi*(freq*3)*t)
            hum += 0.2 * torch.sin(2*math.pi*(freq*5)*t)
        return hum.unsqueeze(0) * 0.1

    def transient_click(self, length):
        """Generates random digital clicks/pops."""
        noise = torch.zeros(length)
        num_clicks = random.randint(1, 5)
        for _ in range(num_clicks):
            idx = random.randint(0, length-1)
            noise[idx] = random.choice([-0.8, 0.8])
        return noise.unsqueeze(0)