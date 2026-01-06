# TitanCognitive-Audio-Suite
### Neural Audio Restoration with Early-Exit & Newtonian Loss
This framework uses the concept of Multi-Exit Processing and a balanced loss function designed to maximize the SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) while preserving the integrity of the original signal through a surgical identity penalty.

# Dataset Structure
The system automatically downloads and pre-processes critical data sources for training:

LibriSpeech (dev-clean-2): Clean speech corpus for supervised training.

ESC-50: Environmental sound dataset for realistic noise injection.
DSP Helpers: Includes `SignalBrutalizer` (gain modulation and clipping) and `RoomSimulator` (RT60 reverberation convolution).

# Evolutionary Curriculum
The dataset evolves dynamically using the `set_phase(epoch)` method. This allows the model to first learn to clean the data (Phases 0-1) and then learn to discriminate (Phases 2-3).
| Phase | Epochs | Name | Main Focus |
|:-----------:|:------------:|:------------:|:------------:|
| 0 | 0-10  | Childhood   | Basic cases: clean voice vs. white noise |
| 1 | 10-40 | Adolescence | Rehabilitation and handling of silences  |
| 2 | 40-55 | SPECIALIST  | High SNR Focus. Maximizing SI-SDR in nearly clean signals |
| 3 | 55+ | Maturity | Balanced mix of identity, noise, and ambiguity |

# Architecture of Loss (Titan Newtonian)
The `TitanNewtonianLoss` class implements a hybrid optimization strategy that combines audio quality and confidence calibration. 
1. Audio Loss (Hybrid SI-SDR):
   For signals with active energy (E > 10⁻⁵), the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) is used. For silent segments, the model automatically switches to L1 Loss to avoid unstable gradients.
   
  $$SI-SDR(y, \hat{y}) = 10 \log_{10} \left( \frac{\|| \alpha y \||^2}{\|| \alpha y - \hat{y} \||^2} \right) \text{ Where } \alpha = \frac{\hat{y}^T y}{y^T y}$$

  2. Multi-Exit Balancing: The gradient is distributed between the early output (`exit_1`) and the final output (`exit_final`) with a 60/40 ratio to ensure that the lightweight processing is extremely accurate.
     
  3. Identity Penalty: To protect the signal when the confidence target is high ($$>0.95$$), an additional penalty for perturbation is applied: 

 $$L_{identity} = 0.5 \cdot |\hat{y} - y| \cdot \mathbb{1}_{conf > 0.95}$$

  4. Confidence Brier Score & Arrogance Penalty: The model estimates its own confidence. If the model is "overconfident" (predicts high confidence >0.8, but the audio error is large), the penalty is doubled.
     
 $$L_{total} = (0.6 L_{exit1} + 0.4 L_{final}) + 0.05 L_{confidence}$$
# Usage and Configuration
Installing dependencies

`pip install torch torchaudio numpy torchcodec`

Dataset initialization
```
from titan import TitanDataset, TitanNewtonianLoss

# The dataset handles downloads and IR caching automatically
dataset = TitanDatasetV15(max_len=16384, items_limit=10000)

# Loss function configuration
criterion = TitanNewtonianLoss()
```
Example of a Training Loop
```
for epoch in range(100):
    dataset.set_phase(epoch)
    for batch in dataloader:
        optimizer.zero_grad()
        
        # El modelo debe retornar un diccionario con salidas múltiples
        outputs = model(batch["x"]) 
        
        loss = criterion(outputs, batch, epoch, dataset.phase)
        loss.backward()
        optimizer.step()
```
**Technical Notes**
  
  Zero-mean centering: Applied internally in the SI-SDR calculation for greater stability.

  Mild Brutalizer: In advanced phases (2 and 3), the identity signal is handled with care to avoid unnecessary artifacts.
