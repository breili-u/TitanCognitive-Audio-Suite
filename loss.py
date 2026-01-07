import torch
import torch.nn as nn

class NewtonianLoss(nn.Module):
    """
    A robust loss function designed for high-fidelity audio reconstruction.
    Combines SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) with L1 Loss
    and automatic DC offset correction.
    
    Features:
    - Scale-invariant (volume does not affect the metric).
    - Immune to DC offset (centers signals before comparison).
    - Gated: avoids exploding (NaN) when the target is pure silence.
    """
    def __init__(self, alpha=1.0, beta=0.1, eps=1e-8):
        super().__init__()
        self.alpha = alpha # Weight for SI-SDR
        self.beta = beta   # Weight for L1 (aux/silence)
        self.eps = eps
        self.l1 = nn.L1Loss(reduction='none')

    def sisdr(self, preds, target):
        # Ensure shape [B, T]
        if preds.ndim == 3: preds = preds.squeeze(1)
        if target.ndim == 3: target = target.squeeze(1)
        
        # 1. NEWTONIAN CENTERING (DC offset removal)
        # Crucial for networks that introduce floating bias
        preds = preds - preds.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        
        # 2. Orthogonal Projection
        # alpha = <preds, target> / ||target||^2
        dot_product = (preds * target).sum(dim=-1, keepdim=True)
        target_energy = target.pow(2).sum(dim=-1, keepdim=True) + self.eps
        scale_factor = dot_product / target_energy
        
        target_scaled = scale_factor * target
        noise = preds - target_scaled
        
        # 3. Ratios
        val_s = target_scaled.pow(2).sum(dim=-1)
        val_n = noise.pow(2).sum(dim=-1)
        
        # 4. dB
        return 10 * torch.log10(val_s / (val_n + self.eps) + self.eps)

    def forward(self, preds, target):
        # Compute energy to determine if silence
        target_energy = target.pow(2).mean(dim=-1)
        # Mask: 1 if active audio, 0 if silence (gate)
        active_mask = (target_energy > 1e-5).float()
        
        # SI-SDR (Negativo porque queremos minimizar)
        # Only valid where signal is active
        sisdr_val = -self.sisdr(preds, target)
        
        # L1 Loss (auxiliary for stability and silent regions)
        l1_val = self.l1(preds, target).mean(dim=-1)
        
        # Hybrid logic:
        # - If there is audio: primarily use SI-SDR (high fidelity)
        # - If silence: use pure L1 (absolute denoising)
        loss_per_batch = active_mask * (self.alpha * sisdr_val + self.beta * l1_val) + \
                 (1 - active_mask) * (l1_val * 10.0) # Strong penalty for noise during silence
                         
        return loss_per_batch.mean()