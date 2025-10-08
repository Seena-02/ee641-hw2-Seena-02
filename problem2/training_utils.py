"""
Training implementations for hierarchical VAE with posterior collapse prevention.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

# KL annealing schedule
def kl_annealing_schedule(epoch, method):
    """
    TODO: Implement KL annealing schedule
    Start with beta ≈ 0, gradually increase to 1.0
    Consider cyclical annealing for better results
    """
    period = 100 // 4

    return min(1.0, (epoch % period) / (period * 0.5))

def temperature_annealing_schedule(epoch, total_epochs=100, start_temp=2.0, end_temp=0.5):
    """
    Temperature annealing schedule for discrete sampling outputs.
    Lowers temperature over epochs for sharper outputs.
    
    Args:
        epoch: current epoch
        total_epochs: total epochs
        start_temp: starting temperature
        end_temp: final temperature
        
    Returns:
        temperature: float
    """
    fraction = min(1.0, epoch / total_epochs)
    temp = start_temp * (1 - fraction) + end_temp * fraction
    return temp

def train_hierarchical_vae(model, data_loader, num_epochs=100, device='cuda'):
    """
    Train hierarchical VAE with KL annealing and other tricks.
    Implements several techniques to prevent posterior collapse:
    1. KL annealing (gradual beta increase)
    2. Free bits (minimum KL per dimension)
    3. Temperature annealing for discrete outputs
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # KL annealing schedule
    def kl_anneal_schedule(epoch, cyclical=False, M=10):
        """
        Start with beta ≈ 0, gradually increase to 1.0.
        Cyclical schedule: M cycles of ramp up.
        """
        if cyclical:
            cycle_length = num_epochs // M
            epoch_in_cycle = epoch % cycle_length
            beta = min(1.0, epoch_in_cycle / (cycle_length // 2))
            return beta
        else:
            return min(1.0, epoch / (num_epochs // 4))

    free_bits = 0.5 # Minimum nats per latent dimension
    history = defaultdict(list)

    for epoch in range(num_epochs):
        model.train()
        beta = kl_anneal_schedule(epoch, cyclical=True, M=6)  # Use cyclical annealing as default
        for batch_idx, batch in enumerate(data_loader):
            patterns = batch[0] if isinstance(batch, (list, tuple)) else batch
            patterns = patterns.to(device)
            optimizer.zero_grad()

            # Forward pass through hierarchical VAE
            recon, loss, mu_low, logvar_low, mu_high, logvar_high = model(patterns, beta=beta)

            # Compute KL divergences
            kl_low = -0.5 * (1 + logvar_low - mu_low.pow(2) - logvar_low.exp())
            kl_high = -0.5 * (1 + logvar_high - mu_high.pow(2) - logvar_high.exp())
            # Apply free bits (floor to minimum per latent dim)
            kl_low = torch.clamp(kl_low, min=free_bits).sum(dim=1).mean()
            kl_high = torch.clamp(kl_high, min=free_bits).sum(dim=1).mean()

            kl_loss = kl_low + kl_high

            # Reconstruction loss (already in model loss)
            recon_loss = nn.functional.binary_cross_entropy(recon, patterns.float(), reduction='sum') / patterns.size(0)

            total_loss = recon_loss + beta * kl_loss
            total_loss.backward()
            optimizer.step()

            history['loss'].append(total_loss.item())
            history['recon_loss'].append(recon_loss.item())
            history['kl_low'].append(kl_low.item())
            history['kl_high'].append(kl_high.item())
            history['beta'].append(beta)

        print(f"Epoch {epoch + 1}, Loss: {np.mean(history['loss'][-len(data_loader):]):.4f}  Beta: {beta:.3f}")

    return history

def sample_diverse_patterns(model, n_styles=5, n_variations=10, device='cuda'):
    """
    Generate diverse drum patterns using the hierarchy.
    1. Sample n_styles from z_high prior
    2. For each style, sample n_variations from conditional p(z_low|z_high)
    3. Decode to patterns
    4. Organize in grid showing style consistency
    """
    model.eval()
    styles = torch.randn(n_styles, model.z_high_dim).to(device)
    all_patterns = []
    for s_idx in range(n_styles):
        style_code = styles[s_idx].unsqueeze(0).repeat(n_variations, 1)
        # For each style, sample different z_low from N(0, I) and optionally condition on z_high
        z_low = torch.randn(n_variations, model.z_low_dim).to(device)
        # Decode patterns
        logits = model.decode_hierarchy(style_code, z_low)
        patt = torch.sigmoid(logits).detach().cpu().numpy()
        all_patterns.append(patt)
    return np.array(all_patterns)  # Shape: [n_styles, n_variations, 16, 9]

def analyze_posterior_collapse(model, data_loader, device='cuda'):
    """
    Diagnose which latent dimensions are being used.
    1. Encode validation data
    2. Compute KL divergence per dimension
    3. Identify collapsed dimensions (KL ≈ 0)
    4. Return utilization statistics
    """
    model.eval()
    kl_lows = []
    kl_highs = []
    with torch.no_grad():
        for batch in data_loader:
            patterns = batch[0] if isinstance(batch, (list, tuple)) else batch
            patterns = patterns.to(device)
            mu_low, logvar_low, mu_high, logvar_high, _ = model.encode_hierarchy(patterns)
            kl_low = -0.5 * (1 + logvar_low - mu_low.pow(2) - logvar_low.exp())  # [batch, z_low_dim]
            kl_high = -0.5 * (1 + logvar_high - mu_high.pow(2) - logvar_high.exp())  # [batch, z_high_dim]
            kl_lows.append(kl_low.cpu())
            kl_highs.append(kl_high.cpu())
    kl_low_mean = torch.cat(kl_lows, 0).mean(dim=0).numpy()
    kl_high_mean = torch.cat(kl_highs, 0).mean(dim=0).numpy()
    low_collapsed = np.sum(kl_low_mean < 0.1)
    high_collapsed = np.sum(kl_high_mean < 0.1)
    print("Low-level latent: collapsed dimensions:", low_collapsed, "/", len(kl_low_mean))
    print("High-level latent: collapsed dimensions:", high_collapsed, "/", len(kl_high_mean))
    return dict(
        kl_low_mean=kl_low_mean,
        kl_high_mean=kl_high_mean,
        low_collapsed=low_collapsed,
        high_collapsed=high_collapsed
    )
