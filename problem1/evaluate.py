"""
Analysis and evaluation experiments for trained GAN models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from provided.metrics import _simple_letter_classifier

# Ensure results folder exists
save_dir = 'results/visualizations'
os.makedirs(save_dir, exist_ok=True)

def interpolation_experiment(generator, device='mps', n_steps=10):
    """
    Interpolate between latent codes to generate smooth transitions.

    Args:
        generator: Trained generator
        device: Device
        n_steps: Number of interpolation steps between codes
    """
    generator.to(device)
    generator.eval()

    # Sample two random latent codes
    z1 = torch.randn(1, generator.z_dim, device=device)
    z2 = torch.randn(1, generator.z_dim, device=device)

    interpolation_images = []
    with torch.no_grad():
        for alpha in np.linspace(0, 1, n_steps):
            z_interp = (1 - alpha) * z1 + alpha * z2
            img = generator(z_interp).squeeze().cpu()
            img = (img + 1) / 2  # Normalize for display
            interpolation_images.append(img)

    # Plot the interpolated images
    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 2, 2))
    for i, img in enumerate(interpolation_images):
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].axis('off')
    plt.suptitle('Latent Space Interpolation')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'interpolation.png'), dpi=150)
    plt.show()


def style_consistency_experiment(conditional_generator, device='mps'):
    """
    Test if conditional GAN maintains style across letters.

    Args:
        conditional_generator: Trained conditional GAN generator
        device: Device
    """
    conditional_generator.to(device)
    conditional_generator.eval()

    # Fix a latent code
    z = torch.randn(1, conditional_generator.z_dim, device=device)

    images = []
    with torch.no_grad():
        for i in range(26):
            label = torch.zeros(1, 26, device=device)
            label[0, i] = 1
            img = conditional_generator(z).squeeze().cpu()
            img = (img + 1) / 2
            images.append(img)

    # Plot images A-Z for the fixed style
    fig, axes = plt.subplots(2, 13, figsize=(26, 4))
    axes = axes.flatten()
    for i, img in enumerate(images):
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(chr(65 + i))
        axes[i].axis('off')
    plt.suptitle('Style Consistency Across Letters')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'style_consistency.png'), dpi=150)
    plt.show()


def mode_recovery_experiment(generator_checkpoints, device='mps'):
    """
    Analyze how mode collapse progresses and potentially recovers.

    Args:
        generator_checkpoints: List of (epoch, generator) tuples
        device: Device
    """
    from training_dynamics import analyze_mode_coverage

    mode_coverages = []
    epochs = []

    for epoch, gen in generator_checkpoints:
        gen.to(device)
        coverage, _ = analyze_mode_coverage(gen, device)
        mode_coverages.append(coverage)
        epochs.append(epoch)
        print(f"Epoch {epoch}: Mode coverage = {coverage:.2f}")

    # Plot mode coverage over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mode_coverages, marker='o', linewidth=2)
    plt.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Perfect (26/26)')
    plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='50% (13/26)')
    plt.xlabel('Epoch')
    plt.ylabel('Mode Coverage')
    plt.title('Mode Collapse Recovery Over Training')
    plt.ylim([0, 1.1])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mode_recovery.png'), dpi=150)
    plt.show()
