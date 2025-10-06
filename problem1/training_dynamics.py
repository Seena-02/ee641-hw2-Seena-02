"""
GAN training implementation with mode collapse analysis.
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

from provided.metrics import _simple_letter_classifier
from provided.visualize import plot_alphabet_grid, plot_training_history

def train_gan(generator, discriminator, data_loader, num_epochs=100, device='mps'):
    """
    Standard GAN training implementation.
    
    Uses vanilla GAN objective which typically exhibits mode collapse.
    
    Args:
        generator: Generator network
        discriminator: Discriminator network
        data_loader: DataLoader for training data
        num_epochs: Number of training epochs
        device: Device for computation
        
    Returns:
        dict: Training history and metrics
    """
    # Initialize optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Training history
    history = defaultdict(list)
    
    for epoch in range(num_epochs + 1):
        for batch_idx, (real_images, labels) in enumerate(data_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Labels for loss computation
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # ========== Train Discriminator ==========
            # TODO: Implement discriminator training step
            # 1. Zero gradients
            # 2. Forward pass on real images
            # 3. Compute real loss
            # 4. Generate fake images from random z
            # 5. Forward pass on fake images (detached)
            # 6. Compute fake loss
            # 7. Backward and optimize

            d_optimizer.zero_grad()

            if discriminator.conditional:
                labels = labels.to(device).float()
                out_real = discriminator(real_images, labels)
            else:
                out_real = discriminator(real_images)
            
            d_loss_real = criterion(out_real, real_labels)

            z = torch.randn(real_images.size(0), generator.z_dim, device=device)

            if generator.conditional:
                labels = labels.to(device)
                labels_one_hot = F.one_hot(labels, num_classes=26).float().to(device)
                fake_images = generator(z, labels_one_hot)  # <-- pass separately
            else:
                fake_images = generator(z)

            if discriminator.conditional:
                out_fake = discriminator(fake_images.detach(), labels)
            else:
                out_fake = discriminator(fake_images.detach())

            d_loss_fake = criterion(out_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # ========== Train Generator ==========
            # TODO: Implement generator training step
            # 1. Zero gradients
            # 2. Generate fake images
            # 3. Forward pass through discriminator
            # 4. Compute adversarial loss
            # 5. Backward and optimize

            g_optimizer.zero_grad();

            # Fake image here
            z = torch.randn(batch_size, generator.z_dim, 1, 1, device=device)
            z = z.view(z.size(0), -1)  # reshape to (batch_size, z_dim)
            if generator.conditional:
                labels_one_hot = F.one_hot(labels.to(device), num_classes=26).float().to(device)
                fake_images = generator(z, labels_one_hot)
            else:
                fake_images = generator(z)

            outputs = discriminator(fake_images)

            g_loss = criterion(outputs, real_labels)

            g_loss.backward()

            g_optimizer.step()


            # Log metrics
            if batch_idx % 10 == 0:
                history['d_loss'].append(d_loss.item())
                history['g_loss'].append(g_loss.item())
                history['epoch'].append(epoch + batch_idx/len(data_loader))
        
        # Analyze mode collapse every 10 epochs
        if epoch % 10 == 0:
            print("here", epoch)
            mode_coverage = analyze_mode_coverage(generator, device)[0]
            history['mode_coverage'].append(mode_coverage)
            print(f"Epoch {epoch}: Mode coverage = {mode_coverage:.2f}")
            # Save checkpoint at same interval
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(generator.state_dict(), f"checkpoints/generator_epoch_{epoch}.pth")
    
    return history

def analyze_mode_coverage(generator, device, n_samples=1000):
    """
    Measure mode coverage by counting unique letters in generated samples.
    
    Args:
        generator: Trained generator network
        device: Device for computation
        n_samples: Number of samples to generate
        
    Returns:
        float: Coverage score (unique letters / 26)
    """
    # TODO: Generate n_samples images
    # Use provided letter classifier to identify generated letters
    # Count unique letters produced
    # Return coverage score (0 to 1)

    z = torch.randn(n_samples, generator.z_dim, device=device)
    if generator.conditional:
        labels = torch.randint(0, 26, (n_samples), device = device)
        fake_images = generator(z, labels)
    else:
        fake_images = generator(z)

    with torch.no_grad():
        preds = []
        for img in fake_images:
            pred = _simple_letter_classifier(img)
            preds.append(pred)
        preds = torch.tensor(preds, device=device)

    unique_letters = preds.unique().numel()

    convergence = unique_letters / 26

    return convergence, preds.unique().tolist()


def visualize_mode_collapse(generator, history, save_path):
    """
    Visualize mode collapse progression over training.
    
    Args:
        history: Training metrics dictionary
        save_path: Output path for visualization
    """
    # TODO: Plot mode coverage over time
    # Show which letters survive and which disappear
    
    # 1. Plot mode coverage over time (training metric)
    plot_training_history(history, save_path=save_path.replace('.png', '_training.png'))
    
    # 2. Show which letters survive / disappear (visual inspection)
    #    Use the provided alphabet grid to generate all letters from A–Z.
    fig = plot_alphabet_grid(generator)
    
    plt.suptitle("Generated Alphabet After Training", fontsize=16)
    fig.savefig(save_path.replace('.png', '_alphabet.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)