"""
Latent space analysis tools for hierarchical VAE.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_latent_hierarchy(model, data_loader, device='cuda'):
    """
    Visualize the two-level latent space structure.
    
    TODO:
    1. Encode all data to get z_high and z_low
    2. Use t-SNE to visualize z_high (colored by genre)
    3. For each z_high cluster, show z_low variations
    4. Create hierarchical visualization
    """
    model.eval()
    z_high_list = []
    z_low_list = []
    density_list = []
    genres = []

    with torch.no_grad():
        for patterns, styles, density in data_loader:
            patterns = patterns.to(device)
            mu_low, logvar_low, mu_high, logvar_high = model.encode_hierarchy(patterns)
            z_high_list.append(mu_high.cpu())
            z_low_list.append(mu_low.cpu())
            density_list.append(density.cpu())
            genres.extend(styles.numpy())

    z_high_all = torch.cat(z_high_list).numpy()
    z_low_all = torch.cat(z_low_list).numpy()
    density_all = torch.cat(density_list).numpy()
    genres = np.array(genres)

    # t-SNE on z_high
    z_high_2d = TSNE(n_components=2, perplexity=10, random_state=42).fit_transform(z_high_all)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_high_2d[:, 0], z_high_2d[:, 1], c=genres, cmap='tab10', alpha=0.7)
    plt.title("t-SNE of z_high (style level)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(*scatter.legend_elements(), title="Genre")

    # t-SNE on z_low
    z_low_2d = TSNE(n_components=2, perplexity=10, random_state=42).fit_transform(z_low_all)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z_low_2d[:, 0], z_low_2d[:, 1], c=genres, cmap='tab10', alpha=0.7)
    plt.title("t-SNE of z_low (style level)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(*scatter.legend_elements(), title="Genre")
    plt.show()

    

def interpolate_styles(model, pattern1, pattern2, n_steps=10, device='cuda'):
    """
    Interpolate between two drum patterns at both latent levels.
    
    TODO:
    1. Encode both patterns to get latents
    2. Interpolate z_high (style transition)
    3. Interpolate z_low (variation transition)
    4. Decode and visualize both paths
    5. Compare smooth vs abrupt transitions
    """
    model.eval()
    pattern1 = pattern1.to(device).unsqueeze(0)
    pattern2 = pattern2.to(device).unsqueeze(0)

    with torch.no_grad():
        # Encode both patterns
        mu_low1, _, mu_high1, _ = model.encode_hierarchy(pattern1)
        mu_low2, _, mu_high2, _ = model.encode_hierarchy(pattern2)

        # Interpolation
        styles = []
        variations = []

        for alpha in np.linspace(0, 1, n_steps):
            z_high = (1 - alpha) * mu_high1 + alpha * mu_high2
            z_low = mu_low1 
            recon_logits = model.decode_hierarchy(z_high, z_low)
            recon1 = torch.sigmoid(recon_logits)
            styles.append(recon1.cpu().squeeze(0).numpy())

        # Visualization: binary heatmaps
        fig, axs = plt.subplots(1, n_steps, figsize=(n_steps * 1.5, 2))
        for i in range(n_steps):
            axs[i].imshow(styles[i].T, aspect='auto', cmap='Greys', origin='lower')
            axs[i].axis('off')
        plt.suptitle("Style Interpolation")
        plt.show()

        for alpha in np.linspace(0, 1, n_steps):
            z_high = mu_high1
            z_low = (1 - alpha) * mu_low1 + alpha * mu_low2
            recon_logits = model.decode_hierarchy(z_high, z_low)
            recon2 = torch.sigmoid(recon_logits)
            variations.append(recon2.cpu().squeeze(0).numpy())

        # Visualization: binary heatmaps
        fig, axs = plt.subplots(1, n_steps, figsize=(n_steps * 1.5, 2))
        for i in range(n_steps):
            axs[i].imshow(variations[i].T, aspect='auto', cmap='Greys', origin='lower')
            axs[i].axis('off')
        plt.suptitle("Variation Interpolation")
        plt.show()

    return styles[-1], variations[-1]

def measure_disentanglement(model, data_loader, device='cuda'):
    """
    Measure how well the hierarchy disentangles style from variation.
    
    TODO:
    1. Group patterns by genre
    2. Compute z_high variance within vs across genres
    3. Compute z_low variance for same genre
    4. Return disentanglement metrics
    """
    model.eval()
    genre_to_z_high = {0:[], 1:[], 2:[], 3:[], 4:[]}
    genre_to_z_low = {0:[], 1:[], 2:[], 3:[], 4:[]}

    with torch.no_grad():
        for patterns, genres, _ in data_loader:
            patterns = patterns.to(device)
            mu_low, _, mu_high, _ = model.encode_hierarchy(patterns)

            for i in range(len(genres)):
                genre = genres[i].item()
                genre_to_z_high[genre].append(mu_high[i].cpu().numpy())
                genre_to_z_low[genre].append(mu_low[i].cpu().numpy())

    all_genres = list(genre_to_z_high.keys())

    # Compute intra-genre variance
    intra_var_high = []
    intra_var_low = []

    for genre in all_genres:
        z_high = np.stack(genre_to_z_high[genre])
        z_low = np.stack(genre_to_z_low[genre])
        intra_var_high.append(np.mean(np.var(z_high, axis=0)))
        intra_var_low.append(np.mean(np.var(z_low, axis=0)))

    # Compute inter-genre variance (z_high only)
    genre_means = np.stack([np.mean(genre_to_z_high[genre], axis=0) for genre in all_genres])
    inter_var_high = np.mean(np.var(genre_means, axis=0))

    return {'intra_var_high': np.mean(intra_var_high),
            'intra_var_low': np.mean(intra_var_low),
            'inter_var_high': inter_var_high,
            'disentanglement_score': inter_var_high / (np.mean(intra_var_high) + 1e-8)
            }

def controllable_generation(model, data_loader, genre_labels, device='mps'):
    """
    Test controllable generation using the hierarchy.
    """
    model.eval()
    genre_embeddings = {}
    patterns_by_genre = []

    # Step 0: collect z_high vectors per genre
    genre_to_z_high = {genre: [] for genre in genre_labels}

    with torch.no_grad():
        for patterns, genres, _ in data_loader:
            patterns = patterns.to(device)
            mu_low, _, mu_high, _ = model.encode_hierarchy(patterns)
            for i, genre_idx in enumerate(genres):
                genre_name = genre_labels[genre_idx]  # convert tensor index to string
                genre_to_z_high[genre_name].append(mu_high[i].cpu())


    # Step 1: average z_high for each genre
    for genre in genre_labels:
        z_high_vectors = genre_to_z_high[genre]
        z_high_vectors = torch.stack(z_high_vectors)
        genre_embeddings[genre] = z_high_vectors.mean(dim=0)

    # Step 2: generate with fixed z_high, varying z_low
    for genre, z_high in genre_embeddings.items():
        z_high = z_high.unsqueeze(0).repeat(10, 1).to(device)
        z_low = torch.randn(10, model.z_low_dim).to(device)
        recon_logits = model.decode_hierarchy(z_high, z_low, temperature=0.7)
        patterns_by_genre.append((genre, torch.sigmoid(recon_logits).detach().cpu().numpy()))


    # Step 3: visualize results
    for genre, patterns in patterns_by_genre:
        fig, axs = plt.subplots(1, 10, figsize=(15, 2))
        for i in range(10):
            axs[i].imshow(patterns[i].T, aspect='auto', cmap='Greys', origin='lower')
            axs[i].axis('off')
        plt.suptitle(f"Generated Patterns for Genre: {genre}")
        plt.show()
