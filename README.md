# EE 641 - HW2

# Generative Modeling Assignment

This repository includes two advanced deep generative modeling projects:

- **Problem 1**: Font Generation using Generative Adversarial Networks (GANs)
- **Problem 2**: Hierarchical Variational Autoencoder (VAE) for Music Generation

Each project is self-contained in its own folder. Training and evaluation scripts, model definitions, and analysis tools are provided.

---

## Problem 1: Font Generation GAN

**Description:**  
Implements and analyzes GAN models to synthesize 28x28 grayscale images of font letters from a synthetic dataset. Includes both standard vanilla GANs and stabilization techniques such as Feature Matching.

**Key Files:**

- `train.py`: Main training script.
- `models.py`: Contains generator and discriminator architectures.
- `dataset.py`: Loads synthetic font data from `data/fonts/`.
- `training_dynamics.py`, `fixes.py`: Training utilities and stabilization implementations.
- `evaluate.py`: Evaluation metrics and experiments.
- Figures and results are saved in `results/` and `checkpoints/`.

**Usage:**

```bash
cd problem1
python3 train.py
```

- Training/experiment configurations can be edited directly in `train.py` or as indicated in comments.
- Outputs: Model checkpoints, figures illustrating alphabet coverage, training dynamics, style transfer, and more.
- Supports CPU, MPS, or CUDA devices.

**Dependencies:**

- Python 3.x
- PyTorch
- numpy
- matplotlib
- pillow (PIL)

---

## Problem 2: Hierarchical VAE for Music Generation

**Description:**  
Implements a hierarchical VAE for generating and analyzing drum patterns (binary piano rolls). The model separates style/genre and pattern variation across two levels of latent variables and uses advanced training tricks to avoid posterior collapse.

**Key Files:**

- `train.py`: Main training script for hierarchical VAE.
- `hierarchical_vae.py`: VAE model definition (encoders, decoders, latent design).
- `dataset.py`: Loads drum pattern data as piano rolls.
- `training_utils.py`: KL annealing, free bits, and temperature schedules.
- `analyze_latent.py`: Latent space visualization and clustering analysis.
- Results: Latent interpolations, style transfer experiments, and visualization figures saved in output folders.

**Usage:**

```bash
cd problem2
python3 train.py
```

- Edit configuration parameters in `train.py` for experiment setup.
- Checks for data availability and generates visualizations of patterns and latent structure.
- Supports both GPU (CUDA/MPS) and CPU.

**Dependencies:**

- Python 3.x
- PyTorch
- numpy
- matplotlib

---

## Notes

- Each project folder (`problem1/`, `problem2/`) is fully self-contained; make sure data directories (`data/fonts/` for problem 1, drum sequence data for problem 2) are correctly set up.
- Results, visualizations, and checkpoints are automatically saved to their respective output folders during training.
- For detailed experiment analysis and theoretical background, see the provided report/LaTeX documentation.
