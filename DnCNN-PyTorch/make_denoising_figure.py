"""
make_denoising_figure.py

Produces a side-by-side denoising visualization using a trained DnCNN model.
Output: a single PNG with clean / noisy / denoised / removed-noise / true-noise.
"""

import argparse
import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from models import DnCNN


def load_model(weights_path, num_layers=17):
    """Load a DnCNN model with the given weights (CPU, strips module. prefix)."""
    net = DnCNN(channels=1, num_of_layers=num_layers)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[len("module."):] if key.startswith("module.") else key
        new_state_dict[new_key] = value
    net.load_state_dict(new_state_dict)
    net.eval()
    return net


def main():
    parser = argparse.ArgumentParser(description="Make a DnCNN denoising visualization")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to net.pth")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to a clean PNG image (grayscale)")
    parser.add_argument("--num_of_layers", type=int, default=17,
                        help="17 for DnCNN-S, 20 for DnCNN-B")
    parser.add_argument("--sigma", type=float, default=25,
                        help="Gaussian noise level (sigma)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for noise (for reproducibility)")
    parser.add_argument("--output", type=str, default="denoising_example.png",
                        help="Output PNG path")
    opt = parser.parse_args()

    # Load model
    print(f"Loading model from {opt.weights}...")
    net = load_model(opt.weights, num_layers=opt.num_of_layers)

    # Load and normalize image
    print(f"Loading image from {opt.image}...")
    clean = cv2.imread(opt.image, cv2.IMREAD_GRAYSCALE)
    if clean is None:
        raise RuntimeError(f"Could not read image: {opt.image}")
    clean = clean.astype(np.float32) / 255.0

    # Add Gaussian noise with a fixed seed (reproducible figure)
    torch.manual_seed(opt.seed)
    clean_tensor = torch.from_numpy(clean).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    noise = torch.FloatTensor(clean_tensor.size()).normal_(mean=0, std=opt.sigma / 255.0)
    noisy_tensor = clean_tensor + noise

    # Inference
    print("Running inference...")
    with torch.no_grad():
        predicted_noise = net(noisy_tensor)
        denoised_tensor = torch.clamp(noisy_tensor - predicted_noise, 0.0, 1.0)

    # Convert back to numpy [0, 1] arrays
    noisy = noisy_tensor.squeeze().numpy()
    denoised = denoised_tensor.squeeze().numpy()
    pred_noise = predicted_noise.squeeze().numpy()
    true_noise = noise.squeeze().numpy()

    # Compute PSNR for the noisy and denoised versions (for the figure caption)
    def psnr(x, y):
        mse = np.mean((x - y) ** 2)
        return 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')

    psnr_noisy = psnr(clean, np.clip(noisy, 0, 1))
    psnr_denoised = psnr(clean, denoised)
    print(f"PSNR noisy: {psnr_noisy:.2f} dB, PSNR denoised: {psnr_denoised:.2f} dB")

    # Plot side-by-side
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

    panels = [
        (clean, "Clean", None),
        (np.clip(noisy, 0, 1), f"Noisy ($\\sigma={int(opt.sigma)}$)\nPSNR {psnr_noisy:.2f} dB", None),
        (denoised, f"Denoised\nPSNR {psnr_denoised:.2f} dB", None),
        (pred_noise, "Removed by model", "noise"),
        (true_noise, "True added noise", "noise"),
    ]
    for ax, (img, title, kind) in zip(axes, panels):
        if kind == "noise":
            # Show noise centered around 0, use a diverging colormap
            vmax = max(abs(img.min()), abs(img.max()))
            ax.imshow(img, cmap="bwr", vmin=-vmax, vmax=vmax)
        else:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(opt.output, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {opt.output}")


if __name__ == "__main__":
    main()