"""
test_cpu.py

CPU-only inference script for DnCNN, modified from the original test.py
in https://github.com/SaoYan/DnCNN-PyTorch.

Modifications from the original test.py:
  1. Removed all GPU-specific code (.cuda(), nn.DataParallel) so the script
     runs on CPU-only machines.
  2. Strip the "module." prefix from the pretrained state_dict keys, which
     was added when the model was originally saved while wrapped in
     nn.DataParallel.
  3. Removed the deprecated `Variable` wrapper (no-op since PyTorch 0.4).
  4. Added weights_only=False to torch.load to suppress the security
     warning introduced in PyTorch 2.6+.
  5. Reads images as grayscale directly with cv2.IMREAD_GRAYSCALE instead
     of reading as BGR and indexing one channel.
"""

import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from models import DnCNN
from utils import batch_PSNR

parser = argparse.ArgumentParser(description="DnCNN_Test (CPU version)")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs/DnCNN-S-25", help="Path to directory containing net.pth")
parser.add_argument("--test_data", type=str, default="Set12", help="Test set folder name under data/")
parser.add_argument("--test_noiseL", type=float, default=25, help="Noise level (sigma) used on test set")
opt = parser.parse_args()


def normalize(data):
    """Scale [0,255] uint8 to [0,1] float."""
    return data / 255.0


def load_state_dict_cpu(model, weights_path):
    """
    Load a state dict saved from a DataParallel-wrapped model into a plain
    (non-DataParallel) model on CPU. DataParallel prefixes every parameter
    name with 'module.', which we strip here.
    """
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    return model


def main():
    # 1. Build the model architecture (no DataParallel, no .cuda())
    print("Loading model ...")
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)

    # 2. Load pretrained weights, stripping the 'module.' prefix
    weights_path = os.path.join(opt.logdir, "net.pth")
    net = load_state_dict_cpu(net, weights_path)
    net.eval()  # set to inference mode (affects BatchNorm behavior)

    # 3. Gather test images
    print("Loading data ...")
    files_source = sorted(glob.glob(os.path.join("data", opt.test_data, "*.png")))
    if not files_source:
        raise RuntimeError(f"No PNG images found in data/{opt.test_data}/")
    print(f"Found {len(files_source)} test images.")

    # 4. Run inference on each image
    psnr_test = 0.0
    for f in files_source:
        # Load image as grayscale, normalize to [0,1]
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        img = normalize(np.float32(img))

        # Add batch and channel dimensions: [H, W] -> [1, 1, H, W]
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=0)
        ISource = torch.from_numpy(img)

        # Generate Gaussian noise at the specified sigma (on [0,1] scale)
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL / 255.0)
        INoisy = ISource + noise

        # Inference: predict the noise residual, subtract from noisy image
        with torch.no_grad():
            Out = torch.clamp(INoisy - net(INoisy), 0.0, 1.0)

        # PSNR between denoised output and clean reference
        psnr = batch_PSNR(Out, ISource, 1.0)
        psnr_test += psnr
        print(f"  {os.path.basename(f):<20s} PSNR = {psnr:.4f}")

    psnr_test /= len(files_source)
    print(f"\nAverage PSNR on {opt.test_data} (sigma={opt.test_noiseL}): {psnr_test:.4f}")


if __name__ == "__main__":
    main()