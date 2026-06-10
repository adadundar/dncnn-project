"""
train_modern.py

Modernized training script for DnCNN, adapted from the original train.py
in the SaoYan/DnCNN-PyTorch repository.

Changes from the original train.py:
  1. tensorboardX -> torch.utils.tensorboard (TensorBoard is now built into
     PyTorch; the standalone tensorboardX package is no longer needed).
  2. nn.MSELoss(size_average=False) -> nn.MSELoss(reduction='sum')
     (size_average was removed in PyTorch >=1.0).
  3. Removed Variable() wrapping (Variable has been merged into Tensor
     since PyTorch 0.4; the wrapping is a no-op).
  4. Removed volatile=True from validation (it was removed in PyTorch 0.4
     in favor of the torch.no_grad() context manager).
  5. argparse --preprocess: switched from type=bool (broken: any non-empty
     string is truthy) to action='store_true' (the standard idiom).
  6. Hyperparameters kept identical to the original SaoYan recipe: Adam
     optimizer, 50 epochs, stepwise LR (1e-3 -> 1e-4 at epoch 30),
     batch size 128. This deviates from the paper's stated recipe (SGD
     with momentum, exponential LR decay, weight decay 1e-4); we keep
     SaoYan's choice because their pretrained weights demonstrably
     reproduce the paper's PSNR results.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import DnCNN
from dataset import prepare_data, Dataset
from utils import weights_init_kaiming, batch_PSNR

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN (modernized)")
parser.add_argument("--preprocess", action='store_true', help='run prepare_data before training')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="Epoch at which to drop the learning rate by 10x")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='Output directory for logs and checkpoints')
parser.add_argument("--mode", type=str, default="S", help='Known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='Noise level for mode S; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='Noise level for validation')
parser.add_argument("--start_epoch", type=int, default=0, help="Epoch to start training from (for resuming)") 
# added resume feature (e.g. start after epoch 29 for runtime crash in colab)
opt = parser.parse_args()


def main():
    # Reproducibility
    print('Loading dataset ...')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4,
                              batch_size=opt.batchSize, shuffle=True)
    print(f"# of training samples: {len(dataset_train)}")

    # Build model and initialize weights
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)

    # Loss: sum-reduction MSE, divided by 2N per batch (the paper's convention)
    criterion = nn.MSELoss(reduction='sum')

    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = nn.DataParallel(net).to(device)
    criterion = criterion.to(device)

    """
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # Ensure output directory exists
    os.makedirs(opt.outf, exist_ok=True)
    """

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # Ensure output directory exists
    os.makedirs(opt.outf, exist_ok=True)

    # Resume support: prefer full checkpoint, fall back to weights-only
    start_epoch = opt.start_epoch
    checkpoint_path = os.path.join(opt.outf, 'checkpoint.pth')
    weights_path = os.path.join(opt.outf, 'net.pth')
    if os.path.exists(checkpoint_path):
        print(f"Found full checkpoint at {checkpoint_path}, resuming...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch + 1}")
    elif os.path.exists(weights_path) and opt.start_epoch > 0:
        print(f"Found weights-only file at {weights_path}, resuming from --start_epoch {opt.start_epoch}")
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        # Optimizer state starts fresh (will rebuild momentum in a few iterations)
    else:
        print("No checkpoint found, starting fresh training.")

    # TensorBoard writer
    writer = SummaryWriter(opt.outf)

    step = 0
    noiseL_B = [0, 55]  # noise range for blind training; ignored when mode='S'

    for epoch in range(start_epoch, opt.epochs):
        # Stepwise learning rate schedule
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.0
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print(f"learning rate {current_lr}")

        # --- Training loop ---
        for i, data in enumerate(loader_train, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            img_train = data
            if opt.mode == 'S':
                noise = torch.FloatTensor(img_train.size()).normal_(
                    mean=0, std=opt.noiseL / 255.0)
            elif opt.mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0, :, :, :].size()
                    noise[n, :, :, :] = torch.FloatTensor(sizeN).normal_(
                        mean=0, std=stdN[n] / 255.0)
            else:
                raise ValueError(f"Unknown mode '{opt.mode}', must be 'S' or 'B'")

            imgn_train = img_train + noise

            # Move tensors to GPU (no Variable wrapping needed)
            img_train = img_train.to(device)
            imgn_train = imgn_train.to(device)
            noise = noise.to(device)

            # Forward: model predicts the noise residual
            out_train = model(imgn_train)
            loss = criterion(out_train, noise) / (imgn_train.size()[0] * 2)
            loss.backward()
            optimizer.step()

            # Compute PSNR on the training batch (for monitoring)
            model.eval()
            with torch.no_grad():
                out_train_clamped = torch.clamp(imgn_train - model(imgn_train), 0., 1.)
            psnr_train = batch_PSNR(out_train_clamped, img_train, 1.)

            if i % 50 == 0:
                print(f"[epoch {epoch+1}][{i+1}/{len(loader_train)}] "
                      f"loss: {loss.item():.4f} PSNR_train: {psnr_train:.4f}")

            if step % 10 == 0:
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1

        # --- End-of-epoch validation ---
        model.eval()
        psnr_val = 0.0
        with torch.no_grad():
            for k in range(len(dataset_val)):
                img_val = torch.unsqueeze(dataset_val[k], 0).to(device)
                noise = torch.FloatTensor(img_val.size()).normal_(
                    mean=0, std=opt.val_noiseL / 255.0).to(device)
                imgn_val = img_val + noise
                out_val = torch.clamp(imgn_val - model(imgn_val), 0., 1.)
                psnr_val += batch_PSNR(out_val, img_val, 1.)
        psnr_val /= len(dataset_val)
        print(f"\n[epoch {epoch+1}] PSNR_val: {psnr_val:.4f}")
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)

        # Log example images
        with torch.no_grad():
            out_train_clamped = torch.clamp(imgn_train - model(imgn_train), 0., 1.)
        Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train_clamped.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)

        # Save checkpoint after every epoch (overwrites previous)
        # commented line torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))

        # save full checkpoint (model + optimizer + epoch) for proper resume
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(opt.outf, 'checkpoint.pth'))
        # Also save weights-only for inference compatibility with test_cpu.py
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
        elif opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)
        else:
            raise ValueError(f"Unknown mode '{opt.mode}', must be 'S' or 'B'")
    main()