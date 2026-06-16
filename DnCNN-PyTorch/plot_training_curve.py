"""
plot_training_curve.py

Reads TensorBoard event files written during training and plots the
validation PSNR per epoch.
"""

import argparse
import glob
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="tb_logs",
                        help="Folder containing events.out.tfevents.* files")
    parser.add_argument("--output", type=str, default="training_curve.png")
    parser.add_argument("--scalar", type=str, default="PSNR on validation data",
                        help="Which logged scalar to plot")
    parser.add_argument("--reference", type=float, default=30.404,
                        help="Reference horizontal line (SaoYan pretrained PSNR)")
    opt = parser.parse_args()

    # Find all event files (one per Colab session)
    event_files = sorted(glob.glob(os.path.join(opt.log_dir, "events.out.tfevents.*")))
    if not event_files:
        raise RuntimeError(f"No event files found in {opt.log_dir}")
    print(f"Found {len(event_files)} event file(s):")
    for ef in event_files:
        print(f"  {ef}")

    # Read each file and collect (step, value) pairs
    all_points = []
    for ef in event_files:
        ea = EventAccumulator(ef)
        ea.Reload()
        available = ea.Tags()["scalars"]
        if opt.scalar not in available:
            print(f"  '{opt.scalar}' not found in {ef}.")
            print(f"  Available scalars: {available}")
            continue
        for event in ea.Scalars(opt.scalar):
            all_points.append((event.step, event.value))

    if not all_points:
        raise RuntimeError(f"No data for scalar '{opt.scalar}'")

    # Deduplicate by step (later sessions may re-log the same epoch)
    all_points.sort(key=lambda p: p[0])
    seen = {}
    for step, value in all_points:
        seen[step] = value
    steps = sorted(seen.keys())
    values = [seen[s] for s in steps]
    epochs = [s + 1 for s in steps]  # 1-indexed for display

    print(f"\n{len(epochs)} unique data points, final value: {values[-1]:.4f} dB")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, values, marker='o', markersize=4, linewidth=1.5, color='#1f77b4')
    plt.axhline(y=opt.reference, linestyle='--', color='gray', alpha=0.7,
                label=f'SaoYan pretrained ({opt.reference:.3f} dB)')
    # Highlight the LR drop at epoch 30 if the data covers it
    if max(epochs) >= 30:
        plt.axvline(x=30, linestyle=':', color='orange', alpha=0.6,
                    label='LR drop ($10^{-3} \\rightarrow 10^{-4}$)')
    plt.xlabel("Epoch")
    plt.ylabel("Set12 validation PSNR (dB)")
    plt.title("DnCNN-S training: validation PSNR over epochs")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(opt.output, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {opt.output}")


if __name__ == "__main__":
    main()