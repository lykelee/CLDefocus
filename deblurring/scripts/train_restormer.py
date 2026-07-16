"""
Train Restormer via BasicSR training loop.

Usage (single GPU):
    python scripts/train_restormer.py \
        -opt configs/train/restormer/ours.yml

Usage (multi-GPU, torchrun):
    torchrun --nproc_per_node=8 scripts/train_restormer.py \
        --launcher pytorch \
        -opt configs/train/restormer/ours.yml
"""

from deblurring.models.restormer.basicsr.train import main

if __name__ == "__main__":
    main()
