"""
Train NAFNet via its BasicSR training loop.

Usage (single GPU):
    python scripts/train_nafnet.py \
        -opt configs/train/nafnet/ours.yml

Usage (multi-GPU, torchrun):
    torchrun --nproc_per_node=8 scripts/train_nafnet.py \
        --launcher pytorch \
        -opt configs/train/nafnet/ours.yml
"""

from deblurring.models.nafnet.basicsr.train import main

if __name__ == "__main__":
    main()
