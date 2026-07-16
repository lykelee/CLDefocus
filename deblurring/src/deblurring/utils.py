from typing import Callable

import torch
import torch.nn.functional as F


def load_test_config_with_overrides(default_config: str) -> dict:
    """
    Parse `--config` plus optional path overrides, return the merged config.

    Overrides (`--img-path/--gt-path/--weights/--output-dir`) win over the yaml's
    `test:` block, letting a batch runner reuse one arch config across many
    (weight, dataset) combinations. `test.batch_size` defaults to 1.
    """
    import argparse
    import yaml

    p = argparse.ArgumentParser()
    p.add_argument("--config", default=default_config)
    p.add_argument("--img-path", dest="img_path")
    p.add_argument("--gt-path", dest="gt_path")
    p.add_argument("--weights")
    p.add_argument("--output-dir", dest="output_dir")
    a = p.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f) or {}
    t = cfg.setdefault("test", {})
    t.setdefault("batch_size", 1)
    if a.img_path:
        t["img_path"] = a.img_path
    if a.gt_path:
        t["gt_path"] = a.gt_path
    if a.weights:
        t["weights_path"] = a.weights
    if a.output_dir:
        t["output_dir"] = a.output_dir
    return cfg


def pad_to_multiple(
    tensor: torch.Tensor, multiple: int
) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """
    Pad (B,C,H,W) to the next multiple of `multiple` using reflect padding.

    Returns (padded_tensor, crop) where `crop` is a closure that crops any
    output tensor back to the original (H, W):

        padded, crop = pad_to_multiple(img, 16)
        out = model(padded)
        out = crop(out)
    """
    _, _, h, w = tensor.shape
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if pad_h == 0 and pad_w == 0:
        padded = tensor
    else:
        padded = F.pad(tensor, [0, pad_w, 0, pad_h], mode="reflect")

    def crop(out: torch.Tensor) -> torch.Tensor:
        return out[..., :h, :w]

    return padded, crop
