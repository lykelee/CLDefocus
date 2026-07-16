from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from deblurring.models.iniknet.data import TestDataset
from deblurring.models.iniknet.network import INIKNet
from deblurring.utils import pad_to_multiple, load_test_config_with_overrides


def save_output_image(tensor: torch.Tensor, path: Path) -> None:
    arr = tensor[0].cpu().clamp(0.0, 1.0).numpy().transpose(1, 2, 0)
    arr = arr[..., ::-1]  # BGR -> RGB
    Image.fromarray(np.ascontiguousarray((arr * 255).round().astype(np.uint8))).save(path)


def main(cfg):
    test_cfg = cfg["test"]
    model_cfg = cfg["model"]

    dataset = TestDataset(test_cfg["img_path"], test_cfg["gt_path"])
    dataloader = DataLoader(
        dataset,
        batch_size=test_cfg["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=1,
        pin_memory=True,
    )

    net = INIKNet(model_cfg).cuda()
    net.load_state_dict(torch.load(test_cfg["weights_path"], map_location="cuda"))
    net.eval()
    print(f"Parameters: {sum(p.nelement() for p in net.parameters()) / 1e6:.2f}M")
    print(f"Loaded weights: {test_cfg['weights_path']}")

    out_dir = Path(test_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            img = batch["img"].cuda()

            img_padded, crop = pad_to_multiple(img, 32)
            outs = net(img_padded)
            pred = crop(outs[-1])

            src_name = Path(dataset.img_names[step]).stem
            save_output_image(pred, out_dir / f"{src_name}.png")

    print(f"images -> {out_dir}")


if __name__ == "__main__":
    main(load_test_config_with_overrides("configs/arch/iniknet.yaml"))
