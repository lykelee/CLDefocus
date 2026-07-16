from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from deblurring.models.nrknet.data import TestDataset
from deblurring.models.nrknet.network import NRKNet
from deblurring.models.nrknet.utils import set_requires_grad
from deblurring.utils import pad_to_multiple, load_test_config_with_overrides


def save_output_image(tensor: torch.Tensor, path: Path) -> None:
    arr = tensor[0].cpu().clamp(0.0, 1.0).numpy().transpose(1, 2, 0)
    arr = arr[..., ::-1]  # BGR -> RGB
    Image.fromarray(np.ascontiguousarray((arr * 255).round().astype(np.uint8))).save(path)


def main(cfg):
    net_cfg = cfg["net"]
    test_cfg = cfg["test"]

    dataset = TestDataset(test_cfg["img_path"], test_cfg["gt_path"])
    dataloader = DataLoader(
        dataset,
        batch_size=test_cfg["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=1,
        pin_memory=True,
    )

    nrknet = NRKNet(
        num_res=net_cfg["num_res"],
        num_kernels=net_cfg["num_kernels"],
        in_ch=net_cfg["in_ch"],
    ).cuda()
    print("Parameters: %.2fM" % (sum(p.nelement() for p in nrknet.parameters()) / 1e6))

    set_requires_grad(nrknet, False)
    weights_path = test_cfg["weights_path"]
    nrknet.load_state_dict(torch.load(weights_path, map_location="cuda"))
    print(f"Loaded weights: {weights_path}")

    out_dir = Path(test_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            img = batch["img256"].cuda()

            img_padded, crop = pad_to_multiple(img, 16)
            dbs, _ = nrknet(img_padded, phase="test")
            pred = crop(dbs[-1])

            src_name = dataset.img_list[step].stem
            save_output_image(pred, out_dir / f"{src_name}.png")

    print(f"images -> {out_dir}")


if __name__ == "__main__":
    main(load_test_config_with_overrides("configs/arch/nrknet.yaml"))
