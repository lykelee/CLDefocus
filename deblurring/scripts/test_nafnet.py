from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from deblurring.models.nafnet.basicsr.models.archs.NAFNet_arch import NAFNetLocal
from deblurring.utils import load_test_config_with_overrides

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class FolderDataset(torch.utils.data.Dataset):
    """Full-size RGB folder pairs, sorted, paired by order."""

    def __init__(self, img_path: str, gt_path: str):
        self.img_list = sorted(
            p for p in Path(img_path).iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        self.gt_list = sorted(p for p in Path(gt_path).iterdir() if p.suffix.lower() in IMAGE_EXTS)
        assert len(self.img_list) == len(self.gt_list), (
            f"img/gt count mismatch: {len(self.img_list)} vs {len(self.gt_list)}"
        )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        def load(path):
            with Image.open(path) as im:
                arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
            return torch.from_numpy(arr.transpose(2, 0, 1))

        return {"img": load(self.img_list[idx]), "gt": load(self.gt_list[idx])}


def save_image(tensor: torch.Tensor, path: Path) -> None:
    arr = tensor[0].cpu().clamp(0.0, 1.0).numpy().transpose(1, 2, 0)
    Image.fromarray((arr * 255).round().astype(np.uint8)).save(path)


def main(cfg):
    test_cfg = cfg["test"]
    model_cfg = cfg["model"]

    dataset = FolderDataset(test_cfg["img_path"], test_cfg["gt_path"])
    dataloader = DataLoader(
        dataset,
        batch_size=test_cfg["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=1,
        pin_memory=True,
    )

    net = NAFNetLocal(**model_cfg).cuda()
    ckpt = torch.load(test_cfg["weights_path"], map_location="cuda")
    state = ckpt.get("params", ckpt) if isinstance(ckpt, dict) else ckpt
    net.load_state_dict(state)
    net.eval()
    print(f"Parameters: {sum(p.nelement() for p in net.parameters()) / 1e6:.2f}M")
    print(f"Loaded: {test_cfg['weights_path']}")

    out_dir = Path(test_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            img = batch["img"].cuda()
            pred = net(img)  # arch self-pads to /16 and crops back
            src_name = dataset.img_list[step].stem
            save_image(pred, out_dir / f"{src_name}.png")

    print(f"images -> {out_dir}")


if __name__ == "__main__":
    main(load_test_config_with_overrides("configs/arch/nafnet.yaml"))
