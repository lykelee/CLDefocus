"""
Usage:
    python scripts/train_iniknet.py --config configs/train/iniknet/ours.yaml
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from deblurring.models.iniknet.data import Dataset
from deblurring.models.iniknet.network import INIKNet
from deblurring.models.iniknet.util import (
    multi_scale_loss,
    toBlue,
    toGreen,
    toRed,
    toYellow,
)


def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(cfg):
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    loss_cfg = cfg.get("loss", {})

    seed_everything(777)

    loss_opt = {
        "freq": loss_cfg.get("freq_weight", 0.1),
        "lpips": loss_cfg.get("lpips_weight", 0.1),
        "vgg": loss_cfg.get("vgg_weight", 0.0),  # paper has no VGG
    }
    print(f"loss: mse 1.0 | {loss_opt}")

    train_data = Dataset(
        img_path=train_cfg["img_path"],
        gt_path=train_cfg["gt_path"],
        crop_size=(train_cfg["crop_size"], train_cfg["crop_size"]),
    )
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=train_cfg["batch_size"] // train_cfg["acc_step"],
        num_workers=train_cfg["num_workers"],
    )
    print(f"train: {train_cfg['img_path']} | {train_cfg['gt_path']} ({len(train_data)} pairs)")

    net = INIKNet(model_cfg).cuda()
    net.train()

    model_name = model_cfg["model_name"]
    ckpt_dir = Path(train_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(ckpt_dir / f"{model_name}.txt", "a")
    log_file.write("\n\n")

    total_iters = train_cfg["total_iters"]
    save_period = train_cfg["save_period"]
    log_period = train_cfg["log_period"]
    acc_step = train_cfg["acc_step"]

    optimizer = torch.optim.Adam(net.parameters(), lr=train_cfg["lr"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, train_cfg["mile_stone"], train_cfg["rate"]
    )

    if train_cfg["start_from_trained"]:
        net.load_state_dict(torch.load(train_cfg["weight_path"]))
        optimizer.load_state_dict(torch.load(train_cfg["optimizer_path"]))
        scheduler.load_state_dict(torch.load(train_cfg["scheduler_path"]))
        print(f"resume: True | from {train_cfg['weight_path']}")
    else:
        print("resume: False | training from scratch")

    # MultiStepLR.last_epoch counts scheduler.step() calls == optimizer updates == iters
    global_iter = scheduler.last_epoch
    sum_los, sum_psnr, sum_freq = [], [], []
    micro_step = 0

    pbar = tqdm(total=total_iters, initial=global_iter, unit="it", ncols=150, desc="train")
    optimizer.zero_grad()
    done = global_iter >= total_iters
    while not done:
        for batch in train_loader:
            gt = batch["gt"].cuda()
            img = batch["img"].cuda()

            out = net(img)
            los = multi_scale_loss(
                out,
                gt,
                mse_lambda=1.0,
                l1_lambda=0.0,
                char_lambda=0.0,
                freq_lambda=loss_opt["freq"],
                lpips_lambda=loss_opt["lpips"],
                vgg_lambda=loss_opt["vgg"],
            )
            los["loss"].backward()

            sum_los.append(los["loss"].item())
            sum_psnr.append(los["psnr"].item())
            sum_freq.append(los["freq loss"].item())

            micro_step += 1
            if micro_step % acc_step != 0:
                continue

            # one optimizer update == one iter
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_iter += 1

            pbar.set_postfix(
                {
                    "loss": toBlue(f"{los['loss'].item():.5f}"),
                    "lr": toYellow(f"{optimizer.param_groups[0]['lr']:.2e}"),
                    "psnr": f"{los['psnr'].item():.5f}",
                }
            )
            pbar.update(1)

            if global_iter % log_period == 0:
                avg_loss = sum(sum_los) / len(sum_los)
                avg_psnr = sum(sum_psnr) / len(sum_psnr)
                avg_freq = sum(sum_freq) / len(sum_freq)
                msg = (
                    f"iter {global_iter}/{total_iters} "
                    f"loss:{avg_loss:.5f} psnr:{avg_psnr:.5f} freq:{avg_freq:.5f} "
                    f"lr:{optimizer.param_groups[0]['lr']:.2e}"
                )
                tqdm.write(msg)
                log_file.write(msg + "\n")
                log_file.flush()
                sum_los, sum_psnr, sum_freq = [], [], []

            if global_iter % save_period == 0:
                torch.save(net.state_dict(), ckpt_dir / f"{model_name}-iter{global_iter:08d}.pth")
                torch.save(net.state_dict(), ckpt_dir / f"{model_name}.pth")
                torch.save(optimizer.state_dict(), ckpt_dir / f"Adam-{model_name}.pth")
                torch.save(scheduler.state_dict(), ckpt_dir / f"scheduler-{model_name}.pth")

            if global_iter >= total_iters:
                done = True
                break

    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/iniknet/dpdd.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
