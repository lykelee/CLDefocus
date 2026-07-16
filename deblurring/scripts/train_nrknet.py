"""
Usage:
    python scripts/train_nrknet.py --config configs/train/nrknet/ours.yaml
"""

import sys
import argparse

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from tqdm.auto import tqdm

from deblurring.models.nrknet.data import Dataset, TestDataset
from deblurring.models.nrknet.log import TensorBoardX
from deblurring.utils import pad_to_multiple
from deblurring.models.nrknet.network import NRKNet
from deblurring.models.nrknet.utils import load_model_iter, save_model_iter, save_optimizer_iter

log10 = np.log(10)
MAX_DIFF = 1
mse = None
mae = None


def setup_seed(seed):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def SSIM_loss(output, target):
    return 1 - torch.mean(ssim(output, target, data_range=1, size_average=False))


def FFT_loss(output, target):
    target_fft = torch.fft.fft2(target)
    output_fft = torch.fft.fft2(output)
    return mae(target_fft.real, output_fft.real) + mae(target_fft.imag, output_fft.imag)


def compute_loss(dbs, bs, clear256, blurry256, it, phase, lr_schedule, loss_opt):
    """Loss config (loss_opt):
        fft_weight    : β for FDR/FFT loss (paper β=0.1)
        reblur_weight : α for reblurring loss (paper α=0.1; 0 = original code, dropped)
        use_ssim      : enable GKMNet-style SSIM "blinking" phase (NOT in NRKNet paper)

    Paper loss (eq.12-14): MSE + β·FFT + α·reblur, all scales, always.
    Reproduced by use_ssim=False, fft_weight=0.1, reblur_weight=0.1.
    """
    scales = len(dbs)

    if phase == "train":
        blurrys = [
            F.interpolate(blurry256, scale_factor=(0.5) ** i, mode="bilinear")
            for i in range(scales - 1)
        ]
        clears = [clear256] + [
            F.interpolate(clear256, scale_factor=(0.5) ** i, mode="bilinear")
            for i in range(scales - 1)
        ]
        weights = [1 - 0.25 * (scales - i - 1) for i in range(scales)]
        blurrys.reverse()
        clears.reverse()

        ssim_loss, mse_loss, fft_loss, reblur_loss = 0, 0, 0, 0

        # SSIM "blinking" phase: alternates on a cycle derived from the first
        # LR-decay iter. Off by default (paper has no SSIM term).
        ssim_phase = False
        if loss_opt["use_ssim"]:
            cycle = lr_schedule[0][0] // 10 if lr_schedule else 20000
            temp = (it % (cycle * 2)) % (cycle // 5)
            ssim_phase = temp > cycle // 10 and it >= cycle

        if ssim_phase:
            loss = mse(dbs[-1], clear256)
            psnr = 10 * torch.log(MAX_DIFF**2 / loss) / log10
            for i in range(len(clears)):
                ssim_loss += SSIM_loss(dbs[i], clears[i]) * weights[i]
        else:
            mse_loss = mse(dbs[-1], clear256)
            psnr = 10 * torch.log(MAX_DIFF**2 / mse_loss) / log10
            mse_loss = 0
            for i in range(scales):
                mse_loss += mse(dbs[i], clears[i]) * weights[i]
                fft_loss += FFT_loss(dbs[i], clears[i]) * weights[i]

        # reblurring loss (paper eq.14). weight 0 reproduces the original (dropped).
        for i in range(len(bs)):
            reblur_loss += mse(bs[i], blurrys[i]) * weights[i]

        total_loss = (
            mse_loss
            + ssim_loss
            + loss_opt["fft_weight"] * fft_loss
            + loss_opt["reblur_weight"] * reblur_loss
        )
        return {"mse": total_loss, "psnr": psnr}
    else:
        loss = mse(dbs[-1], clear256)
        psnr = 10 * torch.log(MAX_DIFF**2 / loss) / log10
        return {"mse": loss, "psnr": psnr}


def set_learning_rate(optimizer, it, base_lr, lr_schedule):
    """Apply stepped LR decay from config schedule."""
    lr = base_lr
    for threshold, scale in sorted(lr_schedule, reverse=True):
        if it >= threshold:
            lr = base_lr * scale
            break
    optimizer.param_groups[0]["lr"] = lr


def worker_init_fn(worker_id):
    np.random.seed(10 + worker_id)


def main(cfg, args):
    global mse, mae
    setup_seed(0)

    net_cfg = cfg["net"]
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    loss_cfg = cfg.get("loss", {})
    lr_schedule = [(e, s) for e, s in train_cfg.get("lr_schedule", [])]

    loss_opt = {
        "fft_weight": loss_cfg.get("fft_weight", 0.1),
        "reblur_weight": loss_cfg.get("reblur_weight", 0.1),
        "use_ssim": loss_cfg.get("use_ssim", False),
    }
    print(f"loss: {loss_opt}")

    tb = TensorBoardX(config_filename=args.config, sub_dir=train_cfg["sub_dir"])
    log_file = open(f"{tb.path}/train.log", "w")

    train_dataset = Dataset(
        data_cfg["train_img_path"],
        data_cfg["train_gt_path"],
        crop_size=(data_cfg["crop_size"], data_cfg["crop_size"]),
    )
    test_dataset = TestDataset(data_cfg["test_img_path"], data_cfg["test_gt_path"])

    print(
        f"train: {data_cfg['train_img_path']} | {data_cfg['train_gt_path']} ({len(train_dataset)} pairs)"
    )
    print(
        f"test : {data_cfg['test_img_path']} | {data_cfg['test_gt_path']} ({len(test_dataset)} pairs)"
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=train_cfg["test_batch_size"],
        shuffle=False,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
    )

    mse = torch.nn.MSELoss().cuda()
    mae = torch.nn.L1Loss().cuda()

    nrknet = torch.nn.DataParallel(
        NRKNet(
            num_res=net_cfg["num_res"], num_kernels=net_cfg["num_kernels"], in_ch=net_cfg["in_ch"]
        )
    ).cuda()
    print("Parameters: %.2fM" % (sum(p.nelement() for p in nrknet.parameters()) / 1e6))

    base_lr = train_cfg["learning_rate"]
    assert train_cfg["optimizer"] in ["Adam", "SGD"]
    if train_cfg["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            nrknet.parameters(), lr=base_lr, weight_decay=train_cfg["weight_l2_reg"]
        )
    else:
        optimizer = torch.optim.SGD(
            nrknet.parameters(),
            lr=base_lr,
            weight_decay=train_cfg["weight_l2_reg"],
            momentum=train_cfg["momentum"],
            nesterov=train_cfg["nesterov"],
        )

    last_iter = -1
    if train_cfg["resume"] is not None:
        last_iter = load_model_iter(nrknet, train_cfg["resume"], it=train_cfg["resume_iter"])
        print(f"resume: True | from {train_cfg['resume']} | resumed iter {last_iter}")
    else:
        print("resume: False | training from scratch")

    num_iters = train_cfg["num_iters"]
    val_period = train_cfg["val_period"]
    save_period = train_cfg["save_period"]
    reload_iters = set(train_cfg.get("reload_iters", []))
    best_val_psnr = 0
    best_iter = None
    train_log, val_log = [], []

    global_iter = last_iter + 1
    pbar = tqdm(total=num_iters, initial=global_iter, file=sys.stdout)
    nrknet.train()
    done = False
    while not done:
        for batch in train_loader:
            if global_iter >= num_iters:
                done = True
                break

            if global_iter in reload_iters and best_iter is not None:
                load_model_iter(nrknet, tb.path, it=best_iter)
                optimizer = torch.optim.Adam(
                    nrknet.parameters(), lr=base_lr, weight_decay=train_cfg["weight_l2_reg"]
                )

            set_learning_rate(optimizer, global_iter, base_lr, lr_schedule)
            nrknet.train()

            img = batch["img256"].cuda()
            label = batch["label256"].cuda()
            dbs, bs = nrknet(img, label)
            loss = compute_loss(dbs, bs, label, img, global_iter, "train", lr_schedule, loss_opt)
            optimizer.zero_grad()
            loss["mse"].backward()
            optimizer.step()
            for k in loss:
                loss[k] = float(loss[k].cpu().detach())
            train_log.append(dict(loss))
            for k, v in loss.items():
                tb.add_scalar(k, v, global_iter, "train")
            tb.add_scalar("lr", optimizer.param_groups[0]["lr"], global_iter, "train")

            if global_iter > 0 and global_iter % val_period == 0:
                nrknet.eval()
                with torch.no_grad():
                    for vbatch in tqdm(test_loader, file=sys.stdout, desc="val"):
                        img = vbatch["img256"].cuda()
                        label = vbatch["label256"].cuda()
                        # val images are full-size (arbitrary HxW); pad to /16 for the
                        # net, then crop the output back to match the label.
                        img_pad, crop = pad_to_multiple(img, 16)
                        dbs, bs = nrknet(img_pad, phase="test")
                        dbs = [crop(dbs[-1])]
                        loss = compute_loss(
                            dbs, bs, label, img, global_iter, "test", lr_schedule, loss_opt
                        )
                        val_log.append({k: float(v.cpu().detach()) for k, v in loss.items()})

                train_dict = {k: float(np.mean([d[k] for d in train_log])) for k in train_log[0]}
                val_dict = {k: float(np.mean([d[k] for d in val_log])) for k in val_log[0]}
                for k, v in val_dict.items():
                    tb.add_scalar(k, v, global_iter, "val")

                prev_best_psnr, prev_best_iter = best_val_psnr, best_iter
                is_best = val_dict["psnr"] > best_val_psnr
                if is_best:
                    best_iter = global_iter
                    best_val_psnr = val_dict["psnr"]
                    save_model_iter(nrknet, tb.path, global_iter)
                    save_optimizer_iter(optimizer, nrknet, tb.path, global_iter)

                train_log.clear()
                val_log.clear()

                prev_str = (
                    f"prev best psnr {prev_best_psnr:.5f} @ iter {prev_best_iter}"
                    if prev_best_iter is not None
                    else "prev best none"
                )
                if is_best:
                    best_str = f"NEW BEST psnr {best_val_psnr:.5f} (saved iter{global_iter:08d}) | {prev_str}"
                else:
                    best_str = f"best psnr {best_val_psnr:.5f} @ iter {best_iter}"

                msg = f"iter {global_iter} | train: " + " ".join(
                    f"{k} {v:.5f}" for k, v in train_dict.items()
                )
                msg += " | val: " + " ".join(f"{k} {v:.5f}" for k, v in val_dict.items())
                msg += " | " + best_str
                tqdm.write(msg, file=sys.stdout)
                log_file.write(msg + "\n")
                log_file.flush()
                nrknet.train()

            # periodic snapshot (independent of best-val checkpointing)
            if global_iter > 0 and global_iter % save_period == 0:
                save_model_iter(nrknet, tb.path, global_iter)
                save_optimizer_iter(optimizer, nrknet, tb.path, global_iter)

            global_iter += 1
            pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/nrknet/dpdd.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg, args)
