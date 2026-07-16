# Deblurring Model Training and Evaluation

We provide code for training and evaluating defocus deblurring models.


## Environment

Dependencies are managed with **uv** package manager.
Inside the container, commands can be run directly with `uv run`.
To synchronize the environment manually, run `uv sync`.

For convenience, we also support short commands via [just](https://github.com/casey/just).


## Supported Models

This repository supports the following four deblurring models:
- [INIKNet](https://github.com/xinyao240/INIKNet)
- [NAFNet](https://github.com/megvii-research/NAFNet)
- [NRKNet](https://github.com/csZcWu/NRKNet)
- [Restormer](https://github.com/swz30/Restormer)


## Pretrained Weights

The default settings point to the weight files under the `weights/` directory.
Refer to `configs/test/weights.yaml` and place the downloaded weights into given paths.
Otherwise, update paths in `configs/test/weights.yaml` to your own paths.


### Models Trained on DPDD

Official DPDD-trained weights are available for all models except **NAFNet**:
- [**INIKNet**](https://github.com/xinyao240/INIKNet/blob/10c2a3262e5b23d0aaa99b763f6a4eba24c01cbb/checkpoints/INIKNet.pth)
- [**NRKNet**](https://github.com/csZcWu/NRKNet/blob/90fad3677a78cd013755e2b425e177bc5123c2a0/NRKNet_epoch3846.pth)
- [**Restormer**](https://github.com/swz30/Restormer/tree/68dc6ac472db26f16361150cb7a96a1bc87da93f/Defocus_Deblurring/pretrained_models)

According to `weights.yaml`, you may need to rename the downloaded files (e.g., `nrknet-train-dpdd.pth`).

**NAFNet** does not provide official DPDD-trained weights.
We trained it ourselves and provide the resulting weights [here](https://drive.google.com/file/d/1STOT2AC6yUOHdLCPw1FebvsVrFcWW_qu/view?usp=drive_link).


### Models Trained on SYNDOF and CLDefocus

Because official SYNDOF- and CLDefocus-trained weights are unavailable, we trained all four models on both datasets.
The resulting weights are available [here](https://drive.google.com/drive/folders/1kmrJ8AmMo3QjIj9kijAgK6-VlorKosGp?usp=drive_link).


## Download Image Datasets

### Existing Datasets

We use the following three existing benchmarks for evaluation:
- **DPDD**: [official repo](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel), download "All images used for training/testing"
- **RealDOF**: [official repo](https://github.com/codeslake/IFAN)
- **RTF Dataset**: NOTE: We could not find an official public download link, but it can be found [here](https://github.com/codeslake/DMENet/tree/d844e5a6ad3e7a1c9157d50935de1a6eb6bc4bf8/evaluation/RTF)

To use **SYNDOF** for training, download it [here](https://junyonglee.me/projects/DMENet/).
It requires reorganization before use, which can be done easily using [our script](#data-setup).


### CLDefocus

> [!NOTE]
> The licensing terms for redistributing **CLDefocus** are not yet confirmed.
> We have contacted the authors and are awaiting their response.
> Once the terms are clarified, we will release the dataset here.
> We apologize for the delay.
> In the meantime, you can reproduce CLDefocus yourself using our synthesis code.

CLDefocus is a synthetic defocus deblurring dataset generated using our pipeline.
It consists of 40,000/1,000/1,000 pairs for train/validation/test splits.
You can reproduce it using the [synthesis pipeline](../datasyn).


### Smartphone Captures

[Google Drive](https://drive.google.com/file/d/1N8GndQW2bPq-hUfDPyKOPIvAbnHW1dnn/view?usp=drive_link)

This set contains photos with substantial defocus blur captured using a Galaxy S24, as used in Sec. S6.
There are no sharp ground-truth images.
The images were captured without a specialized acquisition or post-processing.


## Data Setup

For our code, downloaded datasets should be properly reorganized.
Due to inconsistent layouts across datasets, it is often tedious.
We recommend the following way.

First, make `data-paths.yaml`, copying from `data-paths.example.yaml`:
```bash
cp data-paths.example.yaml data-paths.yaml
```

Then edit each path to your locations of **original** downloads.
e.g., `DPDD: /data/DPDD/dd_dp_dataset_png`.
Note that it should be accessible in the container.

Next, make symbolic views of each dataset under `data/`.
We provide a script for this.
First, run:
```bash
# Using the Justfile
just setup-data --dry-run

# Or run the command manually
uv run scripts/setup_data.py --dry-run
```
Then you will see what views will be created.
You can pick a subset of datasets with `--only` option like:
```bash
just setup-data --dry-run --only DPDD RealDOF
```
After confirmation, run the same command without `--dry-run`.
Then the views are created inside `data/`.
The default configurations are set to use those data, so you do not need to change anything.


## Evaluation

The evaluation pipeline separates model inference from metric computation.

By default, model outputs are saved under `experiments/eval/img/`.

Metrics are computed directly from the saved output images, without rerunning deblurring models.
This separation ensures consistent evaluation using [pyiqa](https://github.com/chaofengc/IQA-PyTorch).
By default, metric results are saved under `experiments/eval/met/`.

Both directories use the same nested organization by model, training dataset, and test dataset:
```text
experiments/eval/
├── img/
│   ├── model-A/
│   │   ├── train-set-A/
│   │   │   ├── test-set-A/
│   │   │   ├── ...
│   │   │   └── test-set-Z/
│   │   ├── ...
│   │   └── train-set-Z/
│   │       └── ...
│   ├── ...
│   └── model-Z/
│       └── ...
└── met/
    └── ...
```


### Quick Test

This quick test evaluates CLDefocus-trained NRKNet on three real-world benchmark datasets.
First, make sure that you have `weights/nrknet-train-ours.pth` [(download here)](https://drive.google.com/file/d/1PXrjjljIoD-8IRcKBce4vYIcs19wcebL/view?usp=drive_link).
Then run:
```bash
# Using the Justfile
just eval-quick

# Or run the command manually
uv run scripts/run_tests.py \
    --list configs/test/quick.csv \
    --metrics-file configs/metrics/simple.txt
```

After the evaluation, generated images are under `experiments/eval/img/nrknet/`, and metric results are under `experiments/eval/met/nrknet/`.


### Evaluate Specific Models and Datasets

You can evaluate any supported model on datasets beyond those used in our experiments.

Create an evaluation-list CSV by following the examples in `configs/test/`, such as `full.csv`.
You can specify a metric list using files such as `configs/metrics/full.txt`.
Metric names must match the identifiers supported by [pyiqa](https://github.com/chaofengc/IQA-PyTorch).

Make sure that required pretrained weights are placed in `weights/`.
Then run the evaluation with:
```bash
uv run scripts/run_tests.py \
    --list path/to/your/list.csv \
    --metrics-file path/to/your/metrics.txt
```

To run a single evaluation entry without creating CSV or metric-list files, use:
```bash
just eval-once  nafnet train_ours realdof  --metrics psnr ssim lpips
```

> [!CAUTION]
> Evaluating Restormer on RealDOF at its original resolution requires at least 40 GB of GPU memory.
> On GPUs with less memory, you may downscale the images.

If you are interested in practical uses, we recommend using **NAFNet** as it often produces the most plausible results.


### Full Evaluation

To reproduce the complete metric table across all evaluated models and datasets in Table S4, run:
```bash
# Using the Justfile
just eval-full

# Or run the command manually
uv run scripts/run_tests.py \
    --list configs/test/full.csv \
    --metrics-file configs/metrics/full.txt
```


### Evaluate the Smartphone Captures

To evaluate the models on the [smartphone captures](#smartphone-captures), run:
```bash
# Using the Justfile
just eval-s24

# Or run the command manually
uv run scripts/run_tests.py \
    --list configs/test/s24.csv \
    --metrics-file configs/metrics/nr.txt
```


### Evaluate a Custom Dataset

To evaluate a custom dataset, create a dataset YAML file and reference it from your evaluation-list CSV.
Follow the provided dataset YAML and evaluation CSV files as templates.


## Training

Each training script requires a model-specific configuration file.
Examples are provided under `configs/train/{model}/`.

To train **NRKNet** or **INIKNet**, run the corresponding commands.

**NRKNet**:
```bash
uv run scripts/train_nrknet.py \
    --config path/to/config.yaml
```

**INIKNet**:
```bash
uv run scripts/train_iniknet.py \
    --config path/to/config.yaml
```

**Restormer** and **NAFNet** use distributed launch through `torchrun`.
Set the environment variable `${GPU_COUNT}` to the number of GPUs.

**Restormer**:
```bash
uv run torchrun \
    --nproc_per_node="${GPU_COUNT}" \
    scripts/train_restormer.py \
    --launcher pytorch \
    -opt path/to/config.yml 
```

**NAFNet**:
```bash
uv run torchrun \
    --nproc_per_node="${GPU_COUNT}" \
    scripts/train_nafnet.py \
    --launcher pytorch \
    -opt path/to/config.yml 
```


## Notes on the Evaluation Results

On the **DPDD test set**, CLDefocus-trained models often leave a substantial amount of blur uncorrected.
In contrast, DPDD-trained models generally produce stable results.
We cautiously attribute this behavior to a large mismatch between the blur distribution in CLDefocus and that of DPDD.

On **RealDOF**, CLDefocus-trained models generally produce more plausible outputs than their DPDD-trained counterparts across most evaluated cases (although PSNR/SSIM are usually lower).
This suggests that models trained only on the DPDD training set may generalize poorly to defocus blur from other cameras.

Models trained on another synthetic dataset, **SYNDOF**, consistently perform worse than CLDefocus-trained models on all benchmarks.
This does not appear to result merely from optimization failure: SYNDOF-trained models remove blur effectively when evaluated on SYNDOF inputs.
This suggests that the simplified blur model used to construct SYNDOF may not provide sufficient optical diversity or realism for generalizing to real defocus blur.
More broadly, this result might help explain why synthetic data has seen limited adoption in learning-based defocus deblurring.

We recommend testing the models on images from your target domain in addition to the provided benchmarks.
This can help determine which training dataset is most suitable for your application.
Moreover, you might find what the potential improvements for CLDefocus are, hopefully leading to follow-up research topics.
