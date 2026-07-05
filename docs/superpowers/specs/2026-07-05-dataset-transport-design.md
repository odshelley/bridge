# Design: Data-to-data transport datasets (CIFAR cat→dog, AFHQ cat→dog)

Date: 2026-07-05
Status: draft, awaiting review

## Goal

Extend the bridge-diffusion library so the Gaussian bridge can be trained as a
data-to-data transport between two image distributions, and add the two
concrete experiments:

1. **CIFAR-10 cat→dog** at 32×32 — cheap stepping stone, runs locally.
2. **AFHQ cat→dog** at 64×64 — headline transport experiment, matching the
   benchmark used by Rectified Flow / DSBM / DDBM.

Deliverable includes runnable configs and a step-by-step training guide.

## Non-goals

- Poisson-bridge transport (integer bridge stays noise→data on MNIST).
- CelebA or ImageNet support.
- Non-independent couplings (OT minibatch coupling, reflow). The paired loader
  samples source and target independently, which is the standard first-pass
  coupling in the literature. The design leaves room to add couplings later.

## Background: what the code already supports

- `BridgeDiffusion.compute_training_loss(x, y)` is already general in `x`; the
  maths never assumes a Gaussian prior (`models/bridge.py`).
- The sampler already accepts arbitrary start points `x0` and only falls back
  to `randn` when none are given (`sampling/sampler.py`).
- The only place noise is hardcoded is the trainer: `y = batch[0]`,
  `x = _sample_prior(...) = randn_like(y)` (`training/trainer.py:24-39,210-214`).

So the work is: dataset entries, a paired loader, one trainer branch, configs,
and evaluation/sampling plumbing for source-conditioned generation.

## Design

### 1. Config (`config/config.py`)

Extend `DataConfig` with flat, backwards-compatible fields:

```python
@dataclass
class DataConfig:
    dataset: Literal["mnist", "cifar10", "afhq"] = "mnist"   # target distribution
    classes: list[str] | None = None          # filter target to these classes
    source_dataset: Literal["mnist", "cifar10", "afhq"] | None = None
    source_classes: list[str] | None = None   # filter source to these classes
    # ... existing fields unchanged (data_dir, image_size, num_workers, ...)
```

Transport mode is active iff `source_dataset is not None`. All existing
configs remain valid with unchanged behaviour. `ExperimentConfig.from_yaml`
needs no structural change (flat fields pass straight through `DataConfig(**...)`).

Validation in `__post_init__`: `source_classes` without `source_dataset` is an
error; class names are validated against the dataset at load time in
`datasets.py` (fail fast with the list of valid names).

### 2. Data loading (`data/datasets.py`)

**AFHQ entry.** torchvision `ImageFolder` rooted at
`{data_dir}/afhq/train` and `{data_dir}/afhq/val` (class subfolders `cat/`,
`dog/`, `wild/`). If the directory is missing, raise with a message pointing
at `scripts/download_afhq.sh`. Transforms: `Resize(image_size)` →
`CenterCrop(image_size)` → `RandomHorizontalFlip()` (train only) →
`ToTensor()` → `Normalize` to [-1, 1]. AFHQ source images are 512×512 so the
resize path also covers running AFHQ at 32×32 for smoke tests.

**Class filtering.** A helper `filter_classes(dataset, class_names) -> Subset`
that works for both `CIFAR10` (via `dataset.classes` / `dataset.targets`) and
`ImageFolder` (via `class_to_idx` / `samples`). Unknown class names raise
`ValueError` listing valid options.

**Paired dataset.** 

```python
class PairedDataset(Dataset):
    """Independent coupling of a source and a target dataset.

    __len__ = len(target). __getitem__(i) returns (x_source, y_target) where
    y_target is item i and x_source is drawn uniformly at random (torch.randint,
    so per-worker seeding behaves under num_workers > 0). Labels are dropped.
    """
```

`get_dataloader` builds the target dataset as today; when
`config.source_dataset` is set it also builds the source dataset (with
`source_classes` filtering) and wraps both in `PairedDataset`.

`get_data_info` gains an `afhq` branch (3 channels; train sizes cat 5153 /
dog 4739 / wild 4738; val ≈ 500 per class) and reports filtered sizes when
`classes` is set.

### 3. Trainer (`training/trainer.py`)

One branch keyed off config, not tensor shape guessing:

```python
if self.transport:            # config.data.source_dataset is not None
    x, y = batch[0].to(device), batch[1].to(device)
else:
    y = batch[0].to(device)
    x = _sample_prior(self.model, y)
loss = self.model.compute_training_loss(x, y)
```

`_log_samples` in transport mode starts the sampler from a fixed batch of
held-out source (val) images instead of noise, so the MLflow sample grids show
actual cat→dog translations during training.

### 4. Sampling and evaluation

- `cli.py sample` gains `--source-dir PATH` (a folder of images used as `x0`,
  loaded with the eval transform). When omitted in transport mode, it falls
  back to the val split of the configured source dataset. Noise fallback
  remains for non-transport checkpoints. Generated images are saved alongside
  their source filenames so translations are inspectable pairwise.
- FID: unchanged mechanically (`evaluate --real-dir ... --generated-dir ...`).
  A small helper script `scripts/export_val_images.py` dumps a dataset/class
  val split to a folder of PNGs (needed once per dataset to create the
  real-dog reference folder).

### 5. AFHQ download (`scripts/download_afhq.sh`)

Fetches the official AFHQ release used by stargan-v2 (Dropbox zip, ~500 MB),
unzips to `data/afhq/{train,val}/{cat,dog,wild}`, and prints counts. The URL
is pinned in the script with a fallback note (Kaggle `afhq` mirror /
HuggingFace `huggan/AFHQ`) in case the Dropbox link rots. Verify the link at
implementation time.

### 6. Configs (`configs/`)

| File | dataset | size | batch | steps | notes |
|---|---|---|---|---|---|
| `cifar_cat2dog.yaml` | cifar10 cat→dog | 32 | 128 | 50k | UNet as in `cifar10.yaml` |
| `cifar_cat2dog_smoke.yaml` | cifar10 cat→dog | 32 | 16 | 200 | minutes on MPS |
| `afhq_cat2dog_64.yaml` | afhq cat→dog | 64 | 64 | 150k | GPU-sized, see below |
| `afhq_cat2dog_smoke.yaml` | afhq cat→dog | 32 | 16 | 200 | minutes on MPS |

AFHQ 64×64 UNet: `block_out_channels (128, 256, 384, 512)`, 2 layers per
block, attention in the two lowest-resolution blocks, EMA 0.9999, lr 1e-4.
Bridge params (`T`, `eps`) start from the CIFAR values; tuning `T` for
transport is an open experimental question, not fixed by this design.

### 7. Testing

- `tests/test_data.py`: `filter_classes` on CIFAR-10 and on a tmpdir
  `ImageFolder`; `PairedDataset` length, shapes, value range, label dropping;
  config round-trip (`from_yaml`/`to_yaml`) with the new fields; validation
  errors (unknown class, `source_classes` without `source_dataset`).
- `tests/test_trainer.py` (or extend existing): transport branch smoke test —
  tiny synthetic ImageFolder pair, a few training steps, loss is finite.
- Smoke configs double as end-to-end tests: train → sample → evaluate on
  ~200 steps.

### 8. Error handling

- Missing AFHQ data dir → actionable error naming the download script.
- Unknown dataset/class names → `ValueError` listing valid options.
- `sample --source-dir` on a non-transport checkpoint → warning, proceeds
  (mathematically fine; the model was just not trained for it).

## What you (Osian) will need to do to train

1. **Smoke test locally (MPS, minutes):**
   ```bash
   uv run bridge-diffusion train --config configs/cifar_cat2dog_smoke.yaml
   uv run bridge-diffusion train --config configs/afhq_cat2dog_smoke.yaml  # needs AFHQ downloaded
   ```
2. **CIFAR cat→dog for real (local, overnight on MPS or ~2-3 h on one GPU):**
   ```bash
   uv run bridge-diffusion train --config configs/cifar_cat2dog.yaml
   ```
3. **AFHQ (needs a CUDA GPU; roughly 1-2 days on a single A100/4090 at 150k steps):**
   ```bash
   bash scripts/download_afhq.sh
   uv run bridge-diffusion train --config configs/afhq_cat2dog_64.yaml
   ```
4. **Sample translations and evaluate:**
   ```bash
   uv run bridge-diffusion sample --checkpoint outputs/.../model.pt --num-steps 100 --use-ema
   uv run python scripts/export_val_images.py --dataset afhq --classes dog --out data/fid_ref/afhq_dog
   uv run bridge-diffusion evaluate --real-dir data/fid_ref/afhq_dog --generated-dir outputs/samples
   ```
5. Monitor via MLflow as today (sample grids will show cat→dog translations).

## Open questions deferred to experiment time

- Best bridge terminal time `T` for transport (start with CIFAR value, sweep).
- Whether 64×64 AFHQ FID is reported against val dogs only (standard) or
  train+val; follow DDBM's protocol when writing the paper section.
