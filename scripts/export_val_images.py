"""Export a dataset's val split (optionally class-filtered) as PNG files.

Used to build the reference folder for FID evaluation, e.g.:

    uv run python scripts/export_val_images.py \
        --dataset afhq --classes dog --out data/fid_ref/afhq_dog
"""

import argparse
from pathlib import Path

from torchvision.utils import save_image

from bridge_diffusion.config import DataConfig
from bridge_diffusion.data import get_dataset


def export_val_images(
    dataset: str,
    classes: list[str] | None,
    out: Path,
    image_size: int = 64,
    data_dir: Path = Path("./data"),
) -> int:
    """Write the val split of a dataset to out/ as PNGs in [0, 1].

    Args:
        dataset: Dataset name (mnist, cifar10, afhq).
        classes: Optional class names to keep.
        out: Output directory.
        image_size: Image size (must match the generated samples for FID).
        data_dir: Dataset root directory.

    Returns:
        Number of images written.
    """
    config = DataConfig(
        dataset=dataset,
        classes=classes,
        image_size=image_size,
        data_dir=data_dir,
    )
    val_dataset = get_dataset(config, train=False)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(len(val_dataset)):
        image = val_dataset[i][0]
        save_image((image.clamp(-1, 1) + 1) / 2, out / f"{i:05d}.png")
    return len(val_dataset)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["mnist", "cifar10", "afhq"])
    parser.add_argument("--classes", nargs="*", default=None, help="Class names to keep")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    args = parser.parse_args()

    count = export_val_images(
        dataset=args.dataset,
        classes=args.classes,
        out=args.out,
        image_size=args.image_size,
        data_dir=args.data_dir,
    )
    print(f"Wrote {count} images to {args.out}")


if __name__ == "__main__":
    main()
