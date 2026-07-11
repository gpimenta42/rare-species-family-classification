from argparse import ArgumentParser
import random
import shutil
from pathlib import Path
from typing import Union
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def split_images(source_dir: Union[str, Path], output_base: Union[str, Path], split_ratios=None):
    if split_ratios is None:
        split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}

    random.seed(42)
    source_dir = Path(source_dir)
    output_base = Path(output_base)

    for species_dir in source_dir.iterdir():
        if species_dir.is_dir() and species_dir.name != "metadata.csv":
            print(f"Processing {species_dir.name} with {len(list(species_dir.glob('*')))} images")
            images = list(species_dir.glob("*"))
            random.shuffle(images)

            total = len(images)
            n_train = int(total * split_ratios["train"])
            n_val = int(total * split_ratios["val"])

            split_map = {
                "train": images[:n_train],
                "val": images[n_train:n_train + n_val],
                "test": images[n_train + n_val:]
            }

            for split, images_for_split in split_map.items():
                dest_dir = output_base / split / species_dir.name
                dest_dir.mkdir(parents=True, exist_ok=True)
                for img_path in images_for_split:
                    shutil.copy(img_path, dest_dir / img_path.name)
    print("✅ Image split complete!")


def split_metadata(metadata_csv: Union[str, Path], output_base: Union[str, Path]):
    df_metadata = pd.read_csv(metadata_csv)
    df_metadata["file_path"] = df_metadata["file_path"].astype(str)
    output_base = Path(output_base)

    def get_split_paths(split_folder):
        split_paths = []
        for path in Path(split_folder).rglob("*.*"):
            rel_path = path.relative_to(output_base).as_posix()
            normalized_path = "/".join(rel_path.split("/")[1:])  # remove split (train/val/test)
            split_paths.append(normalized_path)
        return set(split_paths)

    train_paths = get_split_paths(output_base / "train")
    val_paths = get_split_paths(output_base / "val")
    test_paths = get_split_paths(output_base / "test")

    df_train = df_metadata[df_metadata["file_path"].isin(train_paths)].copy()
    df_val = df_metadata[df_metadata["file_path"].isin(val_paths)].copy()
    df_test = df_metadata[df_metadata["file_path"].isin(test_paths)].copy()

    df_train.to_csv(output_base / "metadata_train.csv", index=False)
    df_val.to_csv(output_base / "metadata_val.csv", index=False)
    df_test.to_csv(output_base / "metadata_test.csv", index=False)

    print(f"✅ Metadata files saved! Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")


def main() -> None:
    parser = ArgumentParser(description="Split rare species images and metadata into train/val/test folders.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "rare_species 1",
        help="Extracted dataset directory. Defaults to '<repo>/rare_species 1'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Output directory for prepared splits. Defaults to '<repo>/data'.",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=None,
        help="Metadata CSV path. Defaults to '<source-dir>/metadata.csv'.",
    )
    args = parser.parse_args()

    metadata_file = args.metadata_file or args.source_dir / "metadata.csv"

    split_images(args.source_dir, args.output_dir)
    split_metadata(metadata_file, args.output_dir)


if __name__ == "__main__":
    main()
