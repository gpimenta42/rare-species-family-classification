from argparse import ArgumentParser
from pathlib import Path
import sys

import pandas as pd
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rare_species_classification.utils import (
    merge_tabular_with_images_main,
    reorder_dataframe,
)


def parse_image_size(value: str) -> tuple[int, int]:
    height, width = value.split(",", maxsplit=1)
    return int(height), int(width)


def load_split_dataset(data_dir: Path, split: str, batch_size: int, image_size: tuple[int, int]):
    dataset = keras.utils.image_dataset_from_directory(
        data_dir / split,
        label_mode="categorical",
        batch_size=batch_size,
        image_size=image_size,
        interpolation="bilinear",
        shuffle=False,
    )
    class_names = dataset.class_names
    return dataset.prefetch(1), class_names


def load_split_tabular(data_dir: Path, split: str):
    train_metadata = pd.read_csv(data_dir / "metadata_train.csv")
    split_metadata = pd.read_csv(data_dir / f"metadata_{split}.csv")
    split_metadata_reordered = reorder_dataframe(split_metadata, data_dir / split)

    missing_metadata = split_metadata_reordered["phylum"].isna().sum()
    if missing_metadata:
        raise ValueError(
            f"{missing_metadata} images in '{split}' do not have matching phylum metadata."
        )

    phylum_columns = pd.get_dummies(train_metadata["phylum"]).columns.tolist()
    split_tabular = (
        pd.get_dummies(split_metadata_reordered["phylum"])
        .reindex(columns=phylum_columns, fill_value=0)
        .to_numpy(dtype="float32")
    )
    return split_tabular


def evaluate_saved_model(
    model_path: Path,
    data_dir: Path,
    split: str = "test",
    batch_size: int = 128,
    image_size: tuple[int, int] = (224, 224),
) -> dict:
    image_dataset, class_names = load_split_dataset(data_dir, split, batch_size, image_size)
    tabular_data = load_split_tabular(data_dir, split)

    evaluation_dataset = merge_tabular_with_images_main(
        image_dataset,
        tabular_data,
        batch_size,
    )

    model = keras.models.load_model(model_path, compile=False)
    model.compile(
        optimizer=keras.optimizers.RMSprop(),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.0),
        metrics=[
            keras.metrics.F1Score(average="macro", name="f1_score"),
            keras.metrics.F1Score(average="weighted", name="f1_score_weighted"),
        ],
    )

    print(f"Evaluating {model_path} on '{split}' split with {len(class_names)} classes.")
    return model.evaluate(evaluation_dataset, return_dict=True, verbose=1)


def main() -> None:
    parser = ArgumentParser(description="Evaluate the saved final model on a prepared dataset split.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "best_model.h5")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=parse_image_size, default=(224, 224), help="Format: HEIGHT,WIDTH")
    args = parser.parse_args()

    results = evaluate_saved_model(
        model_path=args.model,
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    for metric, value in results.items():
        print(f"{metric}: {value:.6f}")


if __name__ == "__main__":
    main()
