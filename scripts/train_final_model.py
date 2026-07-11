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
    InlineLogger,
    build_model,
    compile_model,
    load_images,
    merge_tabular_with_images,
    reorder_dataframe,
    train_model,
)


def train(epochs: int = 80, batch_size: int = 128, image_size=(224, 224)) -> None:
    data_dir_path = PROJECT_ROOT / "data"

    BATCH_SIZE = batch_size
    IMAGE_SIZE = image_size
    IMAGE_SHAPE = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    EPOCHS = epochs

    metrics = [
        keras.metrics.F1Score(average="macro", name="f1_score"),
        keras.metrics.F1Score(average="weighted", name="f1_score_weighted")
    ]

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(PROJECT_ROOT / "checkpoint.keras"),
            monitor="val_loss",
            verbose=0,
        ),
        keras.callbacks.CSVLogger(str(PROJECT_ROOT / "metrics.csv")),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=1,
            patience=5,
            restore_best_weights=True,
        ),
        InlineLogger()
    ]

    print(f"✅ Loading images from {data_dir_path}... and resizing to {IMAGE_SIZE}")

    train_ds, val_ds, test_ds, class_names = load_images(
        folder_name=str(data_dir_path),
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        interpolation="bilinear",
        shuffle=False,
        random_seed=42)

    print(f"✅ Loading tabular metadata")

    df_train = pd.read_csv(data_dir_path / "metadata_train.csv")
    df_val = pd.read_csv(data_dir_path / "metadata_val.csv")
    df_test = pd.read_csv(data_dir_path / "metadata_test.csv")

    df_train_reordered = reorder_dataframe(df_train, data_dir_path / "train")
    df_val_reordered = reorder_dataframe(df_val, data_dir_path / "val")
    df_test_reordered = reorder_dataframe(df_test, data_dir_path / "test")

    for split_name, split_df in (
        ("train", df_train_reordered),
        ("val", df_val_reordered),
        ("test", df_test_reordered),
    ):
        missing_metadata = split_df["phylum"].isna().sum()
        if missing_metadata:
            raise ValueError(
                f"{missing_metadata} images in '{split_name}' do not have matching phylum metadata."
            )

    phylum_columns = pd.get_dummies(df_train_reordered["phylum"]).columns.tolist()

    train_tabular = (
        pd.get_dummies(df_train_reordered["phylum"])
        .reindex(columns=phylum_columns, fill_value=0)
        .to_numpy(dtype="float32")
    )
    val_tabular = (
        pd.get_dummies(df_val_reordered["phylum"])
        .reindex(columns=phylum_columns, fill_value=0)
        .to_numpy(dtype="float32")
    )
    test_tabular = (
        pd.get_dummies(df_test_reordered["phylum"])
        .reindex(columns=phylum_columns, fill_value=0)
        .to_numpy(dtype="float32")
    )

    train_ds = merge_tabular_with_images(train_ds, train_tabular, BATCH_SIZE, shuffle=True, shuffle_buffer=32)
    val_ds = merge_tabular_with_images(val_ds, val_tabular, BATCH_SIZE)
    test_ds = merge_tabular_with_images(test_ds, test_tabular, BATCH_SIZE)

    print(f"✅ Building and compiling model EfficientNetV2B1")

    best_model = build_model(
        input_shape=IMAGE_SHAPE,
        use_tabular=True,
        tabular_input_shape=(len(phylum_columns),),
        project_tabular=False,
        use_data_augmentation=None,
        base_model_name="EfficientNetV2B1",
        freeze_base=True,
        spatial_reduction="avg",
        dropout_rate=0.1,
        use_batch_norm=False,
        n_dense_extra_layers=0,
        n_dense_units=0,
        activation="leaky_relu",
        kernel_regularizer_l2=2e-4,
        n_classes=len(class_names),
        output_activation="softmax",
    )

    best_model = compile_model(
        best_model,
        optimizer_name="RMSprop",
        learning_rate=keras.optimizers.schedules.CosineDecay(
            1e-3, train_tabular.shape[0] / BATCH_SIZE * EPOCHS * 0.6, alpha=0.2),
        weight_decay=2e-4,
        loss_name="CategoricalCrossentropy",
        label_smoothing=0.0,
        metrics=metrics
    )

    best_model.summary()

    _ = train_model(
        best_model,
        train_ds,
        val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=0
    )

    print(f"✅ Evaluating model on test set")
    evaluation_dict = best_model.evaluate(test_ds, return_dict=True, verbose=0)
    print(evaluation_dict)


def main() -> None:
    parser = ArgumentParser(prog="efficientnetv2b1 training")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_size", type=str, required=False, help="Format: HEIGHT,WIDTH (e.g. 224,224)")

    args = parser.parse_args()

    # Parse image size string to tuple
    if args.image_size:
        image_size = tuple(map(int, args.image_size.split(',')))
    else:
        image_size = (224, 224)

    train(args.epochs, args.batch_size, image_size=image_size)

if __name__ == "__main__":
    main()
