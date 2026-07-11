from argparse import ArgumentParser
from pathlib import Path
from utils_ import *

def train(epochs: int = 80, batch_size: int = 128, image_size=(224, 224)) -> None:
    root_dir_path = Path(__file__).parent
    data_dir_path = root_dir_path / "data"

    BATCH_SIZE = batch_size
    IMAGE_SIZE = image_size
    IMAGE_SHAPE = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    EPOCHS = epochs

    metrics = [
        keras.metrics.F1Score(average="macro", name="f1_score"),
        keras.metrics.F1Score(average="weighted", name="f1_score_weighted")
    ]

    callbacks = [
        keras.callbacks.ModelCheckpoint("checkpoint.keras", monitor="val_loss", verbose=0),
        keras.callbacks.CSVLogger("metrics.csv"),
        keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True),
        InlineLogger()
    ]

    print(f"✅ Loading images from {data_dir_path}... and resizing to {IMAGE_SIZE}")

    train_ds, val_ds, test_ds, class_names = load_images(
        folder_name="data",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        interpolation="bilinear",
        shuffle=False,
        random_seed=42)

    print(f"✅ Loading tabular metadata")

    df_train = pd.read_csv("data/metadata_train.csv")
    df_val = pd.read_csv("data/metadata_val.csv")
    df_test = pd.read_csv("data/metadata_test.csv")

    df_train_reordered = reorder_dataframe(df_train, "data/train")
    df_val_reordered = reorder_dataframe(df_val, "data/val")
    df_test_reordered = reorder_dataframe(df_test, "data/test")

    phylum_columns = pd.get_dummies(df_train_reordered["phylum"]).columns.tolist()

    train_tabular = pd.get_dummies(df_train_reordered["phylum"])[phylum_columns].to_numpy()
    val_tabular = pd.get_dummies(df_val_reordered["phylum"])[phylum_columns].to_numpy()
    test_tabular = pd.get_dummies(df_test_reordered["phylum"])[phylum_columns].to_numpy()

    train_ds = merge_tabular_with_images(train_ds, train_tabular, BATCH_SIZE, shuffle=True, shuffle_buffer=32)
    val_ds = merge_tabular_with_images(val_ds, val_tabular, BATCH_SIZE)
    test_ds = merge_tabular_with_images(test_ds, test_tabular, BATCH_SIZE)

    print(f"✅ Building and compiling model EfficientNetV2B1")

    best_model = build_model(
        input_shape=IMAGE_SHAPE,
        use_tabular=True,
        tabular_input_shape=(5,),
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
        n_classes=202,
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
    parser.add_argument("--epochs", type=int, required=False)
    parser.add_argument("--batch_size", type=int, required=False)
    parser.add_argument("--image_size", type=str, required=False, help="Format: HEIGHT,WIDTH (e.g. 224,224)")

    args = parser.parse_args()

    # Parse image size string to tuple
    if args.image_size:
        image_size = tuple(map(int, args.image_size.split(',')))
    else:
        image_size = (224, 224)

    train(args.epochs or 80, args.batch_size or 128, image_size=image_size)

if __name__ == "__main__":
    main()