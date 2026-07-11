
from tensorflow import keras
import keras_cv

from keras.utils import image_dataset_from_directory
import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from keras.applications import EfficientNetV2M

from keras.layers import GlobalAveragePooling2D, Dense, Dropout

from keras import Input, Model

from keras.optimizers import RMSprop

from keras.losses import CategoricalCrossentropy

from keras.metrics import CategoricalAccuracy, AUC, F1Score

from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

from keras.optimizers.schedules import CosineDecayRestarts
from pathlib import Path

# !pip install keras-cv

# ---------------------------------------------------------
# Load images; resize
# ---------------------------------------------------------

def load_images(
    folder_name,
    batch_size,
    image_size,
    interpolation,
    shuffle=False,
    random_seed=42
):
    train_ds = keras.utils.image_dataset_from_directory(
        f"{folder_name}/train",
        label_mode = "categorical",
        batch_size = batch_size,
        image_size = image_size,
        interpolation = interpolation,
        shuffle=shuffle,
        seed=random_seed,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        f"{folder_name}/val",
        label_mode = "categorical",
        batch_size = batch_size,
        image_size = image_size,
        interpolation = interpolation,
        shuffle=shuffle,
    )

    test_ds = keras.utils.image_dataset_from_directory(
        f"{folder_name}/test",
        label_mode = "categorical",
        batch_size = batch_size,
        image_size = image_size,
        interpolation = interpolation,
        shuffle=shuffle,
    )

    class_names = train_ds.class_names

    train_dataset = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset, class_names




def load_images_main(
    folder_name,
    batch_size,
    image_size,
    interpolation,
    shuffle=False,
    random_seed=42
):
    train_ds = keras.utils.image_dataset_from_directory(
        f"{folder_name}/train",
        label_mode="categorical",
        batch_size=batch_size,
        image_size=image_size,
        interpolation=interpolation,
        shuffle=shuffle,
        seed=random_seed,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        f"{folder_name}/val",
        label_mode="categorical",
        batch_size=batch_size,
        image_size=image_size,
        interpolation=interpolation,
        shuffle=shuffle,
    )

    test_ds = keras.utils.image_dataset_from_directory(
        f"{folder_name}/test",
        label_mode="categorical",
        batch_size=batch_size,
        image_size=image_size,
        interpolation=interpolation,
        shuffle=shuffle,
    )

    class_names = train_ds.class_names

    # Avoid caching and prefetch more conservatively for lower RAM usage
    train_dataset = train_ds.prefetch(1)
    validation_dataset = val_ds.prefetch(1)
    test_dataset = test_ds.prefetch(1)

    return train_dataset, validation_dataset, test_dataset, class_names




def load_tabular_features(train_csv, val_csv, test_csv, column="phylum"):
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    phylum_columns = pd.get_dummies(df_train[column]).columns.tolist()

    train_tabular = pd.get_dummies(df_train[column])[phylum_columns].to_numpy()
    val_tabular = pd.get_dummies(df_val[column])[phylum_columns].to_numpy()
    test_tabular = pd.get_dummies(df_test[column])[phylum_columns].to_numpy()

    return train_tabular, val_tabular, test_tabular




def reorder_dataframe(df, folder_path):
    image_files = sorted(Path(folder_path).rglob("*/*.jpg"))

    file_paths = []
    for file in image_files:
        # Example:
        parts = file.parts[-2:]  # last 2 parts: folder/family + filename -  chordata_geoemydidae/20969394_7.jpg
        relative_path = os.path.join(parts[0], parts[1]).replace("\\", "/")  # For Windows compatibility
        file_paths.append(relative_path)

    # 3. Create a new DataFrame to align
    file_paths_df = pd.DataFrame({"file_path": file_paths})

    # 4. Merge to reorder
    df_reordered = file_paths_df.merge(df, on="file_path", how="left")

    return df_reordered



def merge_tabular_with_images(image_ds, tabular_data, batch_size, shuffle=False, shuffle_buffer=1000):
    tabular_ds = tf.data.Dataset.from_tensor_slices(tabular_data)

    merged = tf.data.Dataset.zip((image_ds.unbatch(), tabular_ds)).map(
        lambda img_lbl, tab: ({"image_input": img_lbl[0], "tabular_input": tab}, img_lbl[1]),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:
        merged = merged.shuffle(buffer_size=shuffle_buffer, seed=42, reshuffle_each_iteration=True)

    return merged.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

def merge_tabular_with_images_main(image_ds, tabular_data, batch_size, shuffle=False, shuffle_buffer=1000):
    tabular_ds = tf.data.Dataset.from_tensor_slices(tabular_data)

    merged = tf.data.Dataset.zip((image_ds.unbatch(), tabular_ds)).map(
        lambda img_lbl, tab: ({"image_input": img_lbl[0], "tabular_input": tab}, img_lbl[1]),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:
        merged = merged.shuffle(buffer_size=shuffle_buffer, seed=42, reshuffle_each_iteration=True)

    return merged.batch(batch_size).prefetch(1)

# ---------------------------------------------------------
# Model architecture
# ---------------------------------------------------------

def build_model(
    input_shape=(224, 224, 3),
    use_tabular = False,
    tabular_input_shape=None,
    project_tabular=False,
    use_data_augmentation=None,
    base_model_name="EfficientNetV2M",
    freeze_base=True,
    spatial_reduction="avg",
    dropout_rate=0.3,
    use_batch_norm=False,
    n_dense_extra_layers=1,
    n_dense_units=256,
    activation="leaky_relu",
    kernel_regularizer_l2=1e-2,
    n_classes=202,
    output_activation="softmax"
):
    # image input
    image_input = keras.Input(shape=input_shape, name="image_input")

    # augmentation
    if use_data_augmentation:
        x_img = use_data_augmentation(image_input)
    else:
        x_img = image_input

    # transfer learning
    base_model_class = getattr(tf.keras.applications, base_model_name)

    if base_model_name in ["Xception", "ResNet50V2", "InceptionV3", "DenseNet121", "EfficientNetB0", "EfficientNetB2", "EfficientNetB3", "EfficientNetB1"]:
        preprocess_map = {
            "Xception": tf.keras.applications.xception.preprocess_input,
            "ResNet50V2": tf.keras.applications.resnet_v2.preprocess_input,
            "InceptionV3": tf.keras.applications.inception_v3.preprocess_input,
            "DenseNet121": tf.keras.applications.densenet.preprocess_input,
            "EfficientNetB0": tf.keras.applications.efficientnet.preprocess_input,
            "EfficientNetB1": tf.keras.applications.efficientnet.preprocess_input,
            "EfficientNetB2": tf.keras.applications.efficientnet.preprocess_input,
            "EfficientNetB3": tf.keras.applications.efficientnet.preprocess_input,
        }
        x_img = preprocess_map.get(base_model_name, lambda x: x)(x_img)
        base_model = base_model_class(include_top=False, weights="imagenet", input_shape=input_shape, pooling=None)
    else:
        base_model = base_model_class(include_top=False, weights="imagenet", input_shape=input_shape, pooling=None, include_preprocessing=True)

    if freeze_base:
        base_model.trainable = False

    x_img = base_model(x_img)

    # Add a global spatial average pooling layer
    if spatial_reduction == "avg":
        x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
    elif spatial_reduction == "max":
        x_img = tf.keras.layers.GlobalMaxPool2D()(x_img)
    else:
        raise ValueError("Invalid spatial reduction method.")

    # Tabular input
    if use_tabular:
        tabular_input = keras.Input(shape=tabular_input_shape, name="tabular_input")
        if project_tabular:
            x_tab = keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(kernel_regularizer_l2))(tabular_input)
            if activation == "relu":
                x_tab = keras.layers.Activation("relu")(x_tab)
            elif activation == "leaky_relu":
                x_tab = keras.layers.LeakyReLU(alpha=0.1)(x_tab)
        else:
            x_tab = tabular_input
        x = keras.layers.Concatenate()([x_img, x_tab])
    else:
        x = x_img

    # Dense layers
    if use_batch_norm:
        x = keras.layers.BatchNormalization()(x)

    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate)(x)
        dropout_rate *= 0.5

    for _ in range(n_dense_extra_layers):
        x = keras.layers.Dense(n_dense_units, kernel_regularizer=keras.regularizers.l2(kernel_regularizer_l2))(x)
        if activation == "relu":
            x = keras.layers.Activation("relu")(x)
        elif activation == "leaky_relu":
            x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        n_dense_units = int(n_dense_units * 0.5)
        dropout_rate *= 0.5

    outputs = keras.layers.Dense(n_classes, activation=output_activation)(x)

    if use_tabular:
        model = keras.Model(inputs=[image_input, tabular_input], outputs=outputs, name=f"{base_model_name}_dual_input")
    else:
        model = keras.Model(inputs=image_input, outputs=outputs, name=f"{base_model_name}_image_only")

    return model




# ---------------------------------------------------------
# Custom Callbacks
# ---------------------------------------------------------
import time
class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model.optimizer.iterations)
        lr_value = tf.keras.backend.get_value(lr)
        print(f"Epoch {epoch + 1}: learning rate = {lr_value:.4e}")




class InlineLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        duration = time.time() - self.epoch_start_time

        train_metrics = ' - '.join(
            [f"{k}: {v:.4f}" for k, v in logs.items() if "val_" not in k and "weighted" not in k]
        )
        val_metrics = ' - '.join(
            [f"{k.replace('val_', 'val_')}: {v:.4f}" for k, v in logs.items() if "val_" in k and "weighted" not in k]
        )

        lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)

        print(f"Epoch {epoch + 1} - {duration:.0f}s | {train_metrics} | {val_metrics} | LR: {lr:.4e}")



# ---------------------------------------------------------
# Model compile: optimizer, loss / metrics
# ---------------------------------------------------------

def compile_model(
    model,
    optimizer_name="RMSprop",
    learning_rate=0.001,
    weight_decay=1e-2,
    loss_name="CategoricalCrossentropy",
    label_smoothing=0.1,
    metrics=None,
):
    optimizer = getattr(keras.optimizers, optimizer_name)
    optimizer = optimizer(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    loss = getattr(keras.losses, loss_name)
    loss = loss(label_smoothing=label_smoothing)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    return model


# ---------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------

def train_model(
    model,
    train_dataset,
    validation_dataset,
    epochs=100,
    callbacks=None,
    verbose=1,
):
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose
    )
    return history


def evaluate_model(
    model,
    test_dataset,
    verbose=1
):
    results = model.evaluate(test_dataset, verbose=verbose, return_dict=True)
    return results


# ----------------------------------------------------------
# Results plot
# ----------------------------------------------------------


def store_results(train_results, validation_results, params):
    results = pd.read_csv("metrics.csv")

    best_epoch_idx = results['val_loss'].idxmin()
    best_epoch_value = results['epoch'].iloc[best_epoch_idx]

    best_row = results[results['epoch'] == best_epoch_value].iloc[0]

    train_results.append({
        "model": params,
        "epoch": best_epoch_value,
        "train_f1_score_macro": best_row['f1_score'],
        "train_f1_score_weighted": best_row['f1_score_weighted'],
        "train_loss": best_row['loss'],
    })

    validation_results.append({
        "model": params,
        "epoch": best_epoch_value,
        "val_f1_score_macro": best_row['val_f1_score'],
        "val_f1_score_weighted": best_row['val_f1_score_weighted'],
        "val_loss": best_row['val_loss'],
    })

    return train_results, validation_results




def plot_results(train_results, validation_results):

    models = [res['model'] for res in validation_results]
    val_scores = [res['val_f1_score_macro'] for res in validation_results]
    train_scores = [res['train_f1_score_macro'] for res in train_results]

    sorted_indices = sorted(range(len(val_scores)), key=lambda i: val_scores[i], reverse=True)
    models_sorted = [models[i] for i in sorted_indices]
    val_scores_sorted = [val_scores[i] for i in sorted_indices]
    train_scores_sorted = [train_scores[i] for i in sorted_indices]

    plt.figure(figsize=(10, 6))
    x = range(len(models_sorted))

    train_bars = plt.bar(x, train_scores_sorted, color='lightgray', label='Train F1 (Macro)')

    val_bars = plt.bar(x, val_scores_sorted, color='skyblue', label='Validation F1 (Macro)')

    for i, (train_bar, val_bar) in enumerate(zip(train_bars, val_bars)):
        plt.text(train_bar.get_x() + train_bar.get_width()/2, train_bar.get_height() + 0.01,
                 f"{train_scores_sorted[i]*100:.2f}%", ha='center', va='bottom', fontsize=9, color='dimgray')
        plt.text(val_bar.get_x() + val_bar.get_width()/2, val_bar.get_height() + 0.01,
                 f"{val_scores_sorted[i]*100:.2f}%", ha='center', va='bottom', fontsize=9, color='black')

    plt.xticks(ticks=x, labels=models_sorted, rotation=30, ha='right')
    plt.xlabel("Model")
    plt.ylabel("F1 Score (Macro)")
    plt.title("Train vs Validation F1 Score per Model")
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.9), frameon=False)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()




def plot_training_curves(history, title="Training and Validation Loss", metrics_to_plot=None):
    history_dict = history.history
    epochs = range(1, len(history_dict["loss"]) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, history_dict["loss"], label="Train Loss", linewidth=2)
    plt.plot(epochs, history_dict["val_loss"], label="Val Loss", linewidth=2)

    if metrics_to_plot:
        for metric in metrics_to_plot:
            train_metric = history_dict.get(metric)
            val_metric = history_dict.get(f"val_{metric}")
            if train_metric and val_metric:
                plt.plot(epochs, train_metric, label=f"Train {metric.capitalize()}", linestyle="--")
                plt.plot(epochs, val_metric, label=f"Val {metric.capitalize()}", linestyle="--")

    plt.xlabel("Epochs")
    plt.ylabel("Loss / Metric")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# ----------------------------------------------------------
# Fine Tuning
# ----------------------------------------------------------

def unfreeze_model_layers(base_model, block_name, freeze_batch=True):
    base_model.trainable = True

    for i, layer in enumerate(base_model.layers):
        if layer.name.startswith(block_name):
            unfreeze_from = i
            break
        else:
            layer.trainable = False

    for layer in base_model.layers[unfreeze_from:]:
        if freeze_batch and isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False

    if freeze_batch:
        print(f"Unfreezed from layer: {base_model.layers[unfreeze_from].name}, including BatchNorm layers")
    else:
        print(f"Unfreezed from layer: {base_model.layers[unfreeze_from].name}, including BatchNorm layers")
    return


def get_best_epoch():
    results = pd.read_csv("metrics.csv")

    best_epoch = results['epoch'].iloc[results['val_loss'].idxmin()]
    return best_epoch



# ----------------------------------------------------------
# Others
# ----------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
import tensorflow as tf
from PIL import Image



def plot_metrics(history):
    epochs = range(1, len(history.history['f1_score']) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))


    # this is to plot the f1
    axs[0].plot(epochs, history.history['f1_score'], color='#0ea7b5', label='Train Macro F1')
    axs[0].plot(epochs, history.history['val_f1_score'], color='#ffbe4f', label='Validation Macro F1')
    axs[0].plot(epochs, history.history['f1_score_weighted'], color='#0ea7b5', linestyle='--' ,label='Train Weighted F1')
    axs[0].plot(epochs, history.history['val_f1_score_weighted'], color='#ffbe4f',linestyle='--', label='Validation Weighted F1')

    axs[0].set_title('Training and Validation F1 Scores over Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('F1 Score')
    axs[0].legend(loc='upper left')

    # to plot the loss
    axs[1].plot(epochs, history.history['loss'], '#9ed670', label='Train Loss')
    axs[1].plot(epochs, history.history['val_loss'], '#d64d4d', label='Validation Loss')

    axs[1].set_title('Training and Validation Loss over Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend(loc='upper right')

    axs[0].grid(True)
    axs[1].grid(True)


    plt.tight_layout()
    plt.show()


def plot_results_adjusted(train_results, validation_results):

    models = [res['model'] for res in validation_results]
    val_scores = [res['val_f1_score_macro'] for res in validation_results]
    train_scores = [res['train_f1_score_macro'] for res in train_results]

    sorted_indices = sorted(range(len(val_scores)), key=lambda i: val_scores[i], reverse=True)
    models_sorted = [models[i] for i in sorted_indices]
    val_scores_sorted = [val_scores[i] for i in sorted_indices]
    train_scores_sorted = [train_scores[i] for i in sorted_indices]

    plt.figure(figsize=(10, 6))
    x = range(len(models_sorted))

    train_bars = plt.bar(x, train_scores_sorted, color='lightgray', label='Train F1 (Macro)')

    val_bars = plt.bar(x, val_scores_sorted, color='skyblue', label='Validation F1 (Macro)')

    for i, (train_bar, val_bar) in enumerate(zip(train_bars, val_bars)):
        plt.text(train_bar.get_x() + train_bar.get_width()/2, train_bar.get_height() + 0.01,
                 f"{train_scores_sorted[i]*100:.2f}%", ha='center', va='bottom', fontsize=9, color='dimgray')
        plt.text(val_bar.get_x() + val_bar.get_width()/2, val_bar.get_height() + 0.01,
                 f"{val_scores_sorted[i]*100:.2f}%", ha='center', va='bottom', fontsize=9, color='black')

    plt.ylim(0, max(val_scores_sorted + train_scores_sorted) + 0.05)  # Mais padding para o retangulo

    plt.xticks(ticks=x, labels=models_sorted, rotation=30, ha='right')
    plt.xlabel("Model")
    plt.ylabel("F1 Score (Macro)")
    plt.title("Train vs Validation F1 Score per Model")
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.9), frameon=False)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def load_image_np(image_path):
    image = Image.open(image_path).resize((224, 224))
    img_array = np.array(image)
    img_input = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
    return img_input


def show_augmentation(dictionary_image,data_augmentation,number_images):
    """
    dictionary_image = {"family":"path"}
    data_augmentation = tf.keras.Sequential([...])

    This will show for each of the families the number of images defines, with the subtitle on top.

    """
    for category, imagem in dictionary_image.items():
        img_input = load_image_np(imagem)
        augmented_images = []

        plt.figure(figsize=(8, 7))

        plt.suptitle(f"Category: {category}", fontsize=10, y=0.92)
        for i in range(number_images):
            aug_img = data_augmentation(img_input)

            aug_img = tf.squeeze(aug_img, axis=0)  #

            aug_img = aug_img / 255.0

            augmented_images.append(aug_img)

            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(aug_img)
            plt.axis("off")
        plt.tight_layout()
        plt.show()


def show_images_from_dict(image_dict):
    for category, image_path in image_dict.items():

        img_input = load_image_np(image_path)

        img_input = np.squeeze(img_input, axis=0)
        img_input = img_input / 255.0

        plt.figure(figsize=(4, 4))
        plt.imshow(img_input)
        plt.axis('off')
        plt.title(f"Category: {category}")
        plt.show()
