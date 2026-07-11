# Rare Species Family Classification

This repository contains a deep learning project for classifying rare species images into biological families. The work uses the TreeOfLife rare species dataset and compares progressively stronger image-classification pipelines before selecting a final EfficientNetV2-based transfer-learning model.

The final training script combines image features with one-hot encoded phylum metadata and predicts one of 202 family classes.

## Project Overview

- Task: multi-class image classification for rare species family prediction
- Dataset: TreeOfLife rare species image dataset, approximately 12,000 images
- Target: biological family, with 202 classes
- Main model: EfficientNetV2B1 transfer learning
- Extra input: phylum metadata encoded as tabular features
- Framework: TensorFlow/Keras
- Main evaluation metrics: macro F1 and weighted F1

## Repository Structure

```text
.
├── main.py                         # Final training and evaluation entry point
├── utils_.py                       # Data loading, model building, training helpers
├── organizing_folders.py           # Dataset split and metadata preparation script
├── requirements.txt                # Python dependencies
├── preinstall.txt                  # Minimal pre-install dependency list
├── notebooks/
│   ├── 1_Building_original_model.ipynb
│   ├── 2_Original_model_with_augmentation.ipynb
│   ├── 3_Transfer_learning_selection.ipynb
│   ├── 4_Final_model.ipynb
│   ├── 5_Variational_autoencoder.ipynb
│   └── best_model.h5
├── docs/
│   └── assets/                    # Figures extracted from the final report
└── report/
    └── GROUP_20.pdf
```

## Modeling Workflow

The notebooks document the main experiment path:

1. Build an initial convolutional neural network baseline.
2. Add image augmentation and regularization.
3. Compare transfer-learning backbones.
4. Train the selected final model.
5. Explore a variational autoencoder as an additional representation-learning experiment.

The production-style script in `main.py` reproduces the selected final setup using:

- `EfficientNetV2B1` with ImageNet weights
- frozen convolutional base
- global average pooling
- phylum metadata as a second model input
- RMSprop optimizer with cosine decay
- categorical cross-entropy loss
- macro and weighted F1 metrics
- early stopping and checkpointing on validation loss

### Final Model Architecture

The final model uses a frozen EfficientNetV2B1 image backbone and concatenates the pooled image representation with one-hot encoded phylum metadata before the classification head.

![Final model architecture](docs/assets/final-model-architecture.png)

The diagram is reproduced from the report. The implementation in `main.py` instantiates `EfficientNetV2B1`.

### Results Summary

The final frozen-base model reached:

- Train macro F1: 98.4%
- Validation macro F1: 74.4%
- Test macro F1: 74.7%
- Test weighted F1: 75.36%

![Train and validation loss curves for the final frozen-base model](docs/assets/final-model-loss-freeze-base.png)

The following table summarizes the main transfer-learning experiments from the report. Values are macro F1 percentages. `*` marks configurations with the best validation macro F1 score in their experiment group.

| Experiment group | Configuration | Train macro F1 (%) | Validation macro F1 (%) |
| --- | --- | ---: | ---: |
| Feature selection | Without tabular data | 70.94 | 65.40 |
| Feature selection | With tabular data, no dense layer after concat | 72.30 | 66.58 * |
| Feature selection | With tabular data, dense layer after concat | 70.97 | 66.28 * |
| Batch size | 32 | 71.82 | 66.01 |
| Batch size | 64 | 83.03 | 70.82 |
| Batch size | 128 | 88.60 | 72.16 * |
| Batch size | 224 | 89.94 | 71.69 * |
| Augmentations pre-tune | None | 85.52 | 68.51 * |
| Augmentations pre-tune | Random contrast | 84.52 | 67.38 |
| Augmentations pre-tune | Random flip H+V | 76.49 | 66.96 |
| Augmentations pre-tune | Random flip H | 84.70 | 68.43 * |
| Augmentations pre-tune | Random rotation | 72.37 | 66.19 |
| Augmentations pre-tune | Random zoom | 78.97 | 67.06 |
| Augmentations pre-tune | Random crop resize | 81.03 | 67.02 |
| Augmentations pre-tune | Random Gaussian blur | 85.88 | 68.40 * |
| Augmentations pre-tune | Random hue | 81.38 | 65.45 |
| Augmentations pre-tune | Random channel shift | 82.16 | 65.85 |
| Augmentation fine-tune | None | 97.37 | 73.38 * |
| Augmentation fine-tune | Random contrast | 97.58 | 73.44 * |
| Augmentation fine-tune | Random flip H | 97.26 | 73.31 * |
| Augmentation fine-tune | Random rotation | 87.05 | 71.89 |
| Augmentation fine-tune | Random zoom | 94.37 | 73.34 * |
| Augmentation fine-tune | Random Gaussian blur | 97.36 | 73.45 * |
| Combined augmentation fine-tune | None | 97.87 | 74.22 * |
| Combined augmentation fine-tune | Manual combined | 94.42 | 73.46 * |
| Combined augmentation fine-tune | RandAugment | 85.14 | 68.48 |
| Base model selection pre-tune | Xception | 89.69 | 56.52 |
| Base model selection pre-tune | ResNet50V2 | 91.34 | 55.73 |
| Base model selection pre-tune | InceptionV3 | 92.35 | 51.79 |
| Base model selection pre-tune | DenseNet121 | 74.38 | 60.06 |
| Base model selection pre-tune | EfficientNetB0 | 92.09 | 71.96 * |
| Base model selection pre-tune | EfficientNetV2B1 | 87.69 | 71.09 * |
| Base model selection pre-tune | EfficientNetV2S | 82.38 | 68.10 |
| Base model selection pre-tune | ConvNeXtTiny | 81.23 | 65.50 |
| Base model selection fine-tune | EfficientNetB0 | 94.87 | 74.05 |
| Base model selection fine-tune | EfficientNetV2B1 | 95.75 | 75.19 * |

## Dataset

The image dataset is not included in this repository because of size and distribution constraints.

The full project dataset contains 11,983 images across 202 animal families. The distribution is highly imbalanced: most families have fewer than 50 images, while only a small number of families have more than 200 images.

![Animal family image-count distribution](docs/assets/family-distribution.png)

Download the dataset from:

```text
https://drive.google.com/file/d/1PyxqW_nsORX4PetkQo6OIL0mUL1pFsTD/view
```

After downloading, place the extracted dataset folder in the project root with the name expected by `organizing_folders.py`:

```text
rare_species 1/
├── metadata.csv
├── class_folder_1/
├── class_folder_2/
└── ...
```

Then prepare the train, validation, and test folders:

```bash
python organizing_folders.py
```

This creates the following structure:

```text
data/
├── train/
├── val/
├── test/
├── metadata_train.csv
├── metadata_val.csv
└── metadata_test.csv
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r preinstall.txt
pip install -r requirements.txt
```

The project was built with TensorFlow 2.15 and Keras 2.15.

## Run Training

After preparing the `data/` directory, run the final model:

```bash
python main.py
```

Optional arguments:

```bash
python main.py --epochs 80 --batch_size 128 --image_size 224,224
```

Arguments:

- `--epochs`: number of training epochs, default `80`
- `--batch_size`: batch size, default `128`
- `--image_size`: input image size as `HEIGHT,WIDTH`, default `224,224`

Training writes runtime artifacts such as `checkpoint.keras` and `metrics.csv`.

## Saved Artifacts

The repository includes:

- `notebooks/best_model.h5`: saved model artifact from the notebook workflow
- `report/GROUP_20.pdf`: project report with experiment details

The raw image dataset and generated `data/` directory are intentionally excluded from version control.

## Notes

This project was developed as part of a 2024/2025 deep learning course project. It is intended as a reproducible research/code portfolio project rather than a packaged Python library.
