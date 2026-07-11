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

## Dataset

The image dataset is not included in this repository because of size and distribution constraints.

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
