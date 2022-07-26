import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import sys

sys.path.append("..")
from utils import *


def run():
    train_path = "../data/training"
    val_path = "../data/validation"

    # Load data
    print("1. Loading data...")
    train_images, train_masks, val_images, val_masks = load_data(train_path, val_path)

    print(f"\tTraining images: {train_images.shape[0]}")
    print(f"\tTraining masks: {train_masks.shape[0]}")
    print(f"\tValidation images: {val_images.shape[0]}")
    print(f"\tValidation masks: {val_masks.shape[0]}")

    print("2. Creating patches...")
    train_patches, train_labels = image_to_patches(train_images, train_masks)
    val_patches, val_labels = image_to_patches(val_images, val_masks)

    # Extract features
    print("3. Extracting features...")
    x_train = extract_features(train_patches)
    x_val = extract_features(val_patches)

    print("4. Training model...")
    # Train model
    model = LinearSVC().fit(x_train, train_labels)
    print(f"\tTraining accuracy: {model.score(x_train, train_labels)}")

    print("5. Predicting...")
    val_preds = model.predict(x_val)

    print(val_labels)
    print(val_preds)

    print(f"\tValidation F1-Score: {f1_score(val_labels, val_preds)}")


if __name__ == "__main__":
    run()
