from torch import nn
import torch
import sys
import numpy as np

sys.path.append("..")
from ..losses.dice_loss import BinaryDiceLoss
from ..losses.focal_loss import FocalLoss
from ..losses.mixed_loss import MixedLoss
from utils import *
from dataset import ImageDataset
from train import train
from datetime import datetime


class PatchCNN(nn.Module):
    # simple CNN for classification of patches
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(256, 10),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(10, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def run(
    train_path: str,
    val_path: str,
    test_path: str,
    n_epochs: int = 20,
    batch_size: int = 128,
    checkpoint_path: str = None,
    model_save_dir: str = None,
    loss: str = "bce",
):
    log("Training Patch-CNN Baseline...")
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # automatically select device
    train_dataset = ImageDataset(train_path, device, augment=False, use_patches=True)
    val_dataset = ImageDataset(val_path, device, augment=False, use_patches=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )
    model = PatchCNN().to(device)

    if loss == "bce":
        loss_fn = nn.BCELoss()
    elif loss == "dice":
        loss_fn = BinaryDiceLoss()
    elif loss == "mixed":
        loss_fn = MixedLoss()
    elif loss == "focal":
        loss_fn = FocalLoss()

    metric_fns = {"acc": accuracy_fn}
    best_metric_fn = {"patch_f1_score": patch_f1_score_fn}
    optimizer = torch.optim.Adam(model.parameters())

    train(
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        model=model,
        loss_fn=loss_fn,
        metric_fns=metric_fns,
        best_metric_fn=best_metric_fn,
        optimizer=optimizer,
        n_epochs=n_epochs,
        checkpoint_path=checkpoint_path,
        model_save_path=model_save_dir,
        model_name="baseline_patch_cnn",
    )

    log("Training done!")

    log("Predicting on test set...")
    # predict on test set
    test_path = os.path.join(test_path, "images")
    test_filenames = sorted(glob(test_path + "/*.png"))
    test_images = load_all_from_path(test_path)
    test_images = test_images[:, :, :, :3]
    log(f"{test_images.shape[0]} were loaded")
    test_patches = np.moveaxis(image_to_patches(test_images), -1, 1)  # HWC to CHW
    test_patches = np.reshape(
        test_patches, (25, -1, 3, PATCH_SIZE, PATCH_SIZE)
    )  # split in batches for memory constraints
    test_pred = [
        model(np_to_tensor(batch, device)).detach().cpu().numpy()
        for batch in test_patches
    ]
    test_pred = np.concatenate(test_pred, 0)
    test_pred = np.round(
        test_pred.reshape(
            test_images.shape[0],
            test_images.shape[1] // PATCH_SIZE,
            test_images.shape[1] // PATCH_SIZE,
        )
    )
    log(f"Test predictions shape: {test_pred.shape}")
    now = datetime.now()
    t = now.strftime("%Y-%m-%d_%H:%M:%S")
    os.makedirs("submissions", exist_ok=True)
    create_submission(
        test_pred,
        test_filenames,
        submission_filename=f"./submissions/baseline_cnn_submission_{t}.csv",
    )
    log(f"Created submission!")
