from torch import nn
import torch
import sys
import numpy as np

sys.path.append("..")
from utils import *
from dataset import ImageDataset
from train import train


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


def accuracy_fn(y_hat, y):
    # computes classification accuracy
    return (y_hat.round() == y.round()).float().mean()


def run(train_path: str, val_path: str, test_path: str):
    print("Training Patch-CNN Baseline...")
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # automatically select device
    train_dataset = ImageDataset(train_path, device, augment=False)
    val_dataset = ImageDataset(val_path, device, augment=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=True
    )
    model = PatchCNN().to(device)
    loss_fn = nn.BCELoss()
    metric_fns = {"acc": accuracy_fn}
    optimizer = torch.optim.Adam(model.parameters())
    n_epochs = 20

    train(
        train_dataloader,
        val_dataloader,
        model,
        loss_fn,
        metric_fns,
        optimizer,
        n_epochs,
    )

    # predict on test set
    test_filenames = sorted(glob(test_path + "/*.png"))
    test_images = load_all_from_path(test_path)
    test_images = test_images[:, :, :, :3]
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
    create_submission(
        test_pred, test_filenames, submission_filename="cnn_submission.csv"
    )
