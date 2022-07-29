from datetime import datetime
from train import train
from dataset import ImageDataset, OptimizedImageDataset
from utils import *
from ..losses.dice_loss import BinaryDiceLoss
from ..losses.focal_loss import FocalLoss
from torch import nn
import torch
import sys
import numpy as np
import cv2
from torchmetrics import F1Score

sys.path.append("..")


class Block(nn.Module):
    # a repeating structure composed of two convolutional layers with batch normalization and ReLU activations
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(
                in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    # UNet-like architecture for single class semantic segmentation.
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        enc_chs = chs  # number of channels in the encoder
        dec_chs = chs[::-1][:-1]  # number of channels in the decoder
        self.enc_blocks = nn.ModuleList(
            [Block(in_ch, out_ch) for in_ch, out_ch in zip(enc_chs[:-1], enc_chs[1:])]
        )  # encoder blocks
        self.pool = nn.MaxPool2d(
            2
        )  # pooling layer (can be reused as it will not be trained)
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
                for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])
            ]
        )  # deconvolution
        self.dec_blocks = nn.ModuleList(
            [Block(in_ch, out_ch) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])]
        )  # decoder blocks
        self.head = nn.Sequential(
            nn.Conv2d(dec_chs[-1], 1, 1), nn.Sigmoid()
        )  # 1x1 convolution for producing the output

    def forward(self, x):
        # encode
        enc_features = []
        for block in self.enc_blocks[:-1]:
            x = block(x)  # pass through the block
            enc_features.append(x)  # save features for skip connections
            x = self.pool(x)  # decrease resolution
        x = self.enc_blocks[-1](x)
        # decode
        for block, upconv, feature in zip(
            self.dec_blocks, self.upconvs, enc_features[::-1]
        ):
            x = upconv(x)  # increase resolution
            x = torch.cat([x, feature], dim=1)  # concatenate skip features
            x = block(x)  # pass through the block
        return self.head(x)  # reduce to 1 channel


def run(
    train_path: str,
    val_path: str,
    test_path: str,
    n_epochs=35,
    batch_size=4,
    checkpoint_path=None,
    augment: bool = False,
    model_save_dir: str = None,
    loss: str = "bce",
):
    log("Training Vanilla-UNet Baseline...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # reshape the image to simplify the handling of skip connections and maxpooling
    train_dataset = OptimizedImageDataset(
        path=train_path,
        device=device,
        resize_to=(384, 384),
        type_="training",
        augment=augment,
    )
    val_dataset = OptimizedImageDataset(
        path=val_path,
        device=device,
        resize_to=(384, 384),
        type_="validation",
        augment=augment,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    model = UNet().to(device)
    if loss == "bce":
        loss_fn = nn.BCELoss()
    elif loss == "dice":
        loss_fn = BinaryDiceLoss()
    elif loss == "mixed":
        loss_fn = lambda y_hat, y: 0.4 * torch.nn.BCELoss()(
            y_hat, y
        ) + 0.6 * BinaryDiceLoss()(y_hat, y)
    elif loss == "focal":
        loss_fn = FocalLoss()
    best_metric_fn = {"patch_f1": patch_f1_score_fn}
    metric_fns = {
        "acc": accuracy_fn,
        "patch_acc": patch_accuracy_fn,
        "patch_f1": patch_f1_score_fn,
    }
    optimizer = torch.optim.Adam(model.parameters())

    best_weights_path = train(
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        model=model,
        loss_fn=loss_fn,
        metric_fns=metric_fns,
        best_metric_fn=best_metric_fn,
        checkpoint_path=checkpoint_path,
        model_save_path=model_save_dir,
        save_state=True,
        optimizer=optimizer,
        n_epochs=n_epochs,
        model_name="baseline_vanilla_unet",
    )

    log("Training done!")

    log("Predicting on test set...")
    # predict on test set
    test_path = os.path.join(test_path, "images")
    test_filenames = glob(test_path + "/*.png")
    test_images = load_all_from_path(test_path)
    batch_size = test_images.shape[0]
    size = test_images.shape[1:3]
    # we also need to resize the test images. This might not be the best ideas depending on their spatial resolution.
    log("Resizing test images...")
    test_images = np.stack(
        [cv2.resize(img, dsize=(384, 384)) for img in test_images], 0
    )
    test_images = test_images[:, :, :, :3]
    test_images = np_to_tensor(np.moveaxis(test_images, -1, 1), device)
    log("Making predictions...")
    # Load best model state
    checkpoint = torch.load(best_weights_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    log(f"Loaded best model weights ({best_weights_path})")

    test_pred = [
        model(t).detach().cpu().numpy() for t in tqdm(test_images.unsqueeze(1))
    ]
    test_pred = np.concatenate(test_pred, 0)
    test_pred = np.moveaxis(test_pred, 1, -1)  # CHW to HWC
    test_pred = np.stack(
        [cv2.resize(img, dsize=size) for img in test_pred], 0
    )  # resize to original shape
    # Now compute labels
    test_pred = test_pred.reshape(
        (-1, size[0] // PATCH_SIZE, PATCH_SIZE, size[0] // PATCH_SIZE, PATCH_SIZE)
    )
    test_pred = np.moveaxis(test_pred, 2, 3)
    test_pred = np.round(np.mean(test_pred, (-1, -2)) > CUTOFF)
    log(f"Test predictions shape: {test_pred.shape}")
    now = datetime.now()
    t = now.strftime("%Y-%m-%d_%H:%M:%S")
    os.makedirs("submissions", exist_ok=True)
    create_submission(
        test_pred,
        test_filenames,
        submission_filename=f"./submissions/baseline_unet_submission_{t}.csv",
    )
    log(f"Created submission!")
