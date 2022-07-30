import numpy as np
import os
import cv2
from datetime import datetime
from utils import *
from train import train
from dataset import OptimizedImageDataset
from PIL import Image
import torch

from .losses.dice_loss import BinaryDiceLoss
from .losses.mixed_loss import MixedLoss
from .losses.focal_loss import FocalLoss
from .losses.twersky_focal_loss import FocalTverskyLoss
from .losses.mixed_f1_loss import MixedF1Loss
from .losses.mixed_patch_f1_loss import MixedPatchF1Loss
from torch.utils.data import DataLoader

from .encoders.swin import swin_pretrained_s, swin_pretrained_b
from .decoders.custom_decoder import Decoder
import sys

sys.path.append("..")


INFERED_SIZES = [(768, 384), (384, 192), (192, 96), (96, 48)]
INFERED_SIZES_B = [(1024, 512), (512, 256), (256, 128), (128, 64)]


class SwinUNet(torch.nn.Module):
    def __init__(self, model_type: str = "small"):
        assert model_type in {"small", "base"}
        super(SwinUNet, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_type == "small":
            self.encoder = swin_pretrained_s().to(device)
            self.decoder = Decoder(INFERED_SIZES).to(device)
            self.prev_conv = torch.nn.Conv2d(
                INFERED_SIZES[0][0],
                INFERED_SIZES[0][0],
                kernel_size=3,
                padding=1,
                bias=True,
            )
            self.tail = torch.nn.Sequential(
                torch.nn.Conv2d(3, INFERED_SIZES[-1][-1], 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(INFERED_SIZES[-1][-1]),
                torch.nn.Conv2d(
                    INFERED_SIZES[-1][-1],
                    INFERED_SIZES[-1][-1],
                    kernel_size=3,
                    padding=1,
                ),
                torch.nn.ReLU(),
            )
        else:
            self.encoder = swin_pretrained_b().to(device)
            self.decoder = Decoder(sizes=INFERED_SIZES_B).to(device)
            self.prev_conv = torch.nn.Conv2d(
                INFERED_SIZES_B[0][0],
                INFERED_SIZES_B[0][0],
                kernel_size=3,
                padding=1,
                bias=True,
            )
            self.tail = torch.nn.Sequential(
                torch.nn.Conv2d(3, INFERED_SIZES_B[-1][-1], 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(INFERED_SIZES_B[-1][-1]),
                torch.nn.Conv2d(
                    INFERED_SIZES_B[-1][-1],
                    INFERED_SIZES_B[-1][-1],
                    kernel_size=3,
                    padding=1,
                ),
                torch.nn.ReLU(),
            )
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(self.decoder.last_convs[-2].out_channels, 1, 1),
            torch.nn.Sigmoid(),
        )

        # self.fully_connected = torch.nn.Linear(16 * 16, 1)

    def forward(self, x):
        x_tail = self.tail(x)
        x = self.encoder(x)

        x = self.prev_conv(x)
        self.encoder.x_int.reverse()
        x = self.decoder(x, self.encoder.x_int[1:-1:] + [x_tail])

        return self.head(x)


def run(
    train_path: str,
    val_path: str,
    test_path: str,
    n_epochs: int = 20,
    batch_size: int = 128,
    model_save_dir: str = None,
    checkpoint_path: str = None,
    model_type: str = "small",
    loss: str = "focal",
):
    assert loss in {"bce", "dice", "mixed", "focal", "twersky", "f1", "patch-f1"}
    log(f"Training Swin-{model_type.capitalize()}-UNet...")
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # automatically select device
    train_dataset = OptimizedImageDataset(
        train_path,
        device,
        augment=True,
        crop=True,
        crop_size=208,
        type_="training",
    )
    val_dataset = OptimizedImageDataset(
        val_path,
        device,
        augment=True,
        crop=True,
        crop_size=208,
        type_="validation",
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    model = SwinUNet(model_type=model_type).to(device)

    # model.encoder.features.requires_grad_ = False
    # param.requires_grad = False
    # model.encoder.weight.requires_grad = False
    # exit()
    metric_fns = {"acc": accuracy_fn, "patch_acc": patch_accuracy_fn}
    best_metric_fns = {"patch_acc": patch_accuracy_fn}
    # Observe that all parameters are being optimized
    # optimizer_ft = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = torch.optim.Adam(model.parameters())

    # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer_ft, step_size=7, gamma=0.1
    # )

    if loss == "bce":
        loss_fn = torch.nn.BCELoss()
    elif loss == "dice":
        loss_fn = BinaryDiceLoss()
    elif loss == "mixed":
        loss_fn = MixedLoss()
    elif loss == "focal":
        loss_fn = FocalLoss()
    elif loss == "twersky":
        loss_fn = FocalTverskyLoss()
    elif loss == "f1":
        loss_fn = MixedF1Loss(
            # Can be changed
            other_loss=FocalLoss(),
        )
    elif loss == "patch-f1":
        loss_fn = MixedPatchF1Loss(
            # Can be changed
            other_loss=FocalLoss(),
        )
    else:
        raise NotImplementedError(f"Loss {loss} is not implemented")

    metric_fns = {"acc": accuracy_fn, "patch_acc": patch_accuracy_fn}
    best_metric_fns = {"patch_f1_score": patch_f1_score_fn}

    best_weights_path = train(
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        model=model,
        loss_fn=loss_fn,
        metric_fns=metric_fns,
        best_metric_fn=best_metric_fns,
        optimizer=optimizer_ft,
        # scheduler=exp_lr_scheduler,
        n_epochs=n_epochs,
        checkpoint_path=checkpoint_path,
        save_state=True,
        model_save_path=model_save_dir,
        model_name="swin-unet",
    )

    log("Training done!")
    test_and_create_sub(test_path, best_weights_path, model_type)


def test_and_create_sub(
    test_path: str, model_path: str = None, model_type: str = "small"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log("Predicting on test set...")
    test_path = os.path.join(test_path, "images")
    test_filenames = glob(test_path + "/*.png")
    test_images = load_all_from_path(test_path)
    batch_size = test_images.shape[0]
    size = test_images.shape[1:3]
    # we also need to resize the test images. This might not be the best ideas depending on their spatial resolution.
    log("Resizing test images...")
    test_images = np.stack([img for img in test_images], 0)
    test_images = test_images[:, :, :, :3]
    test_images = np_to_tensor(np.moveaxis(test_images, -1, 1), device)
    log("Making predictions...")

    with torch.no_grad():
        model = SwinUNet(model_type=model_type).to(device)
        if model_path:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            log(f"Loaded best model weights ({model_path})")
        else:
            log("DEBUG: No best weights path using default weights")

        test_pred = []
        CROP_SIZE = 200
        RESIZE_SIZE = 208
        for image in tqdm(test_images):
            np_image = image.cpu().numpy()
            # move channels to last axis
            np_image = np.moveaxis(np_image, 0, -1)

            # splits the image into 4 equal patches
            cropped_image = [
                np_image[0:CROP_SIZE, 0:CROP_SIZE, :],
                np_image[CROP_SIZE : 2 * CROP_SIZE, 0:CROP_SIZE, :],
                np_image[0:CROP_SIZE, CROP_SIZE : 2 * CROP_SIZE, :],
                np_image[CROP_SIZE : 2 * CROP_SIZE, CROP_SIZE : 2 * CROP_SIZE, :],
            ]

            # resize the patches to the same size as the training images
            resized_image = [
                cv2.resize(c_img, dsize=(RESIZE_SIZE, RESIZE_SIZE))
                for c_img in cropped_image
            ]

            # create a tensor from the resized patches
            resized_crops = np.stack(resized_image, 0)

            # BHWC -> BCHW
            # TODO : ASK IF THIS IS CORRECT CLEMENT
            resized_crops = np_to_tensor(np.moveaxis(resized_image, -1, 1), device)

            # predict the segmentation
            # res has shape (4, H, W)
            res = model(resized_crops).detach().cpu().numpy().squeeze(axis=1)

            full_pred = np.zeros((400, 400))

            full_pred[0:CROP_SIZE, 0:CROP_SIZE] = cv2.resize(
                res[0], dsize=(CROP_SIZE, CROP_SIZE)
            )
            full_pred[CROP_SIZE : 2 * CROP_SIZE, 0:CROP_SIZE] = cv2.resize(
                res[1], dsize=(CROP_SIZE, CROP_SIZE)
            )
            full_pred[0:CROP_SIZE, CROP_SIZE : 2 * CROP_SIZE] = cv2.resize(
                res[2], dsize=(CROP_SIZE, CROP_SIZE)
            )
            full_pred[
                CROP_SIZE : 2 * CROP_SIZE, CROP_SIZE : 2 * CROP_SIZE
            ] = cv2.resize(res[3], dsize=(CROP_SIZE, CROP_SIZE))

            test_pred.append(full_pred)

        # test_pred = [model(t).detach().cpu().numpy()
        #              for t in tqdm(test_images.unsqueeze(1))]

        test_pred = np.concatenate(test_pred, 0)
        test_pred = np.moveaxis(test_pred, 1, -1)  # CHW to HWC
        test_pred = np.stack([img for img in test_pred], 0)  # resize to original shape
        # Now compute labels
        test_pred = test_pred.reshape(
            (-1, size[0] // PATCH_SIZE, PATCH_SIZE, size[0] // PATCH_SIZE, PATCH_SIZE)
        )
        test_pred = np.moveaxis(test_pred, 2, 3)
        test_pred = np.round(np.mean(test_pred, (-1, -2)) > CUTOFF)
        log(f"Test predictions shape: {test_pred.shape}")
        now = datetime.now()
        t = now.strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs("submissions", exist_ok=True)
        create_submission(
            test_pred,
            test_filenames,
            submission_filename=f"./submissions/swin_unet_submission_{t}.csv",
        )
        log(f"Created submission!")
