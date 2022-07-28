import numpy as np
import os
from datetime import datetime
from utils import *
from train import train
from dataset import ImageDataset
from PIL import Image
import torch
from .losses.dice_loss import BinaryDiceLoss
from .encoders.swin import swin_pretrained_s, swin_pretrained_b
from .decoders.custom_decoder import Decoder
import sys
from .losses.dice_loss import BinaryDiceLoss
from .losses.focal_loss import FocalLoss

sys.path.append("..")


INFERED_SIZES = [(768, 384), (384, 192), (192, 96), (96, 48)]
INFERED_SIZES_B = [(1024, 512), (512, 256), (256, 128), (128, 64)]


class SwinUnet(torch.nn.Module):
    def __init__(self, model_type: str = "small"):
        assert model_type in {"small", "base"}
        super(SwinUnet, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_type == "small":
            self.encoder = swin_pretrained_s().to(device)
            self.decoder = Decoder(INFERED_SIZES).to(device)
            self.prev_conv = torch.nn.Conv2d(
                INFERED_SIZES[0][0], INFERED_SIZES[0][0], kernel_size=3, padding=1, bias=True)
        else:
            self.encoder = swin_pretrained_b().to(device)
            self.decoder = Decoder(sizes=INFERED_SIZES_B).to(device)
            self.prev_conv = torch.nn.Conv2d(
                INFERED_SIZES_B[0][0], INFERED_SIZES_B[0][0], kernel_size=3, padding=1, bias=True)
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(self.decoder.last_conv2.out_channels, 1, 1),
            torch.nn.Sigmoid(),
        )

        # self.fully_connected = torch.nn.Linear(16 * 16, 1)

    def forward(self, x):
        # askip on preprocess les images
        # x = torch.nn.MaxPool2d((2, 2))(x)
        # x = torch.nn.Conv2d(3, 3, kernel_size=7, padding=1, bias=True)(x)
        x = self.encoder(x)

        # print(f"Shape of x: {x.shape}")
        x = self.prev_conv(x)
        self.encoder.x_int.reverse()
        # for int in self.encoder.x_int[1::]:
        #     log(int.shape)
        x = self.decoder(x, self.encoder.x_int[1::])
        # x = self.fully_connected(x.view(x.shape[0], -1))
        # x = torch.sigmoid(x)
        # x[x > 0.5] = 1
        # x[x <= 0.5] = 0
        # log(x.shape, flush=True)
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
    loss: str = "bce",
):
    assert loss in {"bce", "dice", "mix"}
    log("Training Swin-Unet Baseline...")
    device = "cuda" if torch.cuda.is_available(
    ) else "cpu"  # automatically select device
    train_dataset = ImageDataset(
        train_path, device, use_patches=False, augment=True, crop=True, resize_to=(200, 200))
    val_dataset = ImageDataset(
        val_path, device, use_patches=False, augment=True, crop=True, resize_to=(200, 200))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True)
    model = SwinUnet(model_type=model_type).to(device)

    # model.encoder.features.requires_grad_ = False
    # param.requires_grad = False
    # model.encoder.weight.requires_grad = False
    # exit()
    metric_fns = {"acc": accuracy_fn, "patch_acc": patch_accuracy_fn}
    best_metric_fns = {"patch_acc": patch_accuracy_fn}
    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)

    if loss == "bce":
        loss_fn = torch.nn.BCELoss()
    elif loss == "dice":
        loss_fn = BinaryDiceLoss()
    elif loss == "mix":
        def loss_fn(y_hat, y): return 0.4 * torch.nn.BCELoss()(y_hat,
                                                               y) + 0.6 * BinaryDiceLoss()(y_hat, y)
    elif loss == "focal":
        loss_fn = FocalLoss()
    else:
        raise NotImplementedError(f"Loss {loss} is not implemented")

    metric_fns = {"acc": accuracy_fn, "patch_acc": patch_accuracy_fn}
    best_metric_fns = {"patch_acc": patch_accuracy_fn}
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_weights_path = train(
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        model=model,
        loss_fn=loss_fn,
        metric_fns=metric_fns,
        best_metric_fn=best_metric_fns,
        optimizer=optimizer_ft,
        scheduler=exp_lr_scheduler,
        n_epochs=n_epochs,
        checkpoint_path=checkpoint_path,
        save_state=True,
        model_save_path=model_save_dir,
        model_name="swin-unet",
    )

    log("Training done!")

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

    os.makedirs("preds/segmentations", exist_ok=True)
    with torch.no_grad():
        checkpoint = torch.load(best_weights_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        log(f"Loaded best model weights ({best_weights_path})")
        test_pred = [model(t).detach().cpu().numpy()
                     for t in tqdm(test_images.unsqueeze(1))]

        test_pred = np.concatenate(test_pred, 0)
        test_pred = np.moveaxis(test_pred, 1, -1)  # CHW to HWC
        test_pred = np.stack([img for img in test_pred],
                             0)  # resize to original shape
        # Now compute labels
        test_pred = test_pred.reshape(
            (-1, size[0] // PATCH_SIZE, PATCH_SIZE, size[0] // PATCH_SIZE, PATCH_SIZE))
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
