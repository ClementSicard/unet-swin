from PIL import Image
import torch
from .encoders.swin_small import swin_pretrained
from .decoders.custom_decoder import Decoder
import sys

sys.path.append("..")

from dataset import ImageDataset
from train import train
from utils import *
from datetime import datetime
import os
import numpy as np

INFERED_SIZES = [(768, 384), (384, 192), (192, 96), (96, 3)]


class SwinUnet(torch.nn.Module):
    def __init__(self):
        super(SwinUnet, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.encoder = swin_pretrained().to(device)
        self.decoder = Decoder(sizes=INFERED_SIZES).to(device)
        self.prev_conv = torch.nn.Conv2d(768, 768, kernel_size=3, padding=1, bias=False)
        self.fully_connected = torch.nn.Linear(16 * 16, 1)

    def forward(self, x):
        # askip on preprocess les images
        x = self.encoder(x)
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
        return x


def run(train_path: str, val_path: str, test_path: str, n_epochs=20, batch_size=128):
    log("Training Swin-Unet Baseline...")
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # automatically select device
    train_dataset = ImageDataset(train_path, device, use_patches=False, augment=False)
    val_dataset = ImageDataset(val_path, device, use_patches=False, augment=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )
    model = SwinUnet().to(device)
    # model.encoder.features.requires_grad_ = False
    # param.requires_grad = False
    # model.encoder.weight.requires_grad = False
    # exit()
    loss_fn = torch.nn.BCELoss()
    # loss_fn = BinaryDiceLoss()
    metric_fns = {"acc": accuracy_fn, "patch_acc": patch_accuracy_fn}
    best_metric_fns = {"patch_acc": patch_accuracy_fn}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    train(
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        model=model,
        loss_fn=loss_fn,
        metric_fns=metric_fns,
        best_metric_fns=best_metric_fns,
        optimizer=optimizer,
        n_epoches=n_epochs,
        model_name="swin-unet",
    )

    log("Training done!")

    log("Predicting on test set...")
    # predict on test set
    test_path = os.path.join(test_path, "images")
    test_filenames = sorted(glob(test_path + "/*.png"))
    test_images = load_all_from_path(test_path)
    test_images = test_images[:, :, :, :3]
    log(f"{test_images.shape[0]} were loaded")
    test_images = np.moveaxis(test_images, -1, 1)  # HWC to CHW

    log(test_images.shape)
    os.makedirs("preds/segmentations", exist_ok=True)
    with torch.no_grad():
        for i, test_image in enumerate(test_images):
            test_image = torch.from_numpy(test_image).unsqueeze(0).to(device)
            pred = model(test_image).cpu().numpy().squeeze(0).squeeze(0)

            # log(pred)
            # pred[np.where(pred > 0.5)] = 1
            # pred[np.where(pred <= 0.5)] = 0
            # pred = pred.round()
            tmp = Image.fromarray((pred * 250).astype(np.uint8))
            tmp.save(f"preds/segmentations/pred_{i}.png")

    from mask_to_submission import masks_to_submission

    masks_to_submission(
        "preds/_preds.csv",
        "",
        *map(lambda x: f"preds/segmentations/{x}", os.listdir("preds/segmentations")),
    )
    log(f"Created submission!")
