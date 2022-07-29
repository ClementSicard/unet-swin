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
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader

from ..unet import run as run_unet


sys.path.append("..")


def run(
    train_path: str,
    val_path: str,
    test_path: str,
    checkpoint_path=None,
    model_save_dir: str = None,
):
    run_unet(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        n_epochs=35,
        batch_size=4,
        augment=False,
        crop=False,
        checkpoint_path=checkpoint_path,
        model_save_dir=model_save_dir,
        loss="bce",
    )
