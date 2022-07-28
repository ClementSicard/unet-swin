from utils import *
from torch import nn
import sys
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
        augment=False,
        batch_size=4,
        loss="bce",
        checkpoint_path=checkpoint_path,
        model_save_dir=model_save_dir,
        n_epochs=35,
        val_path=val_path,
        test_path=test_path,
    )
