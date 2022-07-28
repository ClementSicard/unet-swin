from utils import *
import argparse
import models.baselines.baseline_svm_classifier as svc
import models.baselines.baseline_patch_cnn as patch_cnn
import models.baselines.baseline_vanilla_unet as vanilla_unet
from torchvision import __version__

log(f"Running torchvision {__version__}")

try:
    import models.swin_unet as swin_unet
except Exception as e:
    log(e)
    log(f"Could not import swin_unet. Running on torchvision {__version__}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help="Model to use for training.",
        choices=["baseline-svc", "baseline-unet", "baseline-patch-cnn", "swin-unet"],
    )
    parser.add_argument(
        "--model-type",
        type=str,
        help="Model to use for training Swin. Will be ignored otherwise",
        choices=["small", "base"],
        default="small",
    )
    parser.add_argument(
        "--loss",
        type=str,
        help="Loss to train with",
        choices=["bce", "dice", "mix"],
        default="bce",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        required=True,
        help="Path to the training directory",
    )
    parser.add_argument(
        "--model-save-dir",
        type=str,
        help="Path where the model will be saved",
    )
    parser.add_argument(
        "--no-augment",
        action="store_false",
        help="The dataset will not be augmented",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        required=True,
        help="Path to the validation directory",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        required=True,
        help="Path to the test directory",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=20,
        help="Number of epochs to train on",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Number of epochs to train on",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path of a model checkpoint to load",
    )

    args = parser.parse_args()
    log(vars(args))

    device = get_best_available_device()
    log(f"PyTorch will use device: {device}")
    if device == "cuda":
        log(f"GPU used: {torch.cuda.get_device_name(0)}")

    if args.model == "baseline-svc":
        log("Running baseline SVC...")
        svc.run(
            train_path=args.train_dir,
            val_path=args.val_dir,
            test_path=args.test_dir,
        )

    elif args.model == "baseline-patch-cnn":
        log("Running baseline Patch-CNN...")
        patch_cnn.run(
            train_path=args.train_dir,
            val_path=args.val_dir,
            test_path=args.test_dir,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            checkpoint_path=args.checkpoint_path,
        )

    elif args.model == "baseline-unet":
        log("Running baseline Vanilla-UNet...")
        vanilla_unet.run(
            train_path=args.train_dir,
            val_path=args.val_dir,
            test_path=args.test_dir,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            checkpoint_path=args.checkpoint_path,
            augment=args.no_augment,
            model_save_dir=args.model_save_dir,
            loss=args.loss,
        )
    elif args.model == "swin-unet":
        log("Running SWIN-UNet...")
        swin_unet.run(
            train_path=args.train_dir,
            val_path=args.val_dir,
            test_path=args.test_dir,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            checkpoint_path=args.checkpoint_path,
            model_type=args.model_type,
            loss=args.loss,
            # augment=args.no_augment,
            model_save_dir=args.model_save_dir,
        )
    else:
        raise NotImplementedError("Not implemented yet")
