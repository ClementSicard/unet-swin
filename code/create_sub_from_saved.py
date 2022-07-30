from utils import *
import argparse
import models.swin_unet as swin_unet
# TODO import models.unet as unet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help="Model to use for training.",
        # TODO add other choices
        choices=[
            "swin-unet"
        ]
    )
    parser.add_argument(
        "--model-type",
        type=str,
        help="Model to use for training Swin. Will be ignored otherwise",
        choices=["small", "base"],
        default="small",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        required=True,
        help="Path to the test directory",
    )
    parser.add_argument(
        "--model-weights-path",
        type=str,
        required=True,
        help="Path to the model weights"
    )

    args = parser.parse_args()
    log(vars(args))

    if args.model == "swin-unet":
        log("Testing Swin UNet")
        swin_unet.test_and_create_sub(
            test_dir=args.test_dir,
            model_path=args.model_weights_path,
            model_type=args.model_type,
        )
    else:
        raise NotImplementedError("Not implemented yet")
