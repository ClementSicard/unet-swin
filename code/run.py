from utils import *
import argparse
import models.baselines.baseline_svm_classifier as svc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help="Model to use for training.",
        choices=["baseline-svc", "baseline-unet", "unet"],
    )

    args = parser.parse_args()

    if args.model == "baseline-svc":
        print("Running baseline SVC...")
        svc.run()

    else:
        print("Not yet implemented.")
