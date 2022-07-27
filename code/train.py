from datetime import datetime
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from utils import *
from subprocess import Popen

pjoin = os.path.join


def show_val_samples(x, y, y_hat, segmentation=False):
    # training callback to show predictions on validation set
    imgs_to_draw = min(5, len(x))
    if x.shape[-2:] == y.shape[-2:]:  # segmentation
        fig, axs = plt.subplots(3, imgs_to_draw, figsize=(18.5, 12))
        for i in range(imgs_to_draw):
            axs[0, i].imshow(np.moveaxis(x[i], 0, -1))
            axs[1, i].imshow(np.concatenate([np.moveaxis(y_hat[i], 0, -1)] * 3, -1))
            axs[2, i].imshow(np.concatenate([np.moveaxis(y[i], 0, -1)] * 3, -1))
            axs[0, i].set_title(f"Sample {i}")
            axs[1, i].set_title(f"Predicted {i}")
            axs[2, i].set_title(f"True {i}")
            axs[0, i].set_axis_off()
            axs[1, i].set_axis_off()
            axs[2, i].set_axis_off()
    else:  # classification
        fig, axs = plt.subplots(1, imgs_to_draw, figsize=(18.5, 6))
        for i in range(imgs_to_draw):
            axs[i].imshow(np.moveaxis(x[i], 0, -1))
            axs[i].set_title(
                f"True: {np.round(y[i]).item()}; Predicted: {np.round(y_hat[i]).item()}"
            )
            axs[i].set_axis_off()
    plt.show()


def train(
    train_dataloader,
    eval_dataloader,
    model,
    loss_fn,
    metric_fns,
    best_metric_fn,
    optimizer,
    n_epochs,
    model_name,
    save_state=True,
    checkpoint_path=None,
    interactive=False,
):
    # training loop
    logdir = "./tensorboard/net"
    writer = SummaryWriter(logdir)  # tensorboard writer (can also log images)

    history = {}  # collects metrics at the end of each epoch

    best_metric_fn_val = 0.0

    checkpoint_epoch = 0

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_metric_fn = checkpoint["best_metric_fn"]
        best_metric_fn_val = checkpoint["best_metric_fn_val"]
        checkpoint_epoch = checkpoint["epoch"]
        append_to_log(
            f"""
            Model loaded:
            - epoch: {checkpoint_epoch}
            - Best metric function: {list(best_metric_fn.keys())[0]}
            - Current best metric function value: {best_metric_fn_val:.4f}
        """
        )

    for epoch in range(
        checkpoint_epoch, n_epochs
    ):  # loop over the dataset multiple times

        # Add real-time logs

        # initialize metric list
        metrics = {"loss": [], "val_loss": []}
        for k, _ in metric_fns.items():
            metrics[k] = []
            metrics["val_" + k] = []
        for k, _ in best_metric_fn.items():
            metrics[k] = []
            metrics["val_" + k] = []

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        # training
        model.train()
        for (x, y) in pbar:
            optimizer.zero_grad()  # zero out gradients
            y_hat = model(x)  # forward pass
            loss = loss_fn(y_hat, y)
            loss.backward()  # backward pass
            optimizer.step()  # optimize weights

            # log partial metrics
            metrics["loss"].append(loss.item())
            for k, fn in metric_fns.items():
                metrics[k].append(fn(y_hat, y).item())
            for k, fn in best_metric_fn.items():
                metrics[k].append(fn(y_hat, y).item())
            metrics_dict = {
                k: sum(v) / len(v) for k, v in metrics.items() if len(v) > 0
            }
            metrics_dict[f"max_sample_{list(best_metric_fn.keys())[0]}"] = max(
                metrics[list(best_metric_fn.keys())[0]]
            )
            pbar.set_postfix(metrics_dict)

        # validation
        model.eval()
        with torch.no_grad():  # do not keep track of gradients
            for (x, y) in eval_dataloader:
                y_hat = model(x)  # forward pass
                loss = loss_fn(y_hat, y)

                # log partial metrics
                metrics["val_loss"].append(loss.item())
                for k, fn in metric_fns.items():
                    metrics["val_" + k].append(fn(y_hat, y).item())
                for k, fn in best_metric_fn.items():
                    metrics["val_" + k].append(fn(y_hat, y).item())

        # summarize metrics, log to tensorboard and display
        history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}

        for k, v in history[epoch].items():
            writer.add_scalar(k, v, epoch)
        append_to_log(
            " ".join(
                [
                    "\t- " + str(k) + " = " + str(v) + "\n "
                    for (k, v) in history[epoch].items()
                ]
            )
        )
        if interactive:
            show_val_samples(
                x.detach().cpu().numpy(),
                y.detach().cpu().numpy(),
                y_hat.detach().cpu().numpy(),
            )

        best_metric_key = f"val_{list(best_metric_fn.keys())[0]}"
        epoch_best_metric_fn_val = history[epoch][best_metric_key]

        # If a better value for the best metric is found, save the model
        if epoch_best_metric_fn_val > best_metric_fn_val:
            t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            append_to_log(
                f"\t[{t}] New best batch {best_metric_key}: {epoch_best_metric_fn_val:.4f}\tPrevious best batch {best_metric_key}: {best_metric_fn_val:.4f}"
            )
            best_metric_fn_val = epoch_best_metric_fn_val
            if save_state:
                os.makedirs(f"./models/{model_name}", exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_metric_fn_val": best_metric_fn_val,
                        "best_metric_fn": best_metric_fn,
                    },
                    pjoin(
                        "models",
                        model_name,
                        f"best_{best_metric_key}_{best_metric_fn_val:4f}_epoch_{epoch}.pt",
                    ),
                )

    append_to_log("Finished Training")
    # plot loss curves
    plt.plot([v["loss"] for k, v in history.items()], label="Training Loss")
    plt.plot([v["val_loss"] for k, v in history.items()], label="Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    now = datetime.now()
    t = now.strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"./code/models/baselines/plots/loss_{model_name}_{t}.png")
    if interactive:
        plt.show()
