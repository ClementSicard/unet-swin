from datetime import datetime
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from utils import *


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
    optimizer,
    n_epochs,
    model_name,
    interactive=False,
):
    # training loop
    logdir = "./tensorboard/net"
    writer = SummaryWriter(logdir)  # tensorboard writer (can also log images)

    history = {}  # collects metrics at the end of each epoch

    best_acc = 0.0

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        # initialize metric list
        metrics = {"loss": [], "val_loss": []}
        for k, _ in metric_fns.items():
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
            metrics_dict = {
                k: sum(v) / len(v) for k, v in metrics.items() if len(v) > 0
            }
            metrics_dict["max_sample_acc"] = max(metrics["acc"])
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

        # summarize metrics, log to tensorboard and display
        history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}

        for k, v in history[epoch].items():
            writer.add_scalar(k, v, epoch)
        print(
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
        # TODO: Which between acc and val_acc ?
        epoch_acc = history[epoch]["acc"]

        if epoch_acc > best_acc:
            t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"\t[{t}] New best batch accuracy: {epoch_acc:.4f}\tPrevious best batch accuracy: {best_acc:.4f}"
            )
            best_acc = epoch_acc
            os.makedirs(f"./models/{model_name}/states", exist_ok=True)
            torch.save(
                model,
                f"./models/{model_name}/{t}_best_acc_{best_acc:4f}_epoch_{epoch}.pt",
            )
            torch.save(
                model.state_dict(),
                f"./models/{model_name}/states/state_{t}_best_acc_{best_acc:4f}_epoch_{epoch}.pt",
            )

    print("Finished Training")
    # plot loss curves
    plt.plot([v["loss"] for k, v in history.items()], label="Training Loss")
    plt.plot([v["val_loss"] for k, v in history.items()], label="Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    now = datetime.now()
    t = now.strftime("%Y-%m-%d_%H:%M:%S")
    plt.savefig(f"./code/models/baselines/plots/loss_{model_name}_{t}.png")
    if interactive:
        plt.show()
