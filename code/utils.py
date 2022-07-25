from PIL import Image
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from consts import *
import re
import os
import torch
from tqdm.notebook import tqdm


def load_all_from_path(path: str):
    # loads all HxW .pngs contained in path as a 4D np.array of shape (n_images, H, W, 3)
    # images are loaded as floats with values in the interval [0., 1.]
    return (
        np.stack(
            [
                np.array(Image.open(f))
                for f in tqdm(sorted(glob(path + "/*.png")), desc="Loading images")
            ]
        ).astype(np.float32)
        / 255.0
    )


def show_first_n(imgs, masks, n=5):
    # visualizes the first n elements of a series of images and segmentation masks
    imgs_to_draw = min(n, len(imgs))
    fig, axs = plt.subplots(2, imgs_to_draw, figsize=(18.5, 6))
    for i in range(imgs_to_draw):
        axs[0, i].imshow(imgs[i])
        axs[1, i].imshow(masks[i])
        axs[0, i].set_title(f"Image {i}")
        axs[1, i].set_title(f"Mask {i}")
        axs[0, i].set_axis_off()
        axs[1, i].set_axis_off()
    plt.show()


def image_to_patches(images, masks=None):
    # takes in a 4D np.array containing images and (optionally) a 4D np.array containing the segmentation masks
    # returns a 4D np.array with an ordered sequence of patches extracted from the image and (optionally) a np.array containing labels
    n_images = images.shape[0]  # number of images
    h, w = images.shape[1:3]  # shape of images
    assert (h % PATCH_SIZE) + (
        w % PATCH_SIZE
    ) == 0  # make sure images can be patched exactly

    images = images[:, :, :, :3]

    h_patches = h // PATCH_SIZE
    w_patches = w // PATCH_SIZE

    patches = images.reshape(
        (n_images, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE, -1)
    )
    patches = np.moveaxis(patches, 2, 3)
    patches = patches.reshape(-1, PATCH_SIZE, PATCH_SIZE, 3)
    if masks is None:
        return patches

    masks = masks.reshape((n_images, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE, -1))
    masks = np.moveaxis(masks, 2, 3)
    labels = np.mean(masks, (-1, -2, -3)) > CUTOFF  # compute labels
    labels = labels.reshape(-1).astype(np.float32)
    return patches, labels


def show_patched_image(patches, labels, h_patches=25, w_patches=25):
    # reorders a set of patches in their original 2D shape and visualizes them
    fig, axs = plt.subplots(h_patches, w_patches, figsize=(18.5, 18.5))
    for i, (p, l) in enumerate(zip(patches, labels)):
        # the np.maximum operation paints patches labeled as road red
        axs[i // w_patches, i % w_patches].imshow(
            np.maximum(p, np.array([l.item(), 0.0, 0.0]))
        )
        axs[i // w_patches, i % w_patches].set_axis_off()
    plt.show()


def create_submission(labels, test_filenames, submission_filename):
    with open(submission_filename, "w") as f:
        f.write("id,prediction\n")
        for fn, patch_array in zip(sorted(test_filenames), labels):
            img_number = int(re.search(r"\d+", fn).group(0))
            for i in range(patch_array.shape[0]):
                for j in range(patch_array.shape[1]):
                    f.write(
                        "{:03d}_{}_{},{}\n".format(
                            img_number,
                            j * PATCH_SIZE,
                            i * PATCH_SIZE,
                            int(patch_array[i, j]),
                        )
                    )


def extract_features(x):
    return np.concatenate([np.mean(x, (-2, -3)), np.var(x, (-2, -3))], axis=1)


def load_data(train_path, val_path):
    train_images = load_all_from_path(os.path.join(train_path, "images"))
    train_masks = load_all_from_path(os.path.join(train_path, "groundtruth"))
    val_images = load_all_from_path(os.path.join(val_path, "images"))
    val_masks = load_all_from_path(os.path.join(val_path, "groundtruth"))

    return train_images, train_masks, val_images, val_masks


def np_to_tensor(x, device):
    # allocates tensors from np.arrays
    if device == "cpu":
        return torch.from_numpy(x).cpu()
    elif device == "mps":
        return torch.from_numpy(x).contiguous().to(torch.device("mps"))
    else:
        return (
            torch.from_numpy(x)
            .contiguous()
            .pin_memory()
            .to(device=device, non_blocking=True)
        )


def get_best_available_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
