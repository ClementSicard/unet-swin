from PIL import Image
from glob import glob
import numpy as np
import matplotlib.pyplot as plt


def load_all_from_path(path: str):
    # loads all HxW .pngs contained in path as a 4D np.array of shape (n_images, H, W, 3)
    # images are loaded as floats with values in the interval [0., 1.]
    return np.stack([np.array(Image.open(f)) for f in sorted(glob(path + "/*.png"))]).astype(np.float32) / 255.0


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
