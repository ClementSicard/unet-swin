import torchvision.transforms.functional as TF
from torchvision import transforms
from consts import *
import torch
import os
import cv2
from utils import *


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        device,
        use_patches=True,
        augment=True,
        resize_to=(400, 400),
        verbose=False,
    ):
        self.path = path
        self.device = device
        self.use_patches = use_patches
        self.resize_to = resize_to
        self.x, self.y, self.n_samples = None, None, None
        self.augment = augment
        self.verbose = verbose
        self.N_TRANSFORMS = 6
        self._load_data()

    def __repr__(self) -> str:
        return super().__repr__()

    def _load_data(self):  # not very scalable, but good enough for now
        self.x = load_all_from_path(os.path.join(self.path, "images"))[:, :, :, :3]
        self.y = load_all_from_path(os.path.join(self.path, "groundtruth"))

        if self.use_patches:  # split each image into patches
            self.x, self.y = image_to_patches(self.x, self.y)

        elif self.resize_to != (self.x.shape[1], self.x.shape[2]):  # resize images
            self.x = np.stack(
                [cv2.resize(img, dsize=self.resize_to) for img in self.x], 0
            )
            self.y = np.stack(
                [cv2.resize(mask, dsize=self.resize_to) for mask in self.y], 0
            )

        self.x = np.moveaxis(
            self.x, -1, 1
        )  # pytorch works with CHW format instead of HWC
        self.n_samples = len(self.x)
        print(
            f"Using {'AUGMENTED' if self.augment else 'REGULAR'} dataset {'WITH' if self.use_patches else 'WITHOUT'} patches, with {len(self)} samples in total"
        )

    def transform(self, image, mask, index):
        """
        Creates a transform based on the modulo index of the image.

        0. No transform
        1. Horizontal flip
        2. Vertical flip
        3. 90 degree rotation
        4. -90 degree rotation
        5. Random crop
        """

        mod = index % self.N_TRANSFORMS
        if self.verbose:
            print(f"Index: {index}, mod: {mod}")
            desc = f"""
        Type of image: {type(image)}
        Type of mask: {type(mask)}
        Shape of image: {image.shape}
        Shape of mask: {mask.shape}
            """
            print(desc)

        if mod == 0:
            return image, mask

        # Horiztonal flip
        elif mod == 1:
            t_image = TF.hflip(image)
            t_mask = TF.hflip(mask) if not self.use_patches else mask

        # Vertical flip
        elif mod == 2:
            t_image = TF.vflip(image)
            t_mask = TF.vflip(mask) if not self.use_patches else mask

        elif mod == 3:
            t_image = TF.rotate(image, angle=-90)
            t_mask = TF.rotate(mask, angle=-90) if not self.use_patches else mask

        elif mod == 4:
            t_image = (TF.rotate(image, angle=90),)
            t_mask = TF.rotate(mask, angle=90) if not self.use_patches else mask

        elif mod == 5:
            resize_size = (
                (PATCH_SIZE, PATCH_SIZE) if self.use_patches else self.resize_to
            )

            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img=image,
                scale=(0.7, 0.9),
                ratio=(0.9, 1.1),
            )
            t_image = TF.resize(TF.crop(image, i, j, h, w), resize_size)
            t_mask = (
                TF.resize(TF.crop(mask, i, j, h, w), resize_size)
                if not self.use_patches
                else mask
            )

        if self.use_patches:
            return t_image, t_mask[0]
        else:
            return t_image, t_mask

    def __getitem__(self, index):
        if self.augment:
            image, mask = (
                self.x[index // self.N_TRANSFORMS],
                self.y[[index // self.N_TRANSFORMS]],
            )
        else:
            image, mask = self.x[index], self.y[[index]]

        image_tensor = np_to_tensor(image, self.device)
        mask_tensor = np_to_tensor(mask, self.device)

        return (
            self.transform(image_tensor, mask_tensor, index=index)
            if self.augment
            else (image_tensor, mask_tensor)
        )

    def __len__(self):
        return self.n_samples * self.N_TRANSFORMS if self.augment else self.n_samples
