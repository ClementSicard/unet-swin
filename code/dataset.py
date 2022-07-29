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
        crop=False,
        verbose=False,
    ):
        self.path = path
        self.device = device
        self.use_patches = use_patches
        self.resize_to = resize_to
        self.x, self.y, self.n_samples = None, None, None
        self.augment = augment
        self.crop = crop
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

        if self.crop:
            self.x, self.y = crop_to_size(self.x, self.y)
        # resize images
        elif self.resize_to != (self.x.shape[1], self.x.shape[2]):
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
        log(
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
            log(f"Index: {index}, mod: {mod}")
            desc = f"""
        Type of image: {type(image)}
        Type of mask: {type(mask)}
        Shape of image: {image.shape}
        Shape of mask: {mask.shape}
            """
            log(desc)

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
            t_image = TF.rotate(image, angle=90)
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
            assert isinstance(
                t_image, torch.Tensor
            ), f"t_image should be a tensor, but is {type(t_image)}"
            assert isinstance(
                t_mask[0], torch.Tensor
            ), f"t_mask[0] should be a tensor, but is {type(t_mask[0])}"
            return t_image, t_mask[0]
        else:
            assert isinstance(
                t_image, torch.Tensor
            ), f"Mod {mod}\tt_image should be a tensor, but is {type(t_image)}"
            assert isinstance(
                t_mask, torch.Tensor
            ), f"Mod {mod}\tt_mask should be a tensor, but is {type(t_mask)}"
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


class OptimizedImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        device,
        type_="training",
        augment=False,
        crop=False,
        resize_to=None,
        crop_size=None,
        verbose=False,
    ):
        if crop and not crop_size:
            print("Crop size not set, default to 208")
            crop_size = 208

        if crop_size:
            assert isinstance(
                crop_size, int
            ), f"crop_size should be an int to define size = (crop_size, crop_size), but is {type_(crop_size)}"

        self.path = path
        self.device = device
        self.resize_to = resize_to
        self.x, self.y, self.n_samples = None, None, None
        self.augment = augment
        self.crop = crop
        self.crop_size = crop_size
        self.type = type_
        self.verbose = verbose
        self.N_TRANSFORMS = 6
        self._load_data()

        s = f"""
        {self.type.upper()} dataset, with:
        - {'TRANSFORMED' if self.augment else 'REGULAR'} dataset
        - {f'CROPPED TO {self.crop_size}' if self.crop else 'UNCROPPED'} dataset
        - {f'RESIZED TO {self.resize_to}' if self.resize_to else 'UNRESIZED'} dataset
        - {len(self)} SAMPLES in total
        """

        log(s)

    def __repr__(self) -> str:
        return super().__repr__()

    def _load_data(self):  # not very scalable, but good enough for now
        self.x = glob(os.path.join(self.path, "images") + "/*.png")
        self.y = glob(os.path.join(self.path, "groundtruth") + "/*.png")
        self.n_samples = len(self.x)

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
            t_mask = TF.hflip(mask)

        # Vertical flip
        elif mod == 2:
            t_image = TF.vflip(image)
            t_mask = TF.vflip(mask)

        elif mod == 3:
            t_image = TF.rotate(image, angle=-90)
            t_mask = TF.rotate(mask, angle=-90)

        elif mod == 4:
            t_image = TF.rotate(image, angle=90)
            t_mask = TF.rotate(mask, angle=90)

        elif mod == 5:
            _, h_i, w_i = image.shape

            resize_size = (h_i, w_i)

            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img=image,
                scale=(0.7, 0.9),
                ratio=(0.9, 1.1),
            )
            t_image = TF.resize(TF.crop(image, i, j, h, w), resize_size)
            t_mask = TF.resize(TF.crop(mask, i, j, h, w), resize_size)

        return t_image, t_mask

    def __getitem__(self, index):
        # Unravel index
        if self.crop and self.augment:
            img_index, transform_index, crop_index = np.unravel_index(
                index, (self.n_samples, self.N_TRANSFORMS, 4)
            )

            if self.verbose:
                print(
                    f"img_index: {img_index}, transform_index: {transform_index}, crop_index: {crop_index}"
                )

        elif self.augment:
            img_index, transform_index = np.unravel_index(
                index, (self.n_samples, self.N_TRANSFORMS)
            )

            if self.verbose:
                print(f"img_index: {img_index}, transform_index: {transform_index}")

        elif self.crop:
            img_index, crop_index = np.unravel_index(index, (self.n_samples, 4))
            if self.verbose:
                print(f"img_index: {img_index}, crop_index: {crop_index}")

        else:
            img_index = index

        # Select image and mask
        image = (
            np.array(Image.open(self.x[img_index])).astype(np.float32)[:, :, :3] / 255.0
        )
        image = np.moveaxis(image, -1, 0)
        mask = np.array(Image.open(self.y[img_index])).astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)

        image_tensor = np_to_tensor(image, self.device)
        mask_tensor = np_to_tensor(mask, self.device)

        if self.crop:
            image_tensor, mask_tensor = crop_to_size_with_crop_index(
                image_tensor, mask_tensor, crop_index, size=self.crop_size
            )

        if self.augment:
            image_tensor, mask_tensor = self.transform(
                image_tensor, mask_tensor, transform_index
            )

        if self.resize_to:
            image_tensor = TF.resize(image_tensor, self.resize_to)
            mask_tensor = TF.resize(mask_tensor, self.resize_to)

        return image_tensor, mask_tensor

    def __len__(self):
        if self.augment and self.crop:
            return self.n_samples * 4 * self.N_TRANSFORMS
        elif self.crop:
            return self.n_samples * 4
        elif self.augment:
            return self.n_samples * self.N_TRANSFORMS
        else:
            return self.n_samples
