import torch
from utils import *


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, path, device, use_patches=True, resize_to=(400, 400)):
        self.path = path
        self.device = device
        self.use_patches = use_patches
        self.resize_to = resize_to
        self.x, self.y, self.n_samples = None, None, None
        self._load_data()

    def _load_data(self):  # not very scalable, but good enough for now
        self.x = load_all_from_path(os.path.join(self.path, "images"))[:, :, :, :3]
        self.y = load_all_from_path(os.path.join(self.path, "groundtruth"))
        if self.use_patches:  # split each image into patches
            self.x, self.y = image_to_patches(self.x, self.y)
        self.x = np.moveaxis(
            self.x, -1, 1
        )  # pytorch works with CHW format instead of HWC
        self.n_samples = len(self.x)

    def _preprocess(self, x, y):
        # to keep things simple we will not apply transformations to each sample,
        # but it would be a very good idea to look into preprocessing
        return x, y

    def __getitem__(self, item):
        return self._preprocess(
            np_to_tensor(self.x[item], self.device),
            np_to_tensor(self.y[[item]], self.device),
        )

    def __len__(self):
        return self.n_samples
