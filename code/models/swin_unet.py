import torch
from .Encoders.SwinSmall import swin_pretrained
from .Decoders.CustomDecoder import Decoder
from dataset import ImageDataset
from train import train
from utils import *
from datetime import datetime

INFERED_SIZES = [(768, 384), (384, 192), (192, 96), (96, 3)]


class SwinUnet(torch.nn.Module):
    def __init__(
        self
    ):
        super(SwinUnet, self).__init__()
        self.encoder = swin_pretrained().to("cpu")
        self.decoder = Decoder(sizes=INFERED_SIZES)
        self.prev_conv = torch.nn.Conv2d(
            768, 768, kernel_size=3, padding=1, bias=False)
        self.fully_connected = torch.nn.Linear(16*16, 1)

    def forward(self, x):
        # askip on preprocess les images
        x = self.encoder(x)
        x = self.prev_conv(x)
        self.encoder.x_int.reverse()
        for int in self.encoder.x_int[1::]:
            print(int.shape)
        x = self.decoder(x, self.encoder.x_int[1::])
        x = self.fully_connected(x.view(x.shape[0], -1))
        x = torch.sigmoid(x)
        return x


def run(train_path: str, val_path: str, test_path: str, n_epochs=20, batch_size=128):
    print("Training Patch-CNN Baseline...")
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # automatically select device
    train_dataset = ImageDataset(train_path, device, augment=False)
    val_dataset = ImageDataset(val_path, device, augment=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )
    model = SwinUnet().to(device)
    loss_fn = torch.nn.BCELoss()
    # loss_fn = BinaryDiceLoss()
    metric_fns = {"acc": accuracy_fn}
    optimizer = torch.optim.Adam(model.parameters())

    train(
        train_dataloader,
        val_dataloader,
        model,
        loss_fn,
        metric_fns,
        optimizer,
        n_epochs,
        "swin-unet"
    )

    print("Training done!")

    print("Predicting on test set...")
    # predict on test set
    test_path = os.path.join(test_path, "images")
    test_filenames = sorted(glob(test_path + "/*.png"))
    test_images = load_all_from_path(test_path)
    test_images = test_images[:, :, :, :3]
    print(f"{test_images.shape[0]} were loaded")
    test_patches = np.moveaxis(image_to_patches(
        test_images), -1, 1)  # HWC to CHW
    test_patches = np.reshape(
        test_patches, (25, -1, 3, PATCH_SIZE, PATCH_SIZE)
    )  # split in batches for memory constraints
    test_pred = [
        model(np_to_tensor(batch, device)).detach().cpu().numpy()
        for batch in test_patches
    ]
    test_pred = np.concatenate(test_pred, 0)
    test_pred = np.round(
        test_pred.reshape(
            test_images.shape[0],
            test_images.shape[1] // PATCH_SIZE,
            test_images.shape[1] // PATCH_SIZE,
        )
    )
    print(f"Test predictions shape: {test_pred.shape}")
    now = datetime.now()
    t = now.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("submissions", exist_ok=True)
    create_submission(
        test_pred,
        test_filenames,
        submission_filename=f"./submissions/swin_submission_{t}.csv",
    )
    print(f"Created submission!")
