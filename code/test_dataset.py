from dataset import ImageDataset
import torch

dataset = ImageDataset(
    "../data/training", device="cpu", augment=True, use_patches=False
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

count = 0
for x, y in dataloader:
    print(x.shape)
    print(y.shape)
    count += 1
    if count == 10:
        break
