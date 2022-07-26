import torch


class DecoderBlock(torch.nn.Module):
    def __init__(
        self, down_channels, up_channels, kernel_size_up, kernel_size, dropout=0.0
    ):
        super(DecoderBlock, self).__init__()

        self.up = torch.nn.ConvTranspose2d(
            down_channels, up_channels, kernel_size=kernel_size_up
        )

        self.conv1 = torch.nn.Conv2d(
            down_channels, up_channels, kernel_size=kernel_size
        )
        self.conv2 = torch.nn.Conv2d(up_channels, up_channels, kernel_size=kernel_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, skip):
        print(x.shape, skip.shape, self.up(x).shape, flush=True)
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.conv2(x))
        return x


class Decoder(torch.nn.Module):
    def __init__(self, sizes) -> None:
        super().__init__()
        self.blocks = []
        for size in sizes:
            self.blocks.append(
                DecoderBlock(
                    size[0],
                    size[1],
                    kernel_size_up=2,
                    kernel_size=3,
                    dropout=0.0,
                )
            )

    def forward(self, x, skips):
        for block, skip in zip(self.blocks, skips):
            x = block(x, skip)
            # skip = x
        return x
