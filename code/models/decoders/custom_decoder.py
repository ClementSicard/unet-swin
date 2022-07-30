import torch
import torch.nn as nn


class Block(nn.Module):
    # a repeating structure composed of two convolutional layers with batch normalization and ReLU activations
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(
                in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        down_channels,
        up_channels,
        kernel_size_up,
        stride_up,
        kernel_size,
        up_padding=0,
    ):
        super(DecoderBlock, self).__init__()

        self.up = nn.ConvTranspose2d(
            down_channels, up_channels, kernel_size=kernel_size_up, stride=stride_up, padding=up_padding
        )

        self.block = nn.Sequential(
            nn.Conv2d(down_channels, up_channels,
                      kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(up_channels),
            nn.Conv2d(up_channels, up_channels,
                      kernel_size=kernel_size, padding=1),
            nn.ReLU(),
        )

    def forward(self, x, skip):
        # print(x.shape, skip.shape, self.up(x).shape, flush=True)
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class Decoder(nn.Module):
    def __init__(self, sizes) -> None:
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.blocks = []
        for i, size in enumerate(sizes):
            if i == 0:
                self.blocks.append(
                    DecoderBlock(
                        size[0],
                        size[1],
                        kernel_size_up=3,
                        up_padding=1,
                        stride_up=2,
                        kernel_size=3,)
                )
            else:
                self.blocks.append(
                    DecoderBlock(
                        size[0],
                        size[1],
                        kernel_size_up=2,
                        stride_up=2,
                        kernel_size=3,
                        # dropout=0.0,
                    ).to(device)
                )
        FINAL_CHANNEL = sizes[-1][-1]
        self.last_upX4 = nn.Sequential(
            nn.ConvTranspose2d(
                FINAL_CHANNEL*2, FINAL_CHANNEL*2, kernel_size=2, stride=2),
            nn.ConvTranspose2d(
                FINAL_CHANNEL*2, FINAL_CHANNEL, kernel_size=2, stride=2),
        )
        self.last_convs = nn.Sequential(
            nn.Conv2d(
                FINAL_CHANNEL * 2, FINAL_CHANNEL, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(FINAL_CHANNEL),
            nn.Conv2d(
                FINAL_CHANNEL, FINAL_CHANNEL, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x, skips):
        # We apply the same block to all the skips expect the last one as we need first to upscale the image 4 times
        for block, skip in zip(self.blocks, skips[:-1]):
            x = block(x, skip)
        x = self.last_upX4(x)
        x = torch.cat([x, skips[-1]], dim=1)
        x = self.last_convs(x)
        return x
