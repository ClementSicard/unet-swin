import torch


class DecoderBlock(torch.nn.Module):
    def __init__(
        self,
        down_channels,
        up_channels,
        kernel_size_up,
        stride_up,
        kernel_size,
        # dropout=0.0,
    ):
        super(DecoderBlock, self).__init__()

        self.up = torch.nn.ConvTranspose2d(
            down_channels, up_channels, kernel_size=kernel_size_up, stride=stride_up
        )

        self.conv1 = torch.nn.Conv2d(
            down_channels, up_channels, kernel_size=kernel_size, padding=1
        )
        self.conv2 = torch.nn.Conv2d(
            up_channels, up_channels, kernel_size=kernel_size)
        # self.dropout = torch.nn.Dropout(dropout)

        # self.last_conv = torch.nn.Conv2d
        # self.last_layer_up = torch.nn.Sequential(
        #     [
        #         torch.nn.ConvTranspose2d(96, 33, kernel_size=2, stride=2),
        #         torch.nn.Conv2d(33, 3, kernel_size=2, stride=2),
        #     ]
        # )

    def forward(self, x, skip):
        # print(x.shape, skip.shape, self.up(x).shape, flush=True)
        x = self.up(x)
        if x.shape[2] > skip.shape[2]:
            # when x is [batch, channels, 13, 13] to [batch, channels, 25, 25]
            # we have to crop it
            x = x[:, :, :-1, :-1]

        # print(x.shape, skip.shape, flush=True)
        x = torch.cat([x, skip], dim=1)
        # print(x.shape, skip.shape, flush=True)
        x = torch.nn.functional.relu(self.conv1(x))
        # print(x.shape)
        # x = self.dropout(x)
        # x = torch.nn.functional.relu(self.conv2(x))
        return x


class Decoder(torch.nn.Module):
    def __init__(self, sizes) -> None:
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.blocks = []
        for size in sizes:
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
        self.last_conv1 = torch.nn.Conv2d(
            FINAL_CHANNEL//4, FINAL_CHANNEL//8, kernel_size=3, padding=1)
        self.last_relu = torch.nn.ReLU()
        self.last_conv2 = torch.nn.Conv2d(
            FINAL_CHANNEL//16, FINAL_CHANNEL//32, kernel_size=3, padding=1)
        self.last_upX4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                FINAL_CHANNEL*2, FINAL_CHANNEL*2, kernel_size=2, stride=2),
            torch.nn.ConvTranspose2d(
                FINAL_CHANNEL*2, FINAL_CHANNEL, kernel_size=2, stride=2),
        )
        self.last_convs = torch.nn.Sequential(
            torch.nn.Conv2d(
                FINAL_CHANNEL + 3, FINAL_CHANNEL//2, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                FINAL_CHANNEL//2, FINAL_CHANNEL//32, kernel_size=3, padding=1),
            torch.nn.ReLU()
        )

    def forward(self, x, skips):
        for block, skip in zip(self.blocks, skips[:-1]):
            # print("wow")
            # print(x.shape, skip.shape, flush=True)
            x = block(x, skip)
            # skip = x
        # x should be of size [batch, 6, 400, 400]
        x = self.last_upX4(x)
        # print(x.shape, skips[-1].shape)
        x = torch.cat([x, skips[-1]], dim=1)
        x = self.last_convs(x)
        # print(x.shape)
        # x = self.last_conv1(x)
        # x = self.last_relu(x)
        # x = self.last_conv2(x)
        # x is of size [batch, 1, 400, 400]
        # print("final output", x.shape)
        return x
