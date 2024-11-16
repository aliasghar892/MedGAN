# inspiration:
# https://github.com/milesial/Pytorch-UNet
# MedGAN paper:
# https://arxiv.org/pdf/1806.06397

# import torch
import torch.nn as nn
import torch.nn.functional as F


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=4, stride=2):
        super(Down, self).__init__()
        self.seq = nn.Sequential(
            # stride will do the downsampling (like maxpooling)
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.seq(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=4, stride=2, tanh=False, first=False):
        super(Up, self).__init__()
        # could use upsample
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel, stride=stride)
        self.temp = out_channels
        # self.up = nn.Upsample(
        #     scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv = nn.Conv2d(in_channels, out_channels,
        #                       kernel_size=kernel),
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU() if tanh == False else nn.Tanh()

    def forward(self, oldx, x):
        x = self.up(x)
        # input is [C,H,W]
        diffY = oldx.size()[2] - x.size()[2]
        diffX = oldx.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        # x = torch.cat([oldx, x], dim=1) review
        x = x+oldx
        x = self.norm(x)
        x = self.activation(x)
        return x


class UBlock(nn.Module):
    def __init__(self, in_channel):
        super(UBlock, self).__init__()
        # review , diffrent from paper !
        # first , how a 128 filter iamge should be used for final model
        # second , how cat should be donw with diffrent filter size of mirror layers
        # third , this number of down sampling will not work on a 256*256 img ?
        self.down1 = Down(in_channel, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 512)
        self.down6 = Down(512, 512)
        self.down7 = Down(512, 512)
        self.down8 = Down(512, 512)
        self.up1 = Up(512, 512)
        self.up2 = Up(512, 512)
        self.up3 = Up(512, 512)
        self.up4 = Up(512, 512)
        self.up5 = Up(512, 256)
        self.up6 = Up(256, 128)
        self.up7 = Up(128, 64)
        self.up8 = Up(64, in_channel, tanh=True)

    def forward(self, x):
        # only bw images for now
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)
        x8 = self.down8(x7)
        x9 = self.up1(x=x8, oldx=x7)  # review
        x9 = self.up2(x=x9, oldx=x6)
        x9 = self.up3(x=x9, oldx=x5)
        x9 = self.up4(x=x9, oldx=x4)
        x9 = self.up5(x=x9, oldx=x3)
        x9 = self.up6(x=x9, oldx=x2)
        x9 = self.up7(x=x9, oldx=x1)
        x9 = self.up8(x=x9, oldx=x)
        return x9


class CasNet(nn.Module):
    def __init__(self, num_ublocks):
        super(CasNet, self).__init__()
        self.n = num_ublocks
        # Register UBlock layers
        self.ublocks = nn.ModuleList([UBlock(1) for _ in range(self.n)])

    def forward(self, x):
        for ublock in self.ublocks:
            x = ublock(x)
        return x
