import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm


class PatchGAN(nn.Module):  # review
    def __init__(self):
        super(PatchGAN, self).__init__()
        # only bw images

        self.conv1 = spectral_norm(nn.Conv2d(1, 64,
                                             kernel_size=4))
        self.norm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.LeakyReLU()

        self.conv2 = spectral_norm(nn.Conv2d(64, 128,
                                             kernel_size=4))
        self.norm2 = nn.BatchNorm2d(128)
        self.relu2 = nn.LeakyReLU()

        self.conv3 = spectral_norm(nn.Conv2d(128, 1,
                                             kernel_size=4))
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        feature_maps = []
        x = self.conv1(x)
        feature_maps.append(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        feature_maps.append(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.sigm(x)
        return x, feature_maps
