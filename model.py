import torch
import torch.nn as nn


# 128_FCC-GAN
class Disc128(nn.Module):
    def __init__(self, img_channels, features):
        super(Disc128, self).__init__()

        self.disc = nn.Sequential(
            self._block(img_channels, features, kernel_size=4, stride=2, padding=1),
            self._block(features, features * 2, kernel_size=4, stride=2, padding=1),
            self._block(features * 2, features * 4, kernel_size=4, stride=2, padding=1),
            self._block(features * 4, features * 8, kernel_size=4, stride=2, padding=1),
            self._block(features * 8, features * 16, kernel_size=4, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(features * 16 * 4**2, 512),
            nn.Linear(512, 64),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)


class Gen128(nn.Module):
    def __init__(self, img_channels, z_size, features):
        super(Gen128, self).__init__()

        self.fcSeq = nn.Sequential(
            nn.Linear(z_size, 64),
            nn.Linear(64, 512),
            nn.Linear(512, features * 32 * 4 ** 2),
        )

        self.convTransSeq = nn.Sequential(
            self._block(features * 32, features * 16, kernel_size=4, stride=2, padding=1),
            self._block(features * 16, features * 8, kernel_size=4, stride=2, padding=1),
            self._block(features * 8, features * 4, kernel_size=4, stride=2, padding=1),
            self._block(features * 4, features * 2, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(features * 2, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.fcSeq(x)
        out = torch.reshape(out, (out.shape[0], out.shape[1] // 4**2, 4, 4))
        out = self.convTransSeq(out)

        return out


# 64_FCC-GAN
class Disc64(nn.Module):
    def __init__(self, img_channels, features):
        super(Disc64, self).__init__()

        self.disc = nn.Sequential(
            self._block(img_channels, features, kernel_size=4, stride=2, padding=1),
            self._block(features, features * 2, kernel_size=4, stride=2, padding=1),
            self._block(features * 2, features * 4, kernel_size=4, stride=2, padding=1),
            self._block(features * 4, features * 8, kernel_size=4, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(features * 8 * 4**2, 512),
            nn.Linear(512, 64),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)


class Gen64(nn.Module):
    def __init__(self, img_channels, z_size, features):
        super(Gen64, self).__init__()

        self.fcSeq = nn.Sequential(
            nn.Linear(z_size, 64),
            nn.Linear(64, 512),
            nn.Linear(512, features * 16 * 4**2),
        )

        self.convTransSeq = nn.Sequential(
            self._block(features * 16, features * 8, kernel_size=4, stride=2, padding=1),
            self._block(features * 8, features * 4, kernel_size=4, stride=2, padding=1),
            self._block(features * 4, features * 2, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(features * 2, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.fcSeq(x)
        out = torch.reshape(out, (out.shape[0], out.shape[1] // 4**2, 4, 4))
        out = self.convTransSeq(out)

        return out


def init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_size = 100

    x = torch.randn((N, in_channels, H, W))
    disc = Disc64(in_channels, 8)
    assert disc(x).shape == (N, 1), "Disc no work"

    gen = Gen64(in_channels, z_size, 8)
    z = torch.randn((N, z_size))
    assert gen(z).shape == (N, in_channels, H, W), "Gen no work"
