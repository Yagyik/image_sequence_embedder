import torch
import torch.nn as nn

class ImageEncoderDecoder(nn.Module):
    def __init__(self, nc=1, ngf=128, ndf=128, latent_variable_size=128, imsize=64, batchnorm=True):
        super(ImageEncoderDecoder, self).__init__()
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        self.imsize = imsize
        self.batchnorm = batchnorm

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2) if batchnorm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4) if batchnorm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8) if batchnorm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ndf * 8, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ndf * 4, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ndf * 2, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf) if batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ndf, nc, 4, 2, 1, bias=False),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x