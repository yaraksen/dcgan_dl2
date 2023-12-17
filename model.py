import torch
from torch import nn
from torch import Tensor
import math
from numpy import prod

# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#data


class BaseModel(nn.Module):
    def __str__(self):
        """
        Model prints with number of trainable parameters
        Taken from our DLA homework template
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


class Generator(BaseModel):
    def __init__(self, latent_dim, G_feature_map_dim, image_num_channels, **kwargs):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, G_feature_map_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(G_feature_map_dim * 8),
            nn.ReLU(True),
            # state size. ``(G_feature_map_dim*8) x 4 x 4``
            nn.ConvTranspose2d(
                G_feature_map_dim * 8, G_feature_map_dim * 4, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(G_feature_map_dim * 4),
            nn.ReLU(True),
            # state size. ``(G_feature_map_dim*4) x 8 x 8``
            nn.ConvTranspose2d(
                G_feature_map_dim * 4, G_feature_map_dim * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(G_feature_map_dim * 2),
            nn.ReLU(True),
            # state size. ``(G_feature_map_dim*2) x 16 x 16``
            nn.ConvTranspose2d(
                G_feature_map_dim * 2, G_feature_map_dim, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(G_feature_map_dim),
            nn.ReLU(True),
            # state size. ``(G_feature_map_dim) x 32 x 32``
            nn.ConvTranspose2d(
                G_feature_map_dim, image_num_channels, 4, 2, 1, bias=False
            ),
            nn.Tanh()
            # state size. ``(image_num_channels) x 64 x 64``
        )

        self.apply(self.weights_init)

    def forward(self, input):
        return self.main(input)


class Discriminator(BaseModel):
    def __init__(self, D_feature_map_dim, image_num_channels, **kwargs):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is ``(image_num_channels) x 64 x 64``
            nn.Conv2d(image_num_channels, D_feature_map_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(D_feature_map_dim) x 32 x 32``
            nn.Conv2d(D_feature_map_dim, D_feature_map_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_feature_map_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(D_feature_map_dim*2) x 16 x 16``
            nn.Conv2d(
                D_feature_map_dim * 2, D_feature_map_dim * 4, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(D_feature_map_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(D_feature_map_dim*4) x 8 x 8``
            nn.Conv2d(
                D_feature_map_dim * 4, D_feature_map_dim * 8, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(D_feature_map_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(D_feature_map_dim*8) x 4 x 4``
            nn.Conv2d(D_feature_map_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

        self.apply(self.weights_init)

    def forward(self, input):
        return self.main(input)
