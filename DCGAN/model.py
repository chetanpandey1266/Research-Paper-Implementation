import torch 
import torch.nn as nn

class Discriminator(nn.Module):
    """
    channels_img: number of channels in the input image

    features_d: a type of hyperparameter
    """
    def __init__(self, channels_img, features_d): 
        super(Discriminator, self).__init__()
        # LeakyReLU and BatchNorm is added so as to make the training of GAN stable
        self.net = nn.Sequential(
            # N X channels_img X 64 X 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # N X features_d X 32 X 32
            nn.Conv2d(features_d, features_d*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_d*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d*2, features_d*4, stride=2, kernel_size=4, padding=1),
            nn.BatchNorm2d(features_d*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d*4, features_d*8, stride=2, padding = 1, kernel_size = 4),
            nn.BatchNorm2d(features_d*8),
            nn.LeakyReLU(0.2),
            # N X features_d*8 X 4 X 4
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),
            # N X 1 X 1 X 1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)



class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            # N X channels_noise X 1 X 1
            nn.ConvTranspose2d(channels_noise, features_g*16, kernel_size = 4, stride = 1, padding=0),
            nn.BatchNorm2d(features_g*16),
            nn.ReLU(),
            
            # N X features_g*16 X 4 X 4
            nn.ConvTranspose2d(features_g*16, features_g*8, kernel_size = 4, stride = 2, padding=1),
            nn.BatchNorm2d(features_g*8),
            nn.ReLU(),

            nn.ConvTranspose2d(features_g*8, features_g*4, kernel_size = 4, stride = 2, padding=1),
            nn.BatchNorm2d(features_g*4),
            nn.ReLU(),

            nn.ConvTranspose2d(features_g*4, features_g*2, kernel_size = 4, stride = 2, padding=1),
            nn.BatchNorm2d(features_g*2),
            nn.ReLU(),

            nn.ConvTranspose2d(features_g*2, channels_img, kernel_size=4, stride=2, padding=1),
            # N X channels_img X 64 X 64
            nn.Tanh()
            # We have an image output size of 64X64 which we feed into the Discriminator which also takes the same size image as input
        )

    def forward(self, x):
        return self.net(x)