from collections import OrderedDict
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ngpu = config.model.infra.ngpu
        self.hyperparameters = config.model.hyperparameters
        self.discriminator_net = nn.Sequential(OrderedDict([
            # input is (nc) x 64 x 64
            ('conv1', nn.Conv2d(config.data.nc, self.hyperparameters.ndf, 4, 2, 1, bias=False)),
            ('relu1', nn.LeakyReLU(0.2, inplace=True)),
            # state size. (ndf) x 32 x 32
            ('conv2', nn.Conv2d(self.hyperparameters.ndf, self.hyperparameters.ndf * 2, 4, 2, 1, bias=False)),
            ('batchn2', nn.BatchNorm2d(self.hyperparameters.ndf * 2)),
            ('relu2', nn.LeakyReLU(0.2, inplace=True)),
            # state size. (ndf*2) x 16 x 16
            ('conv3', nn.Conv2d(self.hyperparameters.ndf * 2, self.hyperparameters.ndf * 4, 4, 2, 1, bias=False)),
            ('batchn3', nn.BatchNorm2d(self.hyperparameters.ndf * 4)),
            ('relu3', nn.LeakyReLU(0.2, inplace=True)),
            # state size. (ndf*4) x 8 x 8
            ('conv4', nn.Conv2d(self.hyperparameters.ndf * 4, self.hyperparameters.ndf * 8, 4, 2, 1, bias=False)),
            ('batchn4', nn.BatchNorm2d(self.hyperparameters.ndf * 8)),
            ('rel4', nn.LeakyReLU(0.2, inplace=True)),
            # state size. (ndf*8) x 4 x 4
            ('conv5', nn.Conv2d(self.hyperparameters.ndf * 8, 1, 4, 1, 0, bias=False)),
            ('sigmout', nn.Sigmoid())
        ]))

    def forward(self, input):
        return self.discriminator_net(input)