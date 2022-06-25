from collections import OrderedDict
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ngpu = config.model.infra.ngpu
        self.hyperparameters = config.model.hyperparameters
        self.generator_net = nn.Sequential(OrderedDict([
            # input is Z, going into a convolution
            ('convt1', nn.ConvTranspose2d(self.hyperparameters.nz, self.hyperparameters.ngf * 8, 4, 1, 0, bias=False)),
            ('batchn1', nn.BatchNorm2d(self.hyperparameters.ngf * 8)),
            ('relu1', nn.ReLU(True)),
            # state size. (ngf*8) x 4 x 4
            ('convt2', nn.ConvTranspose2d(self.hyperparameters.ngf * 8, self.hyperparameters.ngf * 4, 4, 2, 1, bias=False)),
            ('batchn2', nn.BatchNorm2d(self.hyperparameters.ngf * 4)),
            ('relu2', nn.ReLU(True)),
            # state size. (ngf*4) x 8 x 8
            ('convt3', nn.ConvTranspose2d(self.hyperparameters.ngf * 4, self.hyperparameters.ngf * 2, 4, 2, 1, bias=False)),
            ('batchn3', nn.BatchNorm2d(self.hyperparameters.ngf * 2)),
            ('relu3', nn.ReLU(True)),
            # state size. (ngf*2) x 16 x 16
            ('convt4', nn.ConvTranspose2d(self.hyperparameters.ngf * 2, self.hyperparameters.ngf, 4, 2, 1, bias=False)),
            ('batchn4', nn.BatchNorm2d(self.hyperparameters.ngf)),
            ('relu4', nn.ReLU(True)),
            # state size. (ngf) x 32 x 32
            ('convt5', nn.ConvTranspose2d(self.hyperparameters.ngf, config.data.nc, 4, 2, 1, bias=False)),
            ('tanhout', nn.Tanh())
            # state size. (nc) x 64 x 64
        ]))

    def forward(self, input):
        return self.generator_net(input)