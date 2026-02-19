import torch

from torch import nn

from module.module import Module



class LinearRegression(Module):

    

    def __init__(self, lr, wd):

        super().__init__()

        self.lr = lr

        self.wd = wd

        self.net = nn.LazyLinear(1)

        self.net.weight.data.normal_(0, 0.01)

        self.net.bias.data.fill_(0)



    def loss(self, y_hat, y):

        fn = nn.MSELoss()

        return fn(y_hat, y)



    def forward(self, X):

        return self.net(X)



    def configure_optimizer(self):

        return torch.optim.SGD([

            {'params': self.net.weight, 'weight_decay': self.wd},

            {'params': self.net.bias}

        ])
