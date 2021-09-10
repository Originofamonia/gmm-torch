# https://stackoverflow.com/questions/65755730/estimating-mixture-of-gaussian-models-in-pytorch

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

import torch
from torch import nn
from torch import optim
import torch.distributions as D


def main():
    num_layers = 8
    weights = torch.ones(8, requires_grad=True)
    means = torch.tensor(np.random.randn(8, 2), requires_grad=True)
    stdevs = torch.tensor(np.abs(np.random.randn(8, 2)), requires_grad=True)

    parameters = [weights, means, stdevs]
    optimizer1 = optim.SGD(parameters, lr=0.001, momentum=0.9)

    num_iter = 10001
    for i in range(num_iter):
        mix = D.Categorical(weights)
        comp = D.Independent(D.Normal(means, stdevs), 1)
        gmm = D.MixtureSameFamily(mix, comp)

        optimizer1.zero_grad()
        x = torch.randn(5000, 2)  # this can be an arbitrary x samples
        loss2 = -gmm.log_prob(x).mean()  # -densityflow.log_prob(inputs=x).mean()
        loss2.backward()
        optimizer1.step()

        print(i, loss2)


if __name__ == '__main__':
    main()
