# https://stackoverflow.com/questions/65755730/estimating-mixture-of-gaussian-models-in-pytorch

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

import torch
from torch import nn
from torch import optim
import torch.distributions as D


def main():
    k = 4
    dim = 2  # inputs_dim
    filename = 'points.npz'
    points = np.load(filename, allow_pickle=True)['arr_0']

    weights = torch.ones(k, requires_grad=True)
    means = torch.tensor(np.random.randn(k, dim), requires_grad=True)
    stddevs = torch.tensor(np.abs(np.random.randn(k, dim)), requires_grad=True)

    parameters = [weights, means, stddevs]
    optimizer1 = optim.SGD(parameters, lr=1e-3, momentum=0.9)

    num_iter = 1001
    for i in range(num_iter):
        mix = D.Categorical(weights)
        comp = D.Independent(D.Normal(means, stddevs), 1)
        gmm = D.MixtureSameFamily(mix, comp)

        optimizer1.zero_grad()
        # x = torch.randn(5000, 2)  # this can be an arbitrary x samples
        x = torch.from_numpy(points)
        loss2 = -gmm.log_prob(x).mean()  # -densityflow.log_prob(inputs=x).mean()
        loss2.backward()
        optimizer1.step()

        print(i, loss2.item())
    print(weights, means, stddevs)


if __name__ == '__main__':
    main()
