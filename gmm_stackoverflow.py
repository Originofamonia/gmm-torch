# https://stackoverflow.com/questions/65755730/estimating-mixture-of-gaussian-models-in-pytorch

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

import torch
from torch import nn
from torch import optim
import torch.distributions as D

from example import make_ellipse


def plot_gmm(data, mu, sigma, filename):
    ax1 = plt.subplot(111, aspect='auto')
    xy_lim = 10
    mu = np.squeeze(mu.data.cpu().numpy())
    sigma = np.squeeze(sigma.data.cpu().numpy())
    ax1.scatter(data[:, 0], data[:, 1])

    make_ellipse(mu, sigma, ax1, xy_lim)

    plt.tight_layout()
    plt.savefig(filename)


def main():
    k = 4
    dim = 2  # inputs_dim
    filename = 'points.npz'
    points = np.load(filename, allow_pickle=True)['arr_0']
    # points = np.concatenate(points, axis=0)
    points = points[0].astype('float32')
    print(np.mean(points, axis=0))
    # weights = torch.ones(k, requires_grad=True)
    # means = torch.tensor(np.random.randn(k, dim), requires_grad=True)
    # stddevs = torch.tensor(np.abs(np.random.randn(k, dim)), requires_grad=True)
    mu = torch.zeros(dim, requires_grad=True)
    sigma = torch.eye(dim, requires_grad=True)

    # parameters = [weights, means, stddevs]
    parameters = [mu, sigma]
    optimizer1 = optim.SGD(parameters, lr=1e-3, momentum=0.9)
    # m_list = []
    # for i in range(k):
    #     m_list.append(D.MultivariateNormal(torch.zeros(dim), torch.eye(dim)))

    num_iter = 351
    for i in range(num_iter):
        # mix = D.Categorical(weights)
        # comp = D.Independent(D.Normal(means, stddevs), 1)
        # gmm = D.MixtureSameFamily(mix, comp)
        mm = D.MultivariateNormal(mu, sigma)
        optimizer1.zero_grad()
        # x = torch.randn(5000, 2)  # this can be an arbitrary x samples
        x = torch.from_numpy(points)
        loss2 = -mm.log_prob(x).mean()  # -densityflow.log_prob(inputs=x).mean()
        loss2.backward()
        optimizer1.step()

        print(i, loss2.item())
    plot_gmm(points, mu, sigma, 'gmm_stackoverflow.pdf')


if __name__ == '__main__':
    main()
