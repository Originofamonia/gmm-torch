import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set(style="white", font="Arial")
colors = sns.color_palette("Paired", n_colors=12).as_hex()

import numpy as np
import torch

from gmm import GaussianMixture
from math import sqrt


def main():
    n, d = 300, 2

    # generate some data points ..
    data = torch.Tensor(n, d).normal_()
    # .. and shift them around to non-standard Gaussians
    data[:n//2] -= 1
    data[:n//2] *= sqrt(3)
    data[n//2:] += 1
    data[n//2:] *= sqrt(2)

    # Next, the Gaussian mixture is instantiated and ..
    n_components = 2
    model = GaussianMixture(n_components, d)
    model.fit(data)
    # .. used to predict the data points as they where shifted
    y = model.predict(data)
    plot_gmm(data, model)
    # plot(data, y)


def plot_gmm(data, model):
    ax1 = plt.subplot(111, aspect='auto')
    xy_lim = 10
    mu = np.squeeze(model.mu.data.cpu().numpy())
    sigma = np.squeeze(model.var.data.cpu().numpy())
    ax1.scatter(data[:, 0], data[:, 1])
    for mu_i, sigma_i in zip(mu, sigma):
        make_ellipse(mu_i, sigma_i, ax1, xy_lim)
    plt.tight_layout()
    plt.savefig("draw_gmm.pdf")


def make_ellipse(mu, sigma, ax, xy_lim, edgecolor='black'):
    cov = sigma
    # pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # # Using a special case to obtain the eigenvalues of this
    # # two-dimensionl dataset.
    # ell_radius_x = np.sqrt(1 + pearson)
    # ell_radius_y = np.sqrt(1 - pearson)
    v, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    v = 3. * np.sqrt(2.) * np.sqrt(v)
    mean = mu
    mean = mean.reshape(2, 1)
    print(mean)
    ell = mpl.patches.Ellipse(mean, v[0], v[1],
                              180 + angle, edgecolor=edgecolor, linestyle=':',
                              lw=4, facecolor='none')
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.5)
    ax.add_artist(ell)
    ax.set_xlim(-xy_lim, xy_lim)
    ax.set_ylim(-xy_lim, xy_lim)


def plot(data, y):
    n = y.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=(1.61803398875*4, 4))
    ax.set_facecolor('#bbbbbb')
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    # plot the locations of all data points ..
    for i, point in enumerate(data.data):
        if i <= n//2:
            # .. separating them by ground truth ..
            ax.scatter(*point, color="#000000", s=3, alpha=.75, zorder=n+i)
        else:
            ax.scatter(*point, color="#ffffff", s=3, alpha=.75, zorder=n+i)

        if y[i] == 0:
            # .. as well as their predicted class
            ax.scatter(*point, zorder=i, color="#dbe9ff", alpha=.6, edgecolors=colors[1])
        else:
            ax.scatter(*point, zorder=i, color="#ffdbdb", alpha=.6, edgecolors=colors[5])

    handles = [plt.Line2D([0], [0], color='w', lw=4, label='Ground Truth 1'),
        plt.Line2D([0], [0], color='black', lw=4, label='Ground Truth 2'),
        plt.Line2D([0], [0], color=colors[1], lw=4, label='Predicted 1'),
        plt.Line2D([0], [0], color=colors[5], lw=4, label='Predicted 2')]

    legend = ax.legend(loc="best", handles=handles)

    plt.tight_layout()
    plt.savefig("example.pdf")


if __name__ == "__main__":
    main()
