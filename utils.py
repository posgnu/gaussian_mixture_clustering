import numpy as np
from scipy.stats import multivariate_normal
import pylab as P
import matplotlib.pyplot as plt


def BIC_score(log_likelihood, K, N):
    p_k = K - 1
    p_k += 2 * K
    p_k += 4 * K

    return log_likelihood - ((p_k / 2) * np.log(N))


def plot_gauss_parameters(mu, covar, delta=0.01):
    # make grid
    x = np.arange(
        mu[0] - 3.0 * np.sqrt(covar[0, 0]), mu[0] + 3.0 * np.sqrt(covar[0, 0]), delta
    )
    y = np.arange(
        mu[1] - 3.0 * np.sqrt(covar[1, 1]), mu[1] + 3.0 * np.sqrt(covar[1, 1]), delta
    )
    X, Y = np.meshgrid(x, y)

    # get pdf values
    mn = multivariate_normal(
        [mu[0], mu[1]],
        covar,
    )
    Z = mn.pdf(np.dstack((X, Y)))

    P.contour(X, Y, Z, linewidths=1)


def plot_data_and_gaussians(data, weights, gparams, K, delta=0.01):

    memberships = np.argmax(weights, axis=1)
    for i in range(K):
        P.scatter(data[memberships == i, 0], data[memberships == i, 1], alpha=0.65, s=2)

    for i in range(K):
        plot_gauss_parameters(gparams[i][1], gparams[i][2], delta)
        P.scatter(gparams[i][1][0], gparams[i][1][1], color="r", marker="x")

    P.show()


def plot_image(data, weights, gparams, K, delta=0.01):

    memberships = np.argmax(weights, axis=1)
    result = data.copy()

    for i in range(K):
        result[memberships == i] = gparams[i][1]

    plt.imshow(result.reshape((186, 175, 3)))
    plt.savefig(f"{K}.png")


def compute_log_likelihood(data, gparams):
    return np.sum(
        np.log(
            np.sum(
                [
                    alpha_k * multivariate_normal.pdf(data, mean_k, sigma_k)
                    for (alpha_k, mean_k, sigma_k) in gparams
                ],
                axis=0,
            )
        )
    )
