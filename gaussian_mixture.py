import numpy as np
from scipy.stats import multivariate_normal
from utils import compute_log_likelihood
from kmean import kmean


def gaussian_mixture(
    data, K, init_method=2, epsilon=10e-6, niterations=500, plotflag=0, RSEED=123
):
    log_likelihood_list_result = None
    initial_gparams_result = None
    gparams_result = None
    initial_membership_result = None
    current_log_likelihood = float("-inf")

    for _ in range(10):
        log_likelihood_list = []

        # initialize....
        gparams = [0 for _ in range(K)]
        membership = np.random.rand(data.shape[0], K)
        membership = membership / np.sum(membership, axis=1).reshape(data.shape[0], 1)
        initialize(data, K, init_method, gparams)
        initial_gparams = gparams.copy()

        for epoch in range(niterations):
            # Perform E-step...
            for k in range(K):
                alpha, mean, sigma = gparams[k]

                membership[:, k] = (
                    alpha
                    * multivariate_normal.pdf(data, mean, sigma)
                    / np.sum(
                        [
                            alpha_m * multivariate_normal.pdf(data, mean_m, sigma_m)
                            for (alpha_m, mean_m, sigma_m) in gparams
                        ],
                        axis=0,
                    )
                )

            if epoch == 0:
                initial_membership = membership.copy()

            # Perform M-step...
            for k in range(K):
                # Weight
                n_k = sum(membership[:, k])
                weight = n_k / data.shape[0]

                # Mean
                new_mean_k = (
                    np.sum(
                        (membership[:, k]).reshape(membership[:, k].shape[0], 1) * data,
                        axis=0,
                    )
                    / n_k
                )

                # Covariance
                new_sigma_k = (1 / n_k) * np.dot(
                    (
                        membership[:, k].reshape(data.shape[0], 1) * (data - new_mean_k)
                    ).T,
                    (data - new_mean_k),
                )

                # Check sigularity
                for idx in range(data.shape[1]):
                    variance = (
                        np.dot((data - new_mean_k)[:, idx], (data - new_mean_k)[:, idx])
                        / data.shape[0]
                    )

                    if new_sigma_k[idx][idx] < 10e-3 * variance:
                        new_sigma_k[idx][idx] = 10e-3 * variance

                gparams[k] = (weight, new_mean_k, new_sigma_k)

            # Compute log-likelihood and print to screen.....
            log_likelihood = compute_log_likelihood(data, gparams)

            log_likelihood_list.append(log_likelihood)

            if len(log_likelihood_list) >= 2:
                pass
                # assert (log_likelihood_list[-1] >= log_likelihood_list[-2]), f"{log_likelihood_list}"

            # Check for convergence.....
            if (
                len(log_likelihood_list) >= 2
                and abs(log_likelihood_list[-1] - log_likelihood_list[-2]) < epsilon
            ):
                break

        if check_global_optimum(
            log_likelihood_list_result, current_log_likelihood, log_likelihood_list
        ):
            current_log_likelihood = log_likelihood_list[-1]

            log_likelihood_list_result = log_likelihood_list
            initial_gparams_result = initial_gparams
            gparams_result = gparams
            initial_membership_result = initial_membership

    return (
        gparams_result,
        membership,
        initial_gparams_result,
        log_likelihood_list,
        initial_membership_result,
    )


def check_global_optimum(
    log_likelihood_list_result, current_log_likelihood, log_likelihood_list
):
    return (
        log_likelihood_list_result is None
        or current_log_likelihood < log_likelihood_list[-1]
    )


def initialize(data, K, init_method, gparams):
    if init_method == 1:
        raise NotImplemented
    elif init_method == 2:
        initial_means, _, _ = kmean(data, K)
        assert len(initial_means) == K
        for k, mean_k in enumerate(initial_means):
            weight = 1 / K
            covariance = np.cov(data.T) / K
            gparams[k] = (weight, mean_k, covariance)
    else:
        raise NotImplemented
