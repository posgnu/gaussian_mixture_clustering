from kmean import kmean
import numpy as np
import pylab as P
import matplotlib.pyplot as plt
from gaussian_mixture import gaussian_mixture
from utils import BIC_score, plot_data_and_gaussians, compute_log_likelihood
from sklearn.model_selection import train_test_split

plt.style.use("seaborn")

data1 = np.genfromtxt("./data/dataset1.txt")
data2 = np.genfromtxt("./data/dataset2.txt")
data3 = np.genfromtxt("./data/dataset3.txt")
np.random.shuffle(data1)
np.random.shuffle(data2)
np.random.shuffle(data3)

data_list = [(2, data1), (3, data2), (2, data3)]

for idx_data, (k, data) in enumerate(data_list):
    means_result, squared_error_result, assignments_result = kmean(data, k)

    # Problem (a)
    for idx in range(k):
        P.scatter(
            data[assignments_result == idx, 0],
            data[assignments_result == idx, 1],
            s=20,
            marker=".",
            alpha=0.65,
            linewidths=2,
        )
        P.scatter(means_result[idx][0], means_result[idx][1], color="r", marker="x")
        P.annotate(
            "mean",
            xy=(means_result[idx][0], means_result[idx][1]),
            color="r",
        )
    P.show()

    # Problem (b)
    plt.plot(list(range(len(squared_error_result))), squared_error_result)
    plt.show()

    (
        gparams_result,
        membership,
        initial_gparams_result,
        log_likelihood_list,
        initial_membership_result,
    ) = gaussian_mixture(
        data,
        k,
    )

    # Probelm (c)
    plot_data_and_gaussians(data, membership, gparams_result, k)
    plot_data_and_gaussians(data, initial_membership_result, initial_gparams_result, k)

    # Problem(d)
    plt.plot(list(range(len(log_likelihood_list))), log_likelihood_list)
    plt.show()

    # Problem (f)
    print(f"dataset {idx_data}:")
    for k_candidate in range(1, 6):
        (
            gparams_result,
            membership,
            initial_gparams_result,
            log_likelihood_list,
            initial_membership_result,
        ) = gaussian_mixture(
            data,
            k_candidate,
        )

        log_likelihood = log_likelihood_list[-1]
        bic_score = BIC_score(log_likelihood, k_candidate, data.shape[0])

        print(f"log_likelihood: {log_likelihood}")
        print(f"BIC score: {bic_score}")


for idx_data, (k, data) in enumerate(data_list):
    train_data, test_data = train_test_split(data)

    print(f"dataset {idx_data}")
    for k_candidate in range(1, 6):
        (
            gparams_result,
            membership,
            initial_gparams_result,
            log_likelihood_list,
            initial_membership_result,
        ) = gaussian_mixture(
            train_data,
            k_candidate,
        )

        train_log_likelihood = log_likelihood_list[-1]
        test_log_likelihood = compute_log_likelihood(test_data, gparams_result)
        print(f"train log likelihood = {train_log_likelihood}")
        print(f"test log likelihood = {test_log_likelihood}")

