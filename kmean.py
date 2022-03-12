import numpy as np
from scipy.spatial import distance


def kmean(point_list: list, num_cluster: int) -> tuple:
    means_result = []
    squared_error_result = None
    assignments_result = None

    for _ in range(10):
        # Initialize mean
        index = np.random.choice(point_list.shape[0], num_cluster, replace=False)
        means = point_list[index]
        assignments = np.zeros(point_list.shape[0])
        squared_error = []

        for _ in range(500):
            # M step
            counter = 0
            for idx, point in enumerate(point_list):
                distance_list = [distance.euclidean(point, mean) for mean in means]
                new_assignment = np.argmin(distance_list)

                if assignments[idx] != new_assignment:
                    counter += 1
                assignments[idx] = new_assignment

            # E step
            for assignment in range(means.shape[0]):
                new_mean = np.mean(point_list[assignments == assignment], axis=0)

                means[assignment] = new_mean

            squared_error.append(
                sum(
                    [
                        sum((point - means[int(assignment)]) ** 2)
                        for point, assignment in zip(point_list, assignments)
                    ]
                )
            )

            if counter < 2:
                break

        if (squared_error_result == None) or (
            squared_error_result[-1] > squared_error[-1]
        ):
            squared_error_result = squared_error
            means_result = means
            assignments_result = assignments

    return means_result, squared_error_result, assignments_result
