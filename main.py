from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_image
from gaussian_mixture import gaussian_mixture

plt.style.use("seaborn")

img = Image.open("./data/clown.jpg")
arr = np.array(img)
z = arr[:, :, :3]
z = z.reshape(z.shape[0] * z.shape[1], 3)

for k in range(2, 11):
    (
        gparams_result,
        membership,
        initial_gparams_result,
        log_likelihood_list,
        initial_membership_result,
    ) = gaussian_mixture(z, k)

    plot_image(z, membership, gparams_result, k)
