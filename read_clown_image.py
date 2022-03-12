from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn")

img = Image.open("./data/clown.jpg")
arr = np.array(img)
z = arr[:, :, :3]

r = z[:, :, 0]
g = z[:, :, 1]
b = z[:, :, 2]

plt.scatter(r, g, s=2)
plt.show()

plt.scatter(r, b, s=2)
plt.show()

plt.scatter(g, b, s=2)
plt.show()
