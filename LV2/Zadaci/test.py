import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("road.jpg")
img = img [ : ,: ,0 ].copy()

# -------------------- d --------------------

rotatedImage = np.flip(img,1)

plt.figure()
plt.imshow(rotatedImage, cmap = "gray")
plt.show()