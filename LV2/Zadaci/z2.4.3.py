import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("road.jpg")
img = img [ : ,: ,0 ] . copy ()

# -------------------- a --------------------

maxBrightness = 255
brighterIndex = 175

brightener = np.full(img.shape, brighterIndex, dtype="uint8")
brighterImage = img + brightener
brighterImage[brighterImage < brighterIndex] = maxBrightness

plt.figure()
plt.imshow(brighterImage, cmap = "gray")
plt.show()

# -------------------- b --------------------

croppedImage = img[:int(img.shape[0]/2), int(img.shape[1]/2):]

plt.figure()
plt.imshow(croppedImage, cmap = "gray")
plt.show()

# -------------------- c --------------------

rotatedImage = np.rot90(img,3)

plt.figure()
plt.imshow(rotatedImage, cmap = "gray")
plt.show()

# -------------------- d --------------------

rotatedImage = np.flip(img,1)

plt.figure()
plt.imshow(rotatedImage, cmap = "gray")
plt.show()