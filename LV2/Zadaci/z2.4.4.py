import numpy as np
import matplotlib.pyplot as plt

whiteSquare = np.ones((50,50))
blackSquare = np.zeros((50,50))

bottom = np.hstack((whiteSquare,blackSquare))
top = np.hstack((blackSquare,whiteSquare))

checker = np.vstack((top,bottom))

print(checker)

plt.figure()
plt.imshow(checker,cmap="gray")
plt.show()