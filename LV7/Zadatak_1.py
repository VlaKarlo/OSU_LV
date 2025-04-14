import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering


def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

# generiranje podatkovnih primjera
X = generate_data(500, 1)

# prikazi primjere u obliku dijagrama rasprsenja
plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri')
plt.show()

# ---------------------------------------- A ----------------------------------------

# Moze se pretpostaviti kako postoji 3 grupe

# X = generate_data(500, 1)
# X = generate_data(500, 2)
# X = generate_data(500, 3)
# X = generate_data(500, 4)
# X = generate_data(500, 5)

# plt.figure()
# plt.scatter(X[:,0],X[:,1])
# plt.xlabel('$x_1$')
# plt.ylabel('$x_2$')
# plt.title('podatkovni primjeri')
# plt.show()

# ---------------------------------------- B ----------------------------------------

fig, axs = plt.subplots(2, 3)

model= KMeans(n_clusters=3)
model.fit(X)
labels=model.predict(X)
axs[0, 0].scatter(X[:, 0], X[:, 1], c=labels)
axs[0, 0].set_xlabel('$x_1$')
axs[0, 0].set_ylabel('$x_2$')
axs[0, 0].set_title('Blobs\nK = 3')

X = generate_data(500, 2)

model.fit(X)
labels=model.predict(X)
axs[0, 1].scatter(X[:,0], X[:,1], c=labels)
axs[0, 1].set_xlabel('$x_1$')
axs[0, 1].set_ylabel('$x_2$')
axs[0, 1].set_title('Blobs\nK = 3')

X = generate_data(500, 3)

model= KMeans(n_clusters=4)
model.fit(X)
labels=model.predict(X)
axs[0, 2].scatter(X[:,0], X[:,1], c=labels)
axs[0, 2].set_xlabel('$x_1$')
axs[0, 2].set_ylabel('$x_2$')
axs[0, 2].set_title('Blobs\nK = 4')

X = generate_data(500, 4)

model= KMeans(n_clusters=2)
model.fit(X)
labels=model.predict(X)
axs[1, 0].scatter(X[:,0], X[:,1], c=labels)
axs[1, 0].set_xlabel('$x_1$')
axs[1, 0].set_ylabel('$x_2$')
axs[1, 0].set_title('Circles\nK = 2')

X = generate_data(500, 5)

model.fit(X)
labels=model.predict(X)
axs[1, 1].scatter(X[:,0], X[:,1], c=labels)
axs[1, 1].set_xlabel('$x_1$')
axs[1, 1].set_ylabel('$x_2$')
axs[1, 1].set_title('Moons\nK = 2')
plt.tight_layout()
plt.show()

# U druga 2 primjera moze bolje klasificirati
# Pogresno klasificira posljednja 2 primjera jer je takav algoritam