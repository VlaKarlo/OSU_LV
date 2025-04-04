import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, confusion_matrix


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# ---------------------------------------- A ----------------------------------------

plt.figure()

plt.scatter(
    X_train[:,0],
    X_train[:,1],
    c = y_train,
    cmap = 'magma',
    marker = 'o',
    label = 'Trening skup',
    alpha = 0.4
)

plt.scatter(
    X_test[:,0],
    X_test[:,1],
    c = y_test,
    cmap = 'ocean_r',
    marker = 'x',
    label = 'Testni skup',
    alpha = 0.4
)

plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()

# ---------------------------------------- B ----------------------------------------

logisticRegression = LogisticRegression()
logisticRegression.fit(X_train, y_train)

# ---------------------------------------- C ----------------------------------------

θ1, θ2 = logisticRegression.coef_[0]  
θ0 = logisticRegression.intercept_[0] 

plt.scatter(
    X_train[:, 0], 
    X_train[:, 1], 
    c=y_train, 
    cmap='ocean_r', 
    marker='o', 
    label='Trening skup', 
    alpha=0.4
    )

plt.scatter(
    X_test[:, 0], 
    X_test[:, 1], 
    c=y_test, 
    cmap='magma', 
    marker='x', 
    label='Testni skup', 
    alpha=0.4
    )

x = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)

plt.plot(
    x, 
    (- θ0 - θ1 * x) / θ2, 
    color='black', 
    label='Granica odluke'
    )

plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()

# ---------------------------------------- D ----------------------------------------

y_pred = logisticRegression.predict(X_test)  

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logisticRegression.classes_).plot(cmap='magma')
plt.show()

accuracy = accuracy_score( y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# ---------------------------------------- E ----------------------------------------

correct = y_test == y_pred
incorrect = ~correct

plt.scatter(
    X_test[correct, 0], 
    X_test[correct, 1], 
    c='b', 
    label='Correct', 
    marker='o'
    )

plt.scatter(
    X_test[incorrect, 0], 
    X_test[incorrect, 1], 
    c='r', 
    label='Incorrect', 
    marker='x'
    )

plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()