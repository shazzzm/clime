import clime
from sklearn.datasets import make_sparse_spd_matrix
import numpy as np
import matplotlib.pyplot as plt

p = 5
prec = make_sparse_spd_matrix(p, 0.7)
C = np.linalg.inv(prec)
X = np.random.multivariate_normal(np.zeros(p), C, 100)
S = np.cov(X, rowvar=False)
clime_precision, l = clime.cross_validation(X)
clime_precision[np.abs(clime_precision) < 0.0001] = 0
plt.figure()
plt.title("Empirical Precision")
plt.imshow(np.linalg.inv(S), cmap='hot', interpolation='nearest')

plt.figure()
plt.title("True Precision")
plt.imshow(prec, cmap='hot', interpolation='nearest')

plt.figure()
plt.title("CLIME Precision")
plt.imshow(clime_precision, cmap='hot', interpolation='nearest')

plt.show()