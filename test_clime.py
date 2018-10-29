import clime
from sklearn.datasets import make_sparse_spd_matrix
import numpy as np
import matplotlib.pyplot as plt

p = 10
l = 1
prec = make_sparse_spd_matrix(p, 0.7)
C = np.linalg.inv(prec)
X = np.random.multivariate_normal(np.zeros(p), C, 200)
S = np.cov(X, rowvar=False)
clime_precision = clime.solve(S, l)
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