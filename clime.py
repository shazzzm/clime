import numpy as np
import cvxpy as cvx

def solve(S, l):
    """
    Solves the CLIME problem for empirical covariance matrix S
    """
    p = S.shape[0]
    theta = np.zeros((p, p))
    for i in range(p):
        beta = cvx.Variable(p)
        objective = cvx.Minimize(cvx.norm(beta, 1))
        e = np.zeros(p)
        e[i] = 1
        constraints = [cvx.norm(S @ beta - e, "inf") <= l]
        prob = cvx.Problem(objective, constraints)
        result = prob.solve()
        theta[i, :] = beta.value

    return _make_symmetric(theta)

def _make_symmetric(M):
    """
    Makes the matrix M symmetric
    """
    p = M.shape[0]

    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            x = M[i, j]
            y = M[j, i]

            if x < y:
                M[i, j] = x
                M[j, i] = x
            else:
                M[i, j] = y
                M[j, i] = y

    return M