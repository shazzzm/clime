import numpy as np
import cvxpy as cvx

def solve(S, l):
    """
    Solves the CLIME problem for empirical covariance matrix S
    """
    p = S.shape[0]
    theta = np.zeros((p, p))
    for i in range(p):
        cov_row = S[i, :]
        beta = cvx.Variable(p)
        objective = cvx.Minimize(cvx.norm(beta, 1))
        e = np.zeros(p)
        e[i] = 1
        constraints = [cvx.norm(S @ beta - e, "inf") <= l]
        prob = cvx.Problem(objective, constraints)
        result = prob.solve()
        theta[i, :] = beta.value

    return theta