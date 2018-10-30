import numpy as np
import cvxpy as cvx
import sklearn.model_selection as skms

def solve(X, l, verbose=False):
    """
    Solves the CLIME problem for empirical covariance matrix S
    """
    S = np.cov(X, rowvar=False)
    n = X.shape[0]
    p = S.shape[0]
    theta = np.zeros((p, p))
    if p > n:
        rho = np.sqrt(np.log10(p/n))
        S = S + rho * np.eye(p)
    for i in range(p):
        beta = cvx.Variable(p)
        objective = cvx.Minimize(cvx.norm(beta, 1))
        e = np.zeros(p)
        e[i] = 1
        constraints = [cvx.norm(S @ beta - e, "inf") <= l]
        prob = cvx.Problem(objective, constraints)
        result = prob.solve(solver=cvx.ECOS, verbose=verbose)
        theta[i, :] = beta.value.flatten()

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

def _log_likelihood(S, theta):
    sgn, logdet = np.linalg.slogdet(theta)
    return np.trace(theta * S) - sgn * logdet

def cross_validation(X, max_l=0.8, min_l=None, num_lambdas=25, num_splits = 6):
    """
    Runs cross validation with CLIME to figure out the optimal lambda value
    """
    p = X.shape[1]
    n = X.shape[0]
    if min_l is None:
        if p > n:
            min_l = 0.2
        else:
            min_l = 0.4

    lambdas = np.logspace(np.log10(min_l), np.log10(max_l), num_lambdas)
    mean_likelihood = np.zeros(num_lambdas)
    for j,l in enumerate(lambdas):
        likelihoods = np.zeros(num_splits)
        for i in range(num_splits):
            X_train, X_test = skms.train_test_split(X, test_size=0.25)
            theta = solve(X_train, l)
            S_test = np.cov(X_test, rowvar=False)
            likelihoods[i] = _log_likelihood(S_test, theta)
        mean_likelihood[j] = np.mean(likelihoods)

    min_err_lambda = np.argmin(mean_likelihood)
    return solve(X, lambdas[min_err_lambda]), lambdas[min_err_lambda]


    

    
    