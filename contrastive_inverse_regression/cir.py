import numpy as np
import pandas as pd
import math
import time
import scipy
from scipy.linalg import eigh
from scipy.linalg import solve
from scipy.linalg import cholesky
import sys


__all__ = ['ContrastiveInverseRegression']


def CIR(X, Y, Xt, Yt, alpha, d, n_sliceY=10, n_sliceYt=10, continuous_Y=False, continuous_Yt=False):
    """Contrastive inversse regression for dimension reduction (CIR) [1]

    Under a supervised contrastive dimension reduction setting, the algorithm
    finds a low-dimensional representation of x, denoted by V, by solving the 
    nonconvex optimization problem on the Stiefel manifold, associated with the
    input foreground and background matrix.

    Both the sample covariance matrix of foreground matrix X and the covariance 
    of X with respect to its corresponding output Y, are obtained to construct 
    matrix A and B. A similar procedure for the background matrix Xt and its 
    corresponding output Yt to obtain matrix At and Bt. 

    The contrastive loss function f and gradient of f is provided as input for 
    the gradient-based minimization algorithm on the Stiefel manifold. 

    In the case when alpha = 0, the loss function f is reduced to only pertaining
    to the foreground matrix, becoming a generalized eigenproblem. 

    Parameters
    ----------
    X : array-like, shape(n, p)
        Foreground data, where n is the number of
        observations and p is the number of features. 

    Y : array-like, shape(n, 1)
        Foreground response 1-dimensional array, where n
        is the number of observations.

    Xt : array-like, shape(m, k)
        Backgroundd data, where m is the number of
        observations and k is the number of features.

    Yt : array-like, shape(m, 1)
        Background response 1-dimensional array, 
        where s is the number of observations.

    alpha : float value
        Hyperparameter for importance of background
        group in contrastive loss function.

    d : integer value
        The number of reduced dimension. 

    n_sliceY : int (default = 10)
        The number of slices to split foreground Y if it is continuous, used for
        calculating the covariance of X with respect to Y. For the case if Y is
        discrete, the number of slices equals to the number of unique values in Y.

    n_sliceYt : int (default = 10)
        The number of slices to split background Yt if it is continuous, used for 
        calculating the covariance of Xt with respect to Yt. For the case if Yt is
        discrete, the number of slices equals to the number of unique values in Yt.

    continuous_Y : boolean (default = False)
        Determine whether the foreground Y array is continuous or discrete.

    continuous_Yt : boolean (default = False)
        Determine whether the background Yt array is continuous or discrete.

    Returns
    -------
    v : array-like
        Solution from minimization over the Stiefel Manifold.


    Reference
    -----------
    [1] Hawke, S., Luo, H., & Li, D. (2023)
        "Contrastive Inverse Regression for Dimension Reduction",
        Retrieved from https://arxiv.org/abs/2305.12287 

    [2] Harry Oviedo (2024).
       SGPM for minimization over the Stiefel Manifold (https://www.mathworks.com
       /matlabcentral/fileexchange/73505-sgpm-for-minimization-over-the-stiefel-manifold), 
       MATLAB Central File Exchange. Retrieved January 12, 2024.

    """
    X = np.array(X)
    Xt = np.array(Xt)
    Y = np.array(Y)
    Yt = np.array(Yt)

    n, p = X.shape
    m, k = Xt.shape

    if np.array_equal(X[:, 0], np.arange(1, len(X) + 1)):
        raise ValueError("X should not have an index column")

    if np.array_equal(Xt[:, 0], np.arange(1, len(Xt) + 1)):
        raise ValueError("Xt should not have an index column")

    if np.array_equal(Y[:], np.arange(1, len(Y) + 1)):
        raise ValueError("Y should not have an index column")

    if np.array_equal(Yt[:], np.arange(1, len(Yt) + 1)):
        raise ValueError("Yt should not have an index column")

    if k != p:
        raise ValueError("Xt should have the same number of columns as X")

    if len(Y) != n:
        raise ValueError("Y should have the same number of rows as X")

    if len(Yt) != m:
        raise ValueError("Yt should have the same number of rows as Xt")

    if not isinstance(d, int):
        raise TypeError("d parameter should be an integer")

    if d < 1:
        raise ValueError("d must be greater than or equal to 1")

    if alpha < 0:
        raise ValueError("a must be greater than or equal to 0")

    # Center the matrix X by subtracting the original matrix X by the column means of X
    X = X - np.mean(X, axis=0)

    if alpha == 0:
        Sigma_XX = X.T @ X / n
        Sigma_X = np.zeros((p, p))

        for l in range(1, L):
            X_curr = fg.values[Y == l]
            n_curr = X_curr.shape[0]
            Sigma_X += n_curr * np.outer(X_curr.mean(axis=0) - fg.values.mean(
                axis=0), X_curr.mean(axis=0) - fg.values.mean(axis=0))

        Sigma_X /= n
        eigvals, eigvecs = scipy.linalg.eig(Sigma_XX, Sigma_X)
        V_SIR = eigvecs[:, :d]

        f_v = -1 * (np.trace(v.T @ A @ v @ np.linalg.inv(v.T @ B @ v)))

        print('---------------------------------------------------\n')
        print('Results for SIR (when alpha = 0) \n')
        print('---------------------------------------------------\n')
        print('   Obj. function = %7.6e\n' % f_v)
        return V_SIR

    # Covariance matrix of foreground X
    cov_X = X.T @ X / n

    # Define H, which represents the # of intervals that splits range(Y)
    Y_unique = np.unique(Y)
    Y_unique_len = len(Y_unique)          # num of unique values in Y

    if continuous_Y:
        H = n_sliceY
    else:
        H = Y_unique_len

    # Cov(E[X|Y])
    sigma_X = np.zeros((p, p))
    for l in range(H):
        idx = np.where(Y == Y_unique[l])[0]       # index where Y occur
        X_select = X[idx, :]            # selected corresponding X rows
        num_r = X_select.shape[0]       # num of rows in the selected X
        outer_product = np.outer(np.mean(X_select, axis=0) - np.mean(
            X, axis=0), np.mean(X_select, axis=0) - np.mean(X, axis=0)) * num_r
        sigma_X += outer_product
    sigma_X = sigma_X/n

    A = cov_X @ sigma_X @ cov_X
    B = cov_X @ cov_X

    # --------The following is for background data and the caase when a > 0-------
    # Center the data
    Xt = Xt - np.mean(Xt, axis=0)

    # Covariance matrix of background Xt
    cov_Xt = Xt.T @ Xt / m

    # Define Ht, which represents the # of interval I that splits range(Yt)
    Yt_unique = np.unique(Yt)
    Yt_unique_length = len(Yt_unique)  # num of unique values in Yt

    if continuous_Yt:
        Ht = n_sliceYt
    else:
        Ht = Yt_unique_length

    sigma_Xt = np.zeros((k, k))
    for l in range(Ht):
        idx = np.where(Yt == Yt_unique[l])[0]       # index where Y occur
        Xt_select = Xt[idx, :]            # selected corresponding X rows
        num_r = Xt_select.shape[0]       # num of rows in the selected X
        outer_product = np.outer(np.mean(Xt_select, axis=0) - np.mean(
            Xt, axis=0), np.mean(Xt_select, axis=0) - np.mean(Xt, axis=0)) * num_r
        sigma_Xt += outer_product

    sigma_Xt = sigma_Xt/m

    At = cov_Xt @ sigma_Xt @ cov_Xt
    Bt = cov_Xt @ cov_Xt

    # Use SGPM (Scaled Gradient Projection Method for Minimization over the Stiefel Manifold)
    v = np.random.rand(p, d)
    v, r = np.linalg.qr(v)
    # v = np.eye(p)
    # v = v[:, :d]
    v = SGPM(v, A, B, At, Bt, alpha)
    return v


def f(A, B, alpha, v, At, Bt):
    f_v = -np.trace(v.T @ A @ v @ scipy.linalg.inv(v.T @ B @ v)) + \
        alpha * np.trace(v.T @ At @ v @ scipy.linalg.inv(v.T @ Bt @ v))
    return f_v


def grad(A, B, alpha, v, At, Bt):
    G = -2 * (A @ v @ scipy.linalg.inv(v.T @ B @ v)) + 2 * (B @ v @ scipy.linalg.inv(v.T @ B @ v) @ v.T @ A @ v @ scipy.linalg.inv(v.T @ B @ v)) + 2 * \
        alpha * (At @ v @ scipy.linalg.inv(v.T @ Bt @ v)) - 2 * alpha * (Bt @ v @
                                                                         scipy.linalg.inv(v.T @ Bt @ v) @ v.T @ At @ v @ scipy.linalg.inv(v.T @ Bt @ v))
    return G


def SGPM(X, A, B, At, Bt, a):
    X = np.array(X)
    n, k = X.shape

    xtol = 1e-20
    gtol = 1e-5
    ftol = 1e-20
    rho = 1e-4
    eta = 0.2
    gamma = 0.85
    tau = 1e-3
    STPEPS = 1e-10
    nt = 5
    mxitr = 3000
    alpha = 0.85
    record = 0
    projG = 1
    iscomplex = 0
    crit = np.ones((nt, 3))
    invH = True

    if k < n/2:
        invH = False
        eye2k = np.eye(2 * k)

    # Initial function value and gradient
    # Prepare for iterations
    F = f(A, B, a, X, At, Bt)
    G = grad(A, B, a, X, At, Bt)
    np.set_printoptions(precision=15, suppress=True, threshold=sys.maxsize)

    nfe = 1
    GX = G.T @ X

    if invH:
        GXT = G @ X.T
        H = (GXT - GXT.T)
    else:
        if projG == 1:
            U = np.hstack((G, X))
            V = np.hstack((X, -G))
            VU = V.T @ U
        elif projG == 2:
            GB = G - 0.5 * X @ (X.T @ G)
            U = np.hstack((GB, X))
            V = np.hstack((X, -GB))
            VU = V.T @ U

        VX = V.T @ X

    dtX = G - X @ GX
    nrmG = np.linalg.norm(dtX, ord='fro')
    Q = 1
    Cval = F

    if record == 1:
        fid = 1
        print(fid, '----------- Scaled Gradient Projection Method with Line search ----------- \n')
        print(fid, '%4s %8s %8s %10s %10s\n ',
              'Iter', 'tau', 'F(X)', 'nrmG', 'XDiff')

    # main iteration
    F_eval = np.zeros((mxitr + 2, 1))
    Grad = np.zeros((mxitr + 2, 1))
    F_eval[0] = F
    Grad[0] = nrmG

    start_time = time.time()
    for itr in np.arange(1, mxitr+1).reshape(-1):
        XP = X
        FP = F
        dtXP = dtX
        nrmGP = nrmG

        # scale step size
        nls = 1
        deriv = rho * (nrmG**2)

        while True:
            # Update Scheme
            if invH:
                if abs(alpha) < rho:    # Explicit Euler (Steepest Descent)
                    X = XP - tau * dtXP
                elif abs(alpha - 0.5) < rho:  # Crank-Nicolson
                    X = solve(np.eye(n) + (tau * 0.5) * H, XP -
                              (0.5 * tau) * dtXP, lower=False)
                elif abs(alpha - 1) < rho:  # Implicit EuLer
                    X = solve(np.eye(n) + tau * H, XP, lower=False)
                else:  # Convex Combination
                    X = solve(np.eye(n) + (tau * alpha) * H, XP -
                              ((1 - alpha) * tau) * dtXP, lower=False)

                if abs(alpha - 0.5) > rho:
                    XtX = np.transpose(X) @ X
                    L = cholesky(XtX, lower=False)
                    X = X @ np.linalg.inv(L)

            else:
                aa = solve(eye2k + (alpha * tau) * VU, VX)
                X = XP - U @ (tau * aa)

                if abs(alpha - 0.5) > rho:
                    XtX = X.T @ X
                    L = cholesky(XtX, lower=False)
                    X = X @ np.linalg.inv(L)

            # calculate G, F
            F = f(A, B, a, X, At, Bt)
            G = grad(A, B, a, X, At, Bt)

            nfe = nfe + 1

            if F <= Cval - tau * deriv or nls >= 5:
                break

            tau = eta * tau
            nls = nls + 1

        GX = G.T @ X
        dtX = G - X @ GX
        nrmG = np.linalg.norm(dtX, ord='fro')

        F_eval[itr+1] = F
        Grad[itr+1] = nrmG

        # Adaptive scaling matrix strategy
        if nrmG < nrmGP:
            if nrmG >= 0.5 * nrmGP:
                alpha = max(min(alpha * 1.1, 1), 0)
        else:
            alpha = max(min(alpha * 0.9, 0), 0.5)

        # Computing the Riemannian Gradient
        if invH:
            if abs(alpha) > rho:
                GXT = G @ X.T
                H = GXT - GXT.T
        else:
            if projG == 1:
                U = np.hstack((G, X))
                V = np.hstack((X, -G))
                VU = V.T @ U
            elif projG == 2:
                GB = G - X @ (0.5 * GX.T)
                U = np.hstack((GB, X))
                V = np.hstack((X, -GB))
                VU = V.T @ U
            VX = V.T @ X

        # Compute the Alternate ODH step-size
        S = X - XP
        SS = np.sum(np.sum(np.multiply(S, S), axis=1), axis=0)

        XDiff = math.sqrt(SS * (1/n))
        FDiff = abs(FP - F) / (abs(FP) + 1)

        Y = dtX - dtXP
        SY = abs(np.sum(np.sum(np.multiply(S, Y), axis=1), axis=0))

        if itr % 2 == 0:
            tau = SS / SY
        else:
            YY = np.sum(np.sum(np.multiply(Y, Y), axis=1), axis=0)
            tau = SY / YY

        tau = max(min(tau, 1e20), 1e-20)

        # Stopping Rules
        crit[itr % nt, :] = [nrmG, XDiff, FDiff]
        mcrit = np.mean(crit[max(0, itr-nt+1):itr, :], axis=0)

        if (XDiff < xtol and FDiff < ftol) or (nrmG < gtol) or all(mcrit[1:3] < 10 * np.hstack((xtol, ftol))):
            if itr <= 2:
                ftol = 0.1 * ftol
                xtol = 0.1 * xtol
                gtol = 0.1 * gtol
            else:
                msg = "converge"
                break

        Qp = Q
        Q = gamma * Qp + 1
        Cval = (gamma * Qp * Cval + F) / Q

    end_time = time.time()

    F_eval = F_eval[0:itr, 0]
    Grad = Grad[0:itr, 0]

    if itr >= mxitr:
        msg = "exceed max iteration"

    feasi = np.linalg.norm(np.transpose(X) @ X - np.eye(k), 'fro')

    if feasi > 1e-13:
        L = np.linalg.cholesky(X.T @ X)
        X = X @ np.linalg.inv(L)
        F = f(A, B, a, X, At, Bt)
        G = grad(A, B, a, X, At, Bt)
        dtX = G - X @ GX
        nrmG = np.linalg.norm(dtX, 'fro')
        nfe = nfe + 1
        feasi = np.linalg.norm(X.T @ X - np.eye(k), 'fro')

    print('---------------------------------------------------\n')
    print('Results for Scaled Gradient Projection Method \n')
    print('---------------------------------------------------\n')
    print('   Obj. function = %7.6e\n' % F)
    print('   Gradient norm = %7.6e \n' % nrmG)
    print('   ||X^T*X-I||_F = %3.2e\n' %
          np.linalg.norm(X.T @ X - np.eye(k), 'fro'))
    print('   Iteration number = %d\n' % itr)
    print('   Cpu time (secs) = %3.4f\n' % (end_time - start_time))
    print('   Number of evaluation(Obj. func) = %d\n' % nfe)

    return X
