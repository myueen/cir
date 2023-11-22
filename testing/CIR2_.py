import numpy as np
import pandas as pd
import math
from scipy.linalg import eigh
from scipy.linalg import eig
from numpy.linalg import norm
from scipy.linalg import solve
from scipy.linalg import cholesky


def CIR2(X, Y, Xt, Yt, a, d):
    X = pd.DataFrame(X)
    Xt = pd.DataFrame(Xt)
    Y = pd.DataFrame(Y)
    Yt = pd.DataFrame(Yt)

    # n represents the rows of X. p represents the columns of X. m represent the rows of Xt.
    n = len(X)
    p = len(X.columns)
    m = len(Xt)

    if X.iloc[:, 0].equals(pd.Series(range(1, len(X) + 1))):
        raise ValueError("X should not have an index column")

    if Xt.iloc[:, 0].equals(pd.Series(range(1, len(Xt) + 1))):
        raise ValueError("Xt should not have an index column")

    if Y.iloc[:, 0].equals(pd.Series(range(1, len(Y) + 1))):
        raise ValueError("Y should not have an index column")

    if Yt.iloc[:, 0].equals(pd.Series(range(1, len(Yt) + 1))):
        raise ValueError("Yt should not hav an index column")

    if len(Xt.columns) != p:
        raise ValueError("Xt should have the same number of columns as X")

    if len(Y) != n:
        raise ValueError("Y must have the same number of rows as X")

    if len(Yt) != m:
        raise ValueError("Yt must have the same number of rows as Xt")

    if not isinstance(d, int):
        raise TypeError("d parameter must be an integer")

    if d < 1:
        raise ValueError("d must be greater than or equal to 1")

    if a < 0:
        raise ValueError("a must be greater than or equal to 0")

    # Center the matrix X by subtracting the original matrix X by the column means of X
    col_mean_x = X.mean(axis=0)
    X = X - col_mean_x
    # Calculate the Covariance Matrix
    cov_matrix_X = X.cov()

    # Define H
    Y_unique_value = Y.nunique().item()
    if Y_unique_value == 2:
        H = 2
    elif 2 < Y_unique_value <= 10:
        H = Y_unique_value
    else:
        if d <= 2:
            H = 10
        else:
            H = 4

    # Define Ph, count the of ocurrence of y in each H interval.
    interval_Ph = pd.cut(Y.iloc[:, 0], bins=H)
    Ph = interval_Ph.value_counts().sort_index()

    # Find the mean: Take each row of X and average among each interval separately
    interval = np.linspace(np.min(Y), np.max(Y), num=H+1)
    mh = []
    for i in range(len(interval) - 1):
        mask = (interval[i] <= Y) & (Y <= interval[i+1])
        mask = mask.squeeze()  # convert to Series
        x_rows_mean = np.mean(X[mask.values], axis=0)
        mh.append(x_rows_mean)

    mh = np.array(mh)

    # Cov(E[X|Y])
    sigma_X = np.zeros((X.shape[1], X.shape[1]))
    for i in range(len(interval) - 1):
        mask = (interval[i] <= Y) & (Y <= interval[i+1])
        interval_mh = mh[i]
        outer_product = np.outer(interval_mh, interval_mh)

        if not np.isnan(outer_product).any():
            sigma_X += outer_product

    sigma_X = np.array(sigma_X)
    eigenvalues, eigenvectors = np.linalg.eigh(sigma_X)
    epsilon = 2 * abs(np.min(eigenvalues))
    sigma_X = sigma_X + epsilon * np.identity(p)

    cov_matrix_X = np.array(cov_matrix_X)

    # Generalized Eigenvalue Decomposition
    A = cov_matrix_X @ sigma_X @ cov_matrix_X
    B = cov_matrix_X @ cov_matrix_X
    eigenvalues, eigenvectors = eigh(cov_matrix_X, sigma_X)

    if a == 0:
        v = eigenvectors[:, :d]
        f_v = -1 * (np.trace(v.T @ A @ v @ np.linalg.inv(v.T @ B @ v)))
        return v, f_v
    # =======================================================================================
    # Background data and the case when a > 0

    # Center the data
    col_means_xt = Xt.mean(axis=0)
    Xt = Xt - col_means_xt

    # Covariance Matrix
    cov_matrix_Xt = Xt.cov()

    # Define Ht
    Yt_unique_value = Yt.nunique().item()
    if Yt_unique_value == 2:
        Ht = 2
    elif 2 < Yt_unique_value <= 10:
        Ht = Yt_unique_value
    else:
        if d <= 2:
            Ht = 10
        else:
            Ht = 4

    # Define Ph_t
    interval_Ph_t = pd.cut(Yt.iloc[:, 0], bins=Ht)
    Ph_t = interval_Ph_t.value_counts().sort_index()

    # Find the mean: Take each row of X and average among each interval separately
    interval_t = np.linspace(np.min(Yt), np.max(Yt), num=Ht+1)
    mh_t = []
    for i in range(len(interval_t) - 1):
        mask_t = (interval_t[i] <= Yt) & (Yt <= interval_t[i+1])
        mask = mask.squeeze()
        xt_rows_mean = np.mean(Xt[mask_t.values], axis=0)
        mh_t.append(xt_rows_mean)

    mh_t = np.array(mh_t)

    # Cov(E[Xt|Yt])
    sigma_Xt = np.zeros((Xt.shape[1], Xt.shape[1]))
    for i in range(len(interval_t) - 1):
        mask_t = (interval_t[i] <= Yt) & (Yt <= interval_t[i+1])
        interval_mh_t = mh_t[i]
        outer_product_t = np.outer(interval_mh_t, interval_mh_t)

        if not np.isnan(outer_product_t).any():
            sigma_Xt += outer_product_t

    sigma_Xt = np.array(sigma_Xt)
    eigenvalues_t, eigenvectors_t = np.linalg.eigh(sigma_Xt)
    epsilon_t = 2 * abs(np.min(eigenvalues_t))
    sigma_Xt = sigma_Xt + epsilon_t * np.identity(p)

    cov_matrix_Xt = np.array(cov_matrix_Xt)

   # Generalized Eigenvalue Decomposition
    At = cov_matrix_Xt @ sigma_Xt @ cov_matrix_Xt
    Bt = cov_matrix_Xt @ cov_matrix_Xt

    np.random.seed(2)
    v = np.random.rand(p, d)
    v, r = np.linalg.qr(v)

    output = SGPM(v, A, B, At, Bt, a)
    # return output
    return output @ np.transpose(output)


def f(A, B, a, v, At, Bt):
    bv_inv = np.linalg.inv(np.transpose(v) @ B @ v)
    va = np.transpose(v) @ A @ v
    bv_t_inv = np.linalg.inv(np.transpose(v) @ Bt @ v)
    va_t = np.transpose(v) @ At @ v
    f_v = -np.matrix.trace(va @ bv_inv) + a * np.matrix.trace(va_t @ bv_t_inv)
    return f_v


def grad(A, B, a, v, At, Bt):
    bv_inv = np.linalg.inv(np.transpose(v) @ B @ v)
    va = np.transpose(v) @ A @ v
    bv_t_inv = np.linalg.inv(np.transpose(v) @ Bt @ v)
    va_t = np.transpose(v) @ At @ v

    avb = A @ v @ bv_inv - B @ v @ bv_inv @ va @ bv_inv
    avb_t = At @ v @ bv_t_inv - Bt @ v @ bv_t_inv @ va_t @ bv_t_inv

    gradient = -2 * (avb - a * avb_t)
    return gradient


def SGPM(X, A, B, At, Bt, a):
    X = np.array(X)
    n, k = X.shape

    xtol = 1e-6
    gtol = 1e-4
    ftol = 1e-12
    rho = 1e-4
    eta = 0.1
    gamma = 0.85
    tau = 1e-3
    STPEPS = 1e-10
    nt = 5
    mxitr = 1000
    alpha = 0.85
    record = 0
    projG = 2
    iscomplex = 0
    crit = np.ones((mxitr, 3))
    invH = True

    if k < n/2:
        invH = False
        eye2k = np.identity(2 * k)

    # Initial function value and gradient
    # Prepare for iterations
    F = f(A, B, a, X, At, Bt)
    G = grad(A, B, a, X, At, Bt)

    GX = np.transpose(G) @ X

    if invH:
        GXT = G @ np.transpose(X)
        H = GXT - np.transpose(GXT)
    else:
        if projG == 1:
            U = np.hstack((G, X))
            V = np.hstack((X, -G))
            VU = np.transpose(V) @ U
        elif projG == 2:
            GB = G - 0.5 * X @ (np.transpose(X) @ G)
            U = np.hstack((GB, X))
            V = np.hstack((X, -GB))
            VU = np.transpose(V) @ U
        VX = np.transpose(V) @ X

    dtX = G - X @ GX
    nrmG = np.linalg.norm(dtX, ord='fro')
    Q = 1
    Cval = F

    # main iteration
    F_eval = np.zeros((mxitr + 1, 1))
    Grad = np.zeros((mxitr + 1, 1))
    F_eval[0] = F
    Grad[0] = nrmG
    for itr in range(0, mxitr):
        XP, FP, dtXP, nrmGP = X, F, dtX, nrmG

        # scale step size
        nls = 1
        deriv = rho * (nrmG**2)

        while True:
            # Update Scheme
            if invH:
                if abs(alpha) < rho:    # Explicit Euler (Steepest Descent)
                    X = XP - tau * dtXP
                elif abs(alpha - 0.5) < rho:  # Crank-Nicolson
                    A = np.eye(n) + (tau * 0.5) * H
                    X = solve(
                        A, XP - (0.5 * tau) * dtXP, lower=False)
                elif abs(alpha - 1) < rho:  # Implicit EuLer
                    A = np.eye(n) + tau * H
                    X = solve(A, XP, lower=False)
                else:  # Convex Combination
                    A = np.eye(n) + (tau * alpha) * H
                    X = solve(
                        A, XP - ((1 - alpha) * tau) * dtXP, lower=False)

                if abs(alpha - 0.5) > rho:
                    XtX = np.transpose(X) @ X
                    L = cholesky(XtX, lower=False)
                    X = X @ np.linalg.inv(L)

            else:
                aa = solve(eye2k + (alpha * tau) * VU, VX)
                X = XP - U @ (tau * aa)

                if abs(alpha - 0.5) > rho:
                    XtX = np.transpose(X) @ X
                    L = cholesky(XtX, lower=False)
                    X = X @ np.linalg.inv(L)

            # calculate G, F
            F = f(A, B, a, X, At, Bt)
            G = grad(A, B, a, X, At, Bt)

            if F <= Cval - tau * deriv or nls >= 5:
                break

            tau = eta * tau
            nls += 1

        GX = np.transpose(G) @ X
        dtX = G - X @ GX
        nrmG = np.linalg.norm(dtX, ord='fro')

        F_eval[itr+1] = F
        # print(Grad)
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
                GXT = G @ np.transpose(X)
                H = GXT - np.transpose(GXT)
        else:
            if projG == 1:
                U = np.hstack((G, X))
                V = np.hstack((X, -G))
                VU = np.transpose(V) @ U
            elif projG == 2:
                GB = G - X @ (0.5 * np.transpose(GX))
                U = np.hstack((GB, X))
                V = np.hstack((X, -GB))
                VU = np.transpose(V) @ U
            VX = np.transpose(V) @ X

        # Compute the Alternate ODH step-size
        S = X - XP
        SS = np.sum(np.sum(np.multiply(S, S), axis=1), axis=0)

        XDiff = math.sqrt(SS / n)
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
        crit[itr, :] = [nrmG, XDiff, FDiff]

        mcrit = np.mean(crit[max(0, itr-nt+1):itr, :], axis=0)
        # mcrit = np.mean(crit[itr - min(nt, itr) + 1:itr, :], axis=0)
        # print("Iteration:", itr)
        # print("size of crit", crit.shape)
        # print("size of mcrit", mcrit.shape)

        if (XDiff < xtol and FDiff < ftol) or nrmG < gtol or all(mcrit[1:3] < 10 * np.hstack((xtol, ftol))):
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
        feasi = np.linalg.norm(X.T @ X - np.eye(k), 'fro')

    feasi = np.linalg.norm(X.T @ X - np.eye(k), 'fro')
    return X
