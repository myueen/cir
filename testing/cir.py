import numpy as np
import pandas as pd
import math
import time
import sliced
from sliced import base
from sliced.base import slice_y, grouped_sum, unique_counts
from scipy.linalg import eigh, qr
from numpy.linalg import norm
import torch
import warnings
import geoopt
from geoopt.optim import RiemannianSGD
from geoopt.manifolds import Stiefel
from scipy.linalg import solve
from scipy.linalg import cholesky


def CIR(X, Y, Xt, Yt, alpha, d, opt_option):
    """Apply contrastive inverse regression dimension reduction on X.

    Parameters
    ----------
    X : array-like, shape(n, p)
        Foreground data, where n is the number of observations
        and p is the number of features 

    Y : array-like, shape(t, 1)
        Foreground response 1-dimensional array, where t is the number of observations 

    Xt : array-like, shape(m, k)
         Backgroundd data, where m is the number of observations 
         and k is the number of features 

    Yt : array-like, shape(s, 1)
         Background response 1-dimensional array, where s is the number of observations

    alpha : float value
            hyperparameter for importance of background group in contrastive loss function

    d : integer value
        reduced dimension 

    Returns
    -------


    ## Dependencies 
    This project relies on the following external packages: 
    - [sliced](github.com/joshloyal/sliced) by Joshya Loyal 

    """

    X = pd.DataFrame(X)
    Xt = pd.DataFrame(Xt)
    Y = pd.DataFrame(Y)
    Yt = pd.DataFrame(Yt)

    # n represents # of observation of foreground
    # p represents # of features of foreground
    # m represents # of observation of background
    # k represents # of features of background
    n, p = X.shape
    m, k = Xt.shape

    if X.iloc[:, 0].equals(pd.Series(range(1, len(X) + 1))):
        raise ValueError("X should not have an index column")

    if Xt.iloc[:, 0].equals(pd.Series(range(1, len(Xt) + 1))):
        raise ValueError("Xt should not have an index column")

    if Y.iloc[:, 0].equals(pd.Series(range(1, len(Y) + 1))):
        raise ValueError("Y should not have an index column")

    if Yt.iloc[:, 0].equals(pd.Series(range(1, len(Yt) + 1))):
        raise ValueError("Yt should not have an index column")

    if k != p:
        raise ValueError("Xt should have the same number of columns as X")

    if len(Y) != n:
        raise ValueError("Y should have the same number of rows as X")

    if len(Yt) != m:
        raise ValueError("Yt should have the same number of rows as Xt")

    if not isinstance(d, int):
        raise TypeError("d parameter must be an integer")

    if d < 1:
        raise ValueError("d must be greater than or equal to 1")

    if alpha < 0:
        raise ValueError("a must be greater than or equal to 0")

    # Center the matrix X by subtracting the original matrix X by the column means of X
    X = X - np.mean(X, axis=0)

    # Covariance matrix
    cov_X = X.cov()

    # Define H, which represents the # of intervals I that splits range(Y)
    Y_unique_value = Y.nunique().item()
    if Y_unique_value == 2:
        H = 2                       # number of slices
    elif 2 < Y_unique_value <= 10:
        H = Y_unique_value
    else:
        if d <= 2:
            H = 10
        else:
            H = 4

    # Count the of ocurrence of y in each H interval
    unique_y_vals_F, counts_F = unique_counts(Y)
    cumsum_y_F = np.cumsum(counts_F)
    n_y_values_F = unique_y_vals_F.shape[0]

    if H >= n_y_values_F:
        if H > n_y_values_F:
            warnings.warn(
                "n_slices greater than the number of unique y values.")

        slice_partition_F = np.hstack((0, cumsum_y_F))
    else:
        n_obs_F = np.floor(Y.shape[0] / H)
        n_samples_seen_F = 0
        slice_partition_F = [0]  # index in y of start of a new slice
        while n_samples_seen_F < Y.shape[0] - 2:
            slice_start_F = np.where(
                cumsum_y_F >= n_samples_seen_F + n_obs_F)[0]
            if slice_start_F.shape[0] == 0:  # this means we've reached the end
                slice_start_F = cumsum_y_F.shape[0] - 1
            else:
                slice_start_F = slice_start_F[0]

            n_samples_seen_F = cumsum_y_F[slice_start_F]
            slice_partition_F.append(n_samples_seen_F)

    slice_indicator_F = np.ones(Y.shape[0], dtype=int)

    for j, (start_idx, end_idx) in enumerate(zip(slice_partition_F, slice_partition_F[1:])):
        if j == len(slice_partition_F) - 2:
            slice_indicator_F[start_idx:] = j
        else:
            slice_indicator_F[start_idx:end_idx] = j

    slice_counts_F = np.bincount(slice_indicator_F)

    # Define Sigma X
    Q, R = qr(X, mode='economic')
    Z = np.sqrt(n) * Q
    Y = Y.to_numpy()
    sorted_indices = np.argsort(Y, axis=0, kind='stable')
    Z = Z[sorted_indices, :]
    Z = np.squeeze(Z)

    M_means = grouped_sum(Z, slice_indicator_F) / \
        np.sqrt(slice_counts_F.reshape(-1, 1))

    sigma_X = np.dot(M_means.T, M_means) / n

    eigenvalues, eigenvectors = np.linalg.eigh(sigma_X)
    epsilon = 2 * abs(np.min(eigenvalues))
    sigma_X = sigma_X + epsilon * np.identity(p)

    cov_X = np.array(cov_X)

    # Generalized Eigenvalue Decomposition
    A = cov_X @ sigma_X @ cov_X
    B = cov_X @ cov_X
    eigenvalues, eigenvectors = eigh(cov_X, sigma_X)

    if alpha == 0:
        v = eigenvectors[:, :d]
        f_v = -1 * (np.trace(v.T @ A @ v @ np.linalg.inv(v.T @ B @ v)))
        return v, f_v

    # --------The following is for background data and the caase when a > 0-------
    # Center the data
    Xt = Xt - np.mean(Xt, axis=0)

    # Covariance matrix
    cov_Xt = Xt.cov()

    # Define Ht, which represents the # of interval I that splits range(Yt)
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

    # Count the of ocurrence of y in each H intervaL
    unique_y_vals_B, counts_B = unique_counts(Yt)
    cumsum_y_B = np.cumsum(counts_B)
    n_y_values_B = unique_y_vals_B.shape[0]

    if Ht >= n_y_values_B:
        if Ht > n_y_values_B:
            warnings.warn(
                "n_slices greater than the number of unique y values.")

        slice_partition_B = np.hstack((0, cumsum_y_B))
    else:
        n_obs_B = np.floor(Yt.shape[0] / Ht)
        n_samples_seen_B = 0
        slice_partition_B = [0]  # index in y of start of a new slice
        while n_samples_seen_B < Yt.shape[0] - 2:
            slice_start_B = np.where(
                cumsum_y_B >= n_samples_seen_B + n_obs_B)[0]
            if slice_start_B.shape[0] == 0:  # this means we've reached the end
                slice_start_B = cumsum_y_B.shape[0] - 1
            else:
                slice_start_B = slice_start_B[0]

            n_samples_seen_B = cumsum_y_B[slice_start_B]
            slice_partition_B.append(n_samples_seen_B)

    slice_indicator_B = np.ones(Yt.shape[0], dtype=int)
    for j, (start_idx, end_idx) in enumerate(zip(slice_partition_B, slice_partition_B[1:])):
        if j == len(slice_partition_B) - 2:
            slice_indicator_B[start_idx:] = j
        else:
            slice_indicator_B[start_idx:end_idx] = j

    slice_counts_B = np.bincount(slice_indicator_B)

    # Define Sigma X
    Qt, Rt = qr(Xt, mode='economic')
    Zt = np.sqrt(m) * Qt
    Yt = Yt.to_numpy()
    sorted_indices_t = np.argsort(Yt, axis=0, kind='stable')
    Zt = Zt[sorted_indices_t, :]
    Zt = np.squeeze(Zt)
    Mt_means = grouped_sum(Zt, slice_indicator_B) / \
        np.sqrt(slice_counts_B.reshape(-1, 1))

    sigma_Xt = np.dot(Mt_means.T, Mt_means) / m

    cov_Xt = np.array(cov_Xt)

    At = cov_Xt @ sigma_Xt @ cov_Xt
    Bt = cov_Xt @ cov_Xt

    # Optimization Algorithm on Stiefel Manifold
    if opt_option == "geoopt":
        A = torch.tensor(A, dtype=torch.double)
        B = torch.tensor(B, dtype=torch.double)
        alpha = torch.tensor(alpha, dtype=torch.double)
        At = torch.tensor(At, dtype=torch.double)
        Bt = torch.tensor(Bt, dtype=torch.double)

        stiefel = geoopt.manifolds.Stiefel()
        torch.manual_seed(1)
        initial_point = torch.randn(p, d)
        initial_point, _ = torch.linalg.qr(initial_point)

        v = geoopt.ManifoldParameter(initial_point, manifold=stiefel)

        optimizer = RiemannianSGD([v], lr=1e-3, momentum=0.9)
        max_iterations = 50000

        for step in range(max_iterations):
            vt = v.clone()
            optimizer.zero_grad()
            cost = f_geoopt(A, B, alpha, v, At, Bt)
            gradient = grad_geoopt(A, B, alpha, v, At, Bt).to(torch.float)
            v.grad = gradient
            print(v.grad)
            optimizer.step()
            vt_plus = v.clone()

            if stepExit(vt_plus, vt, cost, A, B, At, Bt, alpha):
                break

        output = v @ v.t()
        return output

    # Use SGPM (Scaled Gradient Projection Method for Minimization over the Stiefel Manifold)
    if opt_option == "SGPM":
        np.random.seed(2)
        v = np.random.rand(p, d)
        v, r = np.linalg.qr(v)
        output = SGPM(v, A, B, At, Bt, alpha)
        # return output
        return output @ np.transpose(output)


def f_geoopt(A, B, a, v, At, Bt):
    v = v.data
    v = v.to(torch.double)

    bv_inv = torch.inverse(v.t() @ B @ v)
    va = v.t() @ A @ v
    bv_t_inv = torch.inverse(v.t() @ Bt @ v)
    va_t = v.t() @ At @ v

    f_v = -torch.trace(va @ bv_inv) + a * torch.trace(va_t @ bv_t_inv)
    return f_v


def grad_geoopt(A, B, a, v, At, Bt):
    v = v.data
    v = v.to(torch.double)

    bv_inv = torch.inverse(v.t() @ B @ v)
    va = v.t() @ A @ v
    bv_t_inv = torch.inverse(v.t() @ Bt @ v)
    va_t = v.t() @ At @ v

    avb = A @ v @ bv_inv - B @ v @ bv_inv @ va @ bv_inv
    avb_t = At @ v @ bv_t_inv - Bt @ v @ bv_t_inv @ va_t @ bv_t_inv

    gradient = -2 * (avb - a * avb_t)
    return gradient


def stepExit(vt_plus, vt, cost, A, B, At, Bt, a) -> bool:
    xtol = 1e-6
    gtol = 1e-4
    ftol = 1e-12
    distance = torch.norm(vt_plus @ vt_plus.t() - vt @ vt.t(), 'fro')
    vt_plus_gradient = grad_geoopt(A, B, a, vt_plus, At, Bt)
    cost_vt_plus = f_geoopt(A, B, a, vt_plus, At, Bt)

    if distance < xtol:
        return True
    elif torch.norm(vt_plus_gradient, 'fro') < gtol:
        return True
    elif abs(cost_vt_plus - cost) < ftol:
        return True
    else:
        return False


def f(A, B, alpha, v, At, Bt):
    bv = np.linalg.inv(np.transpose(v) @ B @ v)
    va = np.transpose(v) @ A @ v
    bv_t = np.linalg.inv(np.transpose(v) @ Bt @ v)
    va_t = np.transpose(v) @ At @ v

    f_v = -np.matrix.trace(va @ bv) + alpha * np.matrix.trace(va_t @ bv_t)
    return f_v


def grad(A, B, alpha, v, At, Bt):
    bv = np.linalg.inv(np.transpose(v) @ B @ v)
    va = np.transpose(v) @ A @ v
    bv_t = np.linalg.inv(np.transpose(v) @ Bt @ v)
    va_t = np.transpose(v) @ At @ v

    gradient = -2 * (A @ v @ bv - B @ v @ bv @ va @ bv -
                     alpha * (At @ v @ bv_t - Bt @ v @ bv_t @ va_t @ bv_t))
    return gradient


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
    nfe = 1

    GX = np.transpose(G) @ X

    if invH:
        GXT = G @ np.transpose(X)
        H = (GXT - np.transpose(GXT))
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

    if record == 1:
        fid = 1
        print(fid, '----------- Scaled Gradient Projection Method with Line search ----------- \n')
        print(fid, '%4s %8s %8s %10s %10s\n',
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
                    XtX = np.transpose(X) @ X
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

        GX = np.transpose(G) @ X
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
        feasi = np.linalg.norm(X.T @ X - np.eye(k), 'fro')

    print('---------------------------------------------------\n')
    print('Results for Scaled Gradient Projection Method \n')
    print('---------------------------------------------------\n')
    print('   Obj. function = %7.6e\n' % (-2*F))
    print('   Gradient norm = %7.6e \n' % nrmG)
    print('   ||X^T*X-I||_F = %3.2e\n' %
          np.linalg.norm(X.T @ X - np.eye(k), 'fro'))
    print('   Iteration number = %d\n' % itr)
    print('   Cpu time (secs) = %3.4f\n' % (end_time - start_time))
    print('   Number of evaluation(Obj. func) = %d\n' % nfe)
    return X
