import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.linalg import eig
from numpy.linalg import norm
import torch
import geoopt
from geoopt.optim import RiemannianSGD
from geoopt.manifolds import Stiefel


def CIR(X, Y, Xt, Yt, a, d):
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
    # print(eigenvectors)

    if a == 0:
        v = eigenvectors[:, :d]
        f_v = -1 * (np.trace(v.T @ A @ v @ np.linalg.inv(v.T @ B @ v)))
        return v, f_v
    # =================================================================
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

    A = torch.tensor(A, dtype=torch.double)
    B = torch.tensor(B, dtype=torch.double)
    a = torch.tensor(a, dtype=torch.double)
    At = torch.tensor(At, dtype=torch.double)
    Bt = torch.tensor(Bt, dtype=torch.double)

    # Optimization on Manifold
    stiefel = geoopt.manifolds.Stiefel()
    torch.manual_seed(2)
    initial_point = torch.randn(p, d)
    initial_point, _ = torch.linalg.qr(initial_point)

    v = geoopt.ManifoldParameter(initial_point, manifold=stiefel)

    optimizer = RiemannianSGD([v], lr=1e-3, momentum=0.9)
    max_iterations = 10000

    for step in range(max_iterations):
        vt = v.clone()
        optimizer.zero_grad()
        cost = f(A, B, a, v, At, Bt)
        gradient = grad(A, B, a, v, At, Bt).to(torch.float)
        v.grad = gradient
        optimizer.step()
        vt_plus = v.clone()

        if stepExit(vt_plus, vt, cost, A, B, At, Bt, a):
            break

    output = v @ v.t()
    return output


def f(A, B, a, v, At, Bt):
    v = v.data
    v = v.to(torch.double)

    bv_inv = torch.inverse(v.t() @ B @ v)
    va = v.t() @ A @ v
    bv_t_inv = torch.inverse(v.t() @ Bt @ v)
    va_t = v.t() @ At @ v

    f_v = -torch.trace(va @ bv_inv) + a * torch.trace(va_t @ bv_t_inv)
    return f_v


def grad(A, B, a, v, At, Bt):
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
    xtol = 1e-6  # 0.025
    gtol = 1e-4
    ftol = 1e-12
    distance = torch.norm(vt_plus @ vt_plus.t() - vt @ vt.t(), "fro")
    vt_plus_gradient = grad(A, B, a, vt_plus, At, Bt)
    cost_vt_plus = f(A, B, a, vt_plus, At, Bt)

    if distance < xtol:
        return True
    elif torch.norm(vt_plus_gradient, "fro") < gtol:
        return True
    elif abs(cost_vt_plus - cost) < ftol:
        return True
    else:
        return False
