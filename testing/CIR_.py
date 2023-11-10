import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.linalg import eig
import torch
import geoopt
from geoopt.optim import RiemannianSGD


def CIR(X, Y, Xt, Yt, a, d):
    X = pd.DataFrame(X)
    Xt = pd.DataFrame(Xt)
    Y = pd.DataFrame(Y)
    Yt = pd.DataFrame(Yt)

    # check whether the first column is sequential numbers
    if X.iloc[:, 0].equals(pd.Series(range(1, len(X) + 1))):
        X = X.iloc[:, 1:]

    if Xt.iloc[:, 0].equals(pd.Series(range(1, len(Xt) + 1))):
        Xt = Xt.iloc[:, 1:]

    if Y.iloc[:, 0].equals(pd.Series(range(1, len(Y) + 1))):
        Y = Y.iloc[:, 1:]

    if Yt.iloc[:, 0].equals(pd.Series(range(1, len(Yt) + 1))):
        Yt = Yt.iloc[:, 1:]

    # n represents the rows of X. p represents the columns of X. m represent the rows of Xt.
    n = len(X)
    p = len(X.columns)
    m = len(Xt)

    # Parameter Check
    if len(Xt.columns) != p:
        raise ValueError('Xt should have the same number of columns as X')

    if len(Y) != n:
        raise ValueError('Y must have the same number of rows as X')

    if len(Yt) != m:
        raise ValueError('Yt must have the same number of rows as Xt')

    if not isinstance(d, int):
        raise TypeError('d parameter must be an integer')

    if d < 1:
        raise ValueError('d must be greater than or equal to 1')

    if a < 0:
        raise ValueError('a must be greater than or equal to 0')

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
    sigma_X = sigma_X + (5*10 ** (-14))*np.identity(p)
    # eigenvalues, eigenvectors = np.linalg.eigh(sigma_X)
    # print(eigenvalues)
    cov_matrix_X = np.array(cov_matrix_X)

    # Generalized Eigenvalue Decomposition
    A = cov_matrix_X @ sigma_X @ cov_matrix_X
    B = cov_matrix_X @ cov_matrix_X
    eigenvalues, eigenvectors = eigh(cov_matrix_X, sigma_X)

    if a == 0:
        v = eigenvectors[:d, :]
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
    cov_matrix_Xt = np.array(cov_matrix_Xt)

    # Generalized Eigenvalue Decomposition
    At = cov_matrix_Xt @ sigma_Xt @ cov_matrix_Xt
    Bt = cov_matrix_Xt @ cov_matrix_Xt

    # Optimization on Manifold
    stiefel = geoopt.manifolds.Stiefel()
    initial_guess = torch.randn(p, d)
    initial_guess, _ = torch.linalg.qr(initial_guess)

    v = geoopt.ManifoldParameter(initial_guess, manifold=stiefel)
    optimizer = RiemannianSGD([v], lr=0.1)
    v = v.to(torch.double)

    for step in range(100000):
        optimizer.zero_grad()
        gradient = grad(A, B, a, v, At, Bt)
        proj_grad = stiefel.proju(v, gradient)
        v.grad = proj_grad
        # v.manifold.proju(v, gradient)
        cost = f(A, B, a, v, At, Bt)
        optimizer.step()

    print(v @ v.t())

    # print(v.shape)


#     return v


def f(A, B, a, v, At, Bt):
    A = torch.tensor(A)
    B = torch.tensor(B)
    At = torch.tensor(At)
    Bt = torch.tensor(Bt)

    bv_inv = torch.inverse(torch.matmul(torch.matmul(v.t(), B), v))
    va = torch.matmul(torch.matmul(v.T, A), v)
    bv_t_inv = torch.inverse(torch.matmul(torch.matmul(v.t(), Bt), v))
    va_t = torch.matmul(torch.matmul(v.T, At), v)

    f_v = -torch.trace(torch.matmul(va, bv_inv)) + a * \
        torch.trace(torch.matmul(va_t, bv_t_inv))

    return f_v


def grad(A, B, a, v, At, Bt):
    A = torch.tensor(A)
    B = torch.tensor(B)
    At = torch.tensor(At)
    Bt = torch.tensor(Bt)

    v = v.to(torch.double)
    A = A.to(torch.double)
    B = B.to(torch.double)
    At = At.to(torch.double)
    Bt = Bt.to(torch.double)

    bv_inv = torch.inverse(v.t() @ B @ v)
    va = v.t() @ A @ v
    bv_t_inv = torch.inverse(v.t() @ Bt @ v)
    va_t = v.t() @ At @ v

    avb = A @ v @ bv_inv - B @ v @ bv_inv @ va @ bv_inv
    avb_t = At @ v @ bv_t_inv - Bt @ v @ bv_t_inv @ va_t @ bv_t_inv

    gradient = -2 * (avb - a * avb_t)

    return gradient
