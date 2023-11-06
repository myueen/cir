import numpy as np
import pandas as pd
from scipy.linalg import eigh
import torch
import geoopt
from geoopt.optim import RiemannianSGD


def CIR(X, Y, Xt, Yt, a, d):
    n = len(X)
    p = len(X[0])
    m = len(Xt)

    # Parameter Check
    if len(Xt[0]) != p:
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

    X_df = pd.DataFrame(X)
    Xt_df = pd.DataFrame(Xt)
    Y_df = pd.DataFrame(Y)
    Yt_df = pd.DataFrame(Yt)

    # Center the data
    X_col_means = X_df.mean(axis=0)
    print(X_col_means)
    # X_centered = X_df - X_col_means

    # # Covariance Matrix
    # X_cov_matrix = X_centered.cov()

    # # Define H
    # Y_unique_value = Y_df.nunique().item()
    # if Y_unique_value == 2:
    #     H = 2
    # elif Y_unique_value > 2 & Y_unique_value <= 10:
    #     H = Y_unique_value
    # else:
    #     if d <= 2:
    #         H = 10
    #     else:
    #         H = 4

    # # Define Ph     (Count the # of ocurrence of y in each H interval)
    # interval_Ph = pd.cut(Y_df[0], bins=H)
    # Ph = interval_Ph.value_counts().sort_index()

    # # Find the mean: Take each row of X and average among each interval separately
    # interval = np.linspace(np.min(Y), np.max(Y), num=H+1)

    # mh = []
    # for i in range(len(interval) - 1):
    #     mask = (Y >= interval[i]) & (Y <= interval[i+1])
    #     x_rows_mean = np.mean(X_centered[mask], axis=0)
    #     mh.append(x_rows_mean)

    # mh_array = np.array(mh)

    # # Cov(E[X|Y])
    # sigma_X = np.zeros((X_centered.shape[1], X_centered.shape[1]))
    # for i in range(len(interval) - 1):
    #     mask = (Y >= interval[i]) & (Y <= interval[i+1])
    #     interval_mh = mh_array[i]
    #     outer_product = np.outer(interval_mh, interval_mh)

    #     if not np.isnan(outer_product).any():
    #         sigma_X += outer_product

    # sigma_X_a = np.array(sigma_X)
    # X_cov_matrix_a = np.array(X_cov_matrix)

    # # Generalized Eigenvalue Decomposition
    # A = X_cov_matrix_a @ sigma_X_a @ X_cov_matrix_a
    # B = X_cov_matrix_a @ X_cov_matrix_a
    # eigenvalues, eigenvectors = eigh(X_cov_matrix_a, sigma_X_a)

#     if a == 0:
#         v = eigenvectors[:d, :]
#         f_v = -1 * (np.trace(v.T @ A @ v @ np.linalg.inv(v.T @ B @ v)))
#         return v, f_v
#     # =======================================================================================
#     # Background data and the case when a > 0

#     # Center the data
#     Xt_col_means = Xt_df.mean(axis=0)
#     Xt_centered = Xt_df - Xt_col_means

#     # Covariance Matrix
#     Xt_cov_matrix = Xt_centered.cov()

#     # Define Ht
#     Yt_unique_value = Yt_df.nunique().item()
#     if Yt_unique_value == 2:
#         Ht = 2
#     elif Yt_unique_value > 2 & Yt_unique_value <= 10:
#         Ht = Yt_unique_value
#     else:
#         if d <= 2:
#             Ht = 10
#         else:
#             Ht = 4

#     # Define Ph_t
#     interval_Ph_t = pd.cut(Yt_df[0], bins=Ht)
#     Ph_t = interval_Ph_t.value_counts().sort_index()

#     # Find the mean: Take each row of X and average among each interval separately
#     interval_t = np.linspace(np.min(Yt), np.max(Yt), num=Ht+1)
#     mh_t = []

#     for i in range(len(interval_t) - 1):
#         mask_t = (Yt >= interval_t[i]) & (Yt <= interval_t[i+1])
#         xt_rows_mean = np.mean(Xt_centered[mask_t], axis=0)
#         mh_t.append(xt_rows_mean)

#     mh_t_array = np.array(mh_t)

#     # Cov(E[Xt|Yt])
#     sigma_Xt = np.zeros((Xt_centered.shape[1], Xt_centered.shape[1]))
#     for i in range(len(interval_t) - 1):
#         mask_t = (Yt >= interval_t[i]) & (Yt <= interval_t[i+1])
#         interval_mh_t = mh_t_array[i]
#         outer_product_t = np.outer(interval_mh_t, interval_mh_t)

#         if not np.isnan(outer_product_t).any():
#             sigma_Xt += outer_product_t

#     sigma_Xt_a = np.array(sigma_Xt)
#     Xt_cov_matrix_a = np.array(Xt_cov_matrix)

#     # Generalized Eigenvalue Decomposition
#     At = Xt_cov_matrix_a @ sigma_Xt_a @ Xt_cov_matrix_a
#     Bt = Xt_cov_matrix_a @ Xt_cov_matrix_a

#     # Optimization on Manifold
#     stiefel = geoopt.manifolds.Stiefel()
#     initial_guess = torch.randn(p, d)
#     initial_guess, _ = torch.linalg.qr(initial_guess)

#     v = geoopt.ManifoldParameter(initial_guess, manifold=stiefel)
#     optimizer = RiemannianSGD([v], lr=0.1)

#     v = v.to(torch.double)

#     for step in range(100):
#         optimizer.zero_grad()
#         gradient = grad(A, B, a, v, At, Bt)
#         proj_grad = stiefel.proju(v, gradient)
#         v.grad = proj_grad
#         # v.manifold.proju(v, gradient)
#         cost = f(A, B, a, v, At, Bt)
#         optimizer.step()

#     optimized_matrix = v

#     print("Optimized Orthonormal Matrix: ")
#     print(optimized_matrix)

#     optimized_tensor = v.clone().detach()
#     print("Optimized Orthonormmal Matrix (as PyTorch tensor):")
#     print(optimized_tensor)

#     return v


# def f(A, B, a, v, At, Bt):
#     A = torch.tensor(A)
#     B = torch.tensor(B)
#     At = torch.tensor(At)
#     Bt = torch.tensor(Bt)

#     bv_inv = torch.inverse(torch.matmul(torch.matmul(v.t(), B), v))
#     va = torch.matmul(torch.matmul(v.T, A), v)
#     bv_t_inv = torch.inverse(torch.matmul(torch.matmul(v.t(), Bt), v))
#     va_t = torch.matmul(torch.matmul(v.T, At), v)

#     f_v = -torch.trace(torch.matmul(va, bv_inv)) + a * \
#         torch.trace(torch.matmul(va_t, bv_t_inv))

#     return f_v


# def grad(A, B, a, v, At, Bt):
#     A = torch.tensor(A)
#     B = torch.tensor(B)
#     At = torch.tensor(At)
#     Bt = torch.tensor(Bt)

#     v = v.to(torch.double)
#     A = A.to(torch.double)
#     B = B.to(torch.double)
#     At = At.to(torch.double)
#     Bt = Bt.to(torch.double)

#     bv_inv = torch.inverse(v.t() @ B @ v)
#     va = v.t() @ A @ v
#     bv_t_inv = torch.inverse(v.t() @ Bt @ v)
#     va_t = v.t() @ At @ v

#     avb = A @ v @ bv_inv - B @ v @ bv_inv @ va @ bv_inv
#     avb_t = At @ v @ bv_t_inv - Bt @ v @ bv_t_inv @ va_t @ bv_t_inv

#     gradient = -2 * (avb - a * avb_t)

#     return gradient
