import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import minimize
from pymanopt.manifolds import Stiefel
from pymanopt import Problem


def CIR(X, Y, Xt, Yt, a, d):
    n = len(X)
    p = len(X[0])
    m = len(Xt)

    # Parameter Check
    if len(Xt[0]) != p:
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

    X_df = pd.DataFrame(X)
    Xt_df = pd.DataFrame(Xt)
    Y_df = pd.DataFrame(Y)
    Yt_df = pd.DataFrame(Yt)

    # Center the data
    X_col_means = X_df.mean(axis=0)
    X_centered = X_df - X_col_means

    # Covariance Matrix
    X_cov_matrix = X_centered.cov()

    # Define H, Ht
    Y_unique_value = Y_df.nunique().item()
    if Y_unique_value == 2:
        H = 2
    elif Y_unique_value > 2 & Y_unique_value <= 10:
        H = Y_unique_value
    else:
        if d <= 2:
            H = 10
        else:
            H = 4

    # Define Ph     (Count the # of ocurrence of y in each H interval)
    interval_Ph = pd.cut(Y_df[0], bins=H)
    Ph = interval_Ph.value_counts().sort_index()

    # Find the mean: Take each row of X and average among each interval separately
    interval = np.linspace(np.min(Y), np.max(Y), num=H+1)

    mh = []
    for i in range(len(interval) - 1):
        mask = (Y >= interval[i]) & (Y <= interval[i+1])
        x_rows_mean = np.mean(X_centered[mask], axis=0)
        mh.append(x_rows_mean)

    mh_array = np.array(mh)

    # Cov(E[X|Y])
    sigma_X = np.zeros((X_centered.shape[1], X_centered.shape[1]))
    for i in range(len(interval) - 1):
        mask = (Y >= interval[i]) & (Y <= interval[i+1])
        interval_mh = mh_array[i]
        outer_product = np.outer(interval_mh, interval_mh)

        if not np.isnan(outer_product).any():
            sigma_X += outer_product

    sigma_X_a = np.array(sigma_X)
    X_cov_matrix_a = np.array(X_cov_matrix)

    # Generalized Eigenvalue Decomposition
    A = np.matmul(np.matmul(X_cov_matrix_a, sigma_X_a), X_cov_matrix_a)
    B = np.matmul(X_cov_matrix_a, X_cov_matrix_a)

    if a == 0:
        eigenvalues, eigenvectors = eigh(X_cov_matrix_a, sigma_X_a)
        v = eigenvectors[:d, :]
        return v, f(A, B, a, v, 0, 0)

    # ====================================================================================
    # ====================================================================================
    # The following are the same process to calculate the background data
    # And the case when a > 0

    # background data
    Xt_col_means = Xt_df.mean(axis=0)
    Xt_centered = Xt_df - Xt_col_means
    Xt_cov_matrix = Xt_centered.cov()
    Yt_unique_value = Yt_df.nunique().item()

    if Yt_unique_value == 2:
        Ht = 2
    elif Yt_unique_value > 2 & Yt_unique_value <= 10:
        Ht = Yt_unique_value
    else:
        if d <= 2:
            Ht = 10
        else:
            Ht = 4

    interval_Ph_t = pd.cut(Yt_df[0], bins=Ht)
    Ph_t = interval_Ph_t.value_counts().sort_index()
    interval_t = np.linspace(np.min(Yt), np.max(Yt), num=Ht+1)
    mh_t = []

    for i in range(len(interval_t) - 1):
        mask_t = (Yt >= interval_t[i]) & (Yt <= interval_t[i+1])
        xt_rows_mean = np.mean(Xt_centered[mask_t], axis=0)
        mh_t.append(xt_rows_mean)

    sigma_Xt = np.zeros((Xt_centered.shape[1], Xt_centered.shape[1]))
    for i in range(len(interval_t) - 1):
        mask_t = (Yt >= interval_t[i]) & (Yt <= interval_t[i+1])
        interval_mh_t = mh_t[i]
        outer_product_t = np.outer(interval_mh_t, interval_mh_t)
        if not np.isnan(outer_product_t).any():
            sigma_Xt += outer_product_t
        sigma_Xt += outer_product_t

    # f(v)
    sigma_Xt_df = pd.DataFrame(sigma_Xt)
    At = np.matmul(np.matmul(Xt_cov_matrix, sigma_Xt_df), Xt_cov_matrix)
    Bt = np.matmul(Xt_cov_matrix, Xt_cov_matrix)
    bv_inverse_t = np.linalg.inv(np.matmul(np.matmul(v.T, Bt), v))
    va_t = np.matmul(np.matmul(v.T, At), v)
    trace_ab_t = np.matmul(va_t, bv_inverse_t)

    # gradient f(v)
    # avb = np.dot(np.dot(A, v), bv_inverse) - \
    #     np.dot(np.dot(np.dot(np.dot(B, v), bv_inverse), va), bv_inverse)
    # avb_t = np.dot(np.dot(At, v), bv_inverse_t) - \
    #     np.dot(np.dot(np.dot(np.dot(Bt, v), bv_inverse_t), va_t), bv_inverse_t)
    # grad = -2 * (avb - a * (avb_t))

    # if a > 0:
    # f_v = -1 * (np.trace(trace_ab)) + a * (np.trace(trace_ab_t))

    # ====================================================================================
    # ====================================================================================
    # The following are print statement for code testing.
    # print("Original Matix")
    # print(X_df)
    # print("             ")
    # print("Centered Matix")
    # print(X_centered)
    # print("             ")
    # print("Covariance Matix for X")
    # print(X_cov_matrix)
    # print("             ")
    # print("Y original 1d array")
    # print(Y_df)
    # print("             ")
    # print("Number of unique value in Y")
    # print(Y_unique_value)
    # print("             ")
    # print("H: # intervals split range(Y)")
    # print(H)
    # print("             ")
    # print("Intervals and count: ")
    # print(Ph)


def f(A, B, a, v, At=0, Bt=0):
    bv_inverse = np.linalg.inv(np.matmul(np.matmul(v.T, B), v))
    va = np.matmul(np.matmul(v.T, A), v)
    f_v = -1 * (np.trace(np.matmul(va, bv_inverse)))
    return f_v


# Test code
X = [[1, 4, 5],
     [-5, 8, 9],
     [2, 3, 6]]

X1 = [[10, 18, 5],
      [-8, 11, 12],
      [2, 17, 2]]

X2 = [[55, 28, 34],
      [30, 32, 35],
      [45, 29, 28]]

X3 = [[120, 28, 34],
      [30, 98, 35],
      [45, 103, 28],
      [33, 62, 59],
      [89, 75, 28]]

Y_numerical = [1, 8.3, 2.7]
Y1 = [1, 13, 20, 55, 23]
Y_binary = [0, 1, 1]
Y_unique_less_10 = [1, 2, 3]
Y_categorical = ["a", "b", "b"]

Xt = [[1, 4, 5],
      [-5, 8, 9]]

Yt = [[1], [2]]

result = CIR(X3, Y1, Xt, Yt, 0, 3)
# print(result)
# print(result[0])
# print(result[1])
