import numpy as np
import pandas as pd
from scipy.linalg import eig


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
        raise ValueError("Yt must have the same number of rows as X")

    if not isinstance(d, int):
        raise TypeError("d parameter must be an integer")

    if d < 0:
        raise ValueError("d must be greater than or equal to 0")

    if a < 0:
        raise ValueError("a must be greater than or equal to 0")

    X_df = pd.DataFrame(X)
    Xt_df = pd.DataFrame(Xt)
    Y_df = pd.DataFrame(Y)
    Yt_df = pd.DataFrame(Yt)

    # Center the data
    X_col_means = X_df.mean(axis=0)
    X_centered = X_df - X_col_means
    Xt_col_means = Xt_df.mean(axis=0)
    Xt_centered = Xt_df - Xt_col_means

    # Covariance Matrix
    X_cov_matrix = X_centered.cov()
    Xt_cov_matrix = Xt_centered.cov()

    # Define H, Ht
    Y_unique_value = Y_df.nunique().item()
    Yt_unique_value = Yt_df.nunique().item()
    if Y_unique_value == 2:
        H = 2
    elif Y_unique_value > 2 & Y_unique_value <= 10:
        H = Y_unique_value
    else:
        if d <= 2:
            H = 10
        else:
            H = 4

    if Yt_unique_value == 2:
        Ht = 2
    elif Yt_unique_value > 2 & Yt_unique_value <= 10:
        Ht = Yt_unique_value
    else:
        if d <= 2:
            Ht = 10
        else:
            Ht = 4

    # Define Ph     (Count the # of ocurrence of y in each H interval)
    interval_Ph = pd.cut(Y_df[0], bins=H)
    Ph = interval_Ph.value_counts().sort_index()  # interval_counts

    # Find the mean: Take each row of X and average among each interval separately
    interval = np.linspace(np.min(Y), np.max(Y), num=H+1)
    mask = (Y >= interval[:-1, np.newaxis]) & (Y <= interval[1:, np.newaxis])
    mh = np.mean(X_centered[mask], axis=0)

    # Cov(E[X|Y])
    sigma_X = np.outer(mh, mh)

    if a == 0:
        v = eig(X_cov_matrix, sigma_X)
        f_v = - np.trace()

    elif a > 0:
        A = np.dot(np.dot(X_cov_matrix, sigma_X), X_cov_matrix)
        B = np.dot(X_cov_matrix, X_cov_matrix)

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

    print(interval)
    print(mask)
    print("")
    print(mh)


# Test code
X = [[1, 4, 5],
     [-5, 8, 9],
     [2, 3, 6]]

Y_numerical = [1, 8.3, 2.7]
Y_binary = [0, 1, 1]
Y_unique_less_10 = [1, 2, 3]
Y_categorical = ["a", "b", "b"]

Xt = [[1, 4, 5],
      [-5, 8, 9]]

Yt = [[1], [2]]

CIR(X, Y_numerical, Xt, Yt, 1, 2)
