import numpy as np
import pandas as pd


def CIR(X, Y, Xt, Yt, a, d):
    n = len(X)
    p = len(X[0])

    m = len(Xt)

    # Check same dimensions
    if len(Xt[0]) != p:
        raise ValueError("Xt should have the same # of columns as X")

    if len(Y) != n:
        raise ValueError("Y must have the same # of rows as X")

    if len(Yt) != m:
        raise ValueError("Yt must have the same # of rows as X")

    if d < 0:
        raise ValueError("d must be a non-negative number.")

    if a < 0:
        raise ValueError("a must be a non-negative number.")

    X_df = pd.DataFrame(X)
    Xt_df = pd.DataFrame(Xt)
    Y_df = pd.DataFrame(Y)
    Yt_df = pd.DataFrame(Yt)
    # test: print(Y_df)

    # Center the data
    X_colmeans = X_df.sum()/n
    # test: print(X_colmeans)
    X_ = X_df - X_colmeans
    # test: print(X_)

    Xt_colmean = Xt_df.sum()/m
    Xt_ = Xt_df - Xt_colmean

    # Covariance Matrix
    X_cov_matrix = (1/n) * np.matmul(X_.transpose(), X_)
    Xt_cov_matrix = (1/m) * np.matmul(Xt_.transpose(), Xt_)
    # test: print(X_.transpose())
    # test: print(X_cov_matrix)

    # Define H, Ht
    is_numerical = np.issubdtype(Y_df.to_numpy().dtype, np.number)
    # test: print(Y_df.dtypes)
    # test: print(is_numerical)
    is_categorical = not is_numerical

    if is_categorical:
        H = int(Y_df.nunique()[0])
        print(H)
    elif is_numerical:
        if d <= 2:
            H = 10
            # test: print(H)
        else:
            H = 4
            # test: print(H)






# Test code
X = [[1, 4, 5],
     [-5, 8, 9]]

Y_numerical = [[1], [2]]

Y_categorical = ["a", "b"]

Xt = [[1, 4, 5],
      [-5, 8, 9]]

Yt = [[1], [2]]

CIR(X, Y_categorical, Xt, Yt, 1, 2)
