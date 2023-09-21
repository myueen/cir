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

    # Center the data
    X_colmeans = X_df.sum()/n
    # test: print(X_colmeans) .
    # test: print(X_)
    X_ = X_df - X_colmeans

    Xt_colmean = Xt_df.sum()/n
    Xt_ = Xt_df - Xt_colmean

    # Covariance Matrix
    X_cov_matrix = X_.cov()
    print(X_cov_matrix)


# Test code
X = [[1, 4, 5],
     [-5, 8, 9]]

Y = [[1], [2]]

Xt = [[1, 4, 5],
      [-5, 8, 9]]

Yt = [[1], [2]]

CIR(X, Y, Xt, Yt, 1, 3)
