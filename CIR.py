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
    print(X_)

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
        # test: print(H)
    elif is_numerical:
        if d <= 2:
            H = 10
            # test: print(H)
        else:
            H = 4
            # test: print(H)
        # Partition linearly
        max_Y = np.max(Y_df)
        min_Y = np.min(Y_df)
        width = (max_Y - min_Y) / H
        partition_pts = [min_Y + i * width for i in range(H)]
        partition_pts.append(max_Y)
        print(partition_pts)

    # Define Ph     (Count the # of ocurrence of y in each H interval)
    Ph = [0] * H
    mh = [0] * H

    for elt in Y_df.to_numpy():
        for i in range(H):
            if partition_pts[i] <= elt <= partition_pts[i + 1]:
                Ph[i] += 1
                Y_df.values.tolist().index(elt)

    print(Ph)

    # Find th mean: Take each row of X and average among each interval separately
    mh = [0] * H
    interval_indices = []

    # indices = np.searchsorted(partition_pts, elt)
    # interval_indices.append(indices)
    # print(interval_indices)

    # for interval in partition_pts:
    #     indices = np.where(Y_df < interval)
    # print(indices)
    # if Ph[i] != 0:
    #     filtered_mean = (1/Ph[i]) * row_mean[indices]
    #     i += 1
    #     mh.append(filtered_mean)


# Test code
X = [[1, 4, 5],
     [-5, 8, 9],
     [2, 3, 6],
     [4, 3, 2],
     [2, 3, 4]]

Y_numerical = [[1], [8.3], [2.7], [3], [8]]

Y_categorical = ["a", "b"]

Xt = [[1, 4, 5],
      [-5, 8, 9]]

Yt = [[1], [2]]

CIR(X, Y_numerical, Xt, Yt, 1, 2)
