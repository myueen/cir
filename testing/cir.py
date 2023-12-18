import numpy as np
import pandas as pd
import math
import sliced
from sliced import base
from sliced.base import slice_y, grouped_sum
from scipy.linalg import eigh
from scipy.linalg import eig
from numpy.linalg import norm
import torch
import geoopt
from geoopt.optim import RiemannianSGD
from geoopt.manifolds import Stiefel
from scipy.linalg import solve
from scipy.linalg import cholesky


def CIR(X, Y, Xt, Yt, alpha, d):
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

    # n represents # of observation of foreground; p represents # of features of foreground
    # m represents # of observation of background; k represents # of features of background
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
        H = 2
    elif 2 < Y_unique_value <= 10:
        H = Y_unique_value
    else:
        if d <= 2:
            H = 10
        else:
            H = 4

    # Define Ph, count the of ocurrence of y in each H interval

    """The following is for background data and the caase when a > 0"""
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
