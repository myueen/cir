import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import cir_
from cir_ import CIR
from importlib import reload
reload(cir_)


data = pd.read_table('contrastive-inverse-regression/cir/Retinol.txt',
                     header=None, delim_whitespace=True)
data = data.iloc[:, :]

# foreground data
fg = data.iloc[:, :12].dropna()

# foreground response (continuous)
Y = data.iloc[:, 12]
n, p = fg.shape  # foreground sample size
X = fg - np.mean(fg, axis=0)
X = X.values

# foreground slices
L = 10
partition = np.linspace(min(Y), max(Y), L)
labels = np.zeros(n)
for i in range(n):
    labels[i] = max(np.where(Y.iloc[i] >= partition)[0]) + 1

# background data
bg = data.iloc[:, :12].dropna()
m, p = bg.shape     # background sample size

# background data
Yt = data.iloc[:, 13]
Lt = 10
partitiont = np.linspace(min(Yt), max(Yt), Lt)
labelst = np.zeros(m)
for i in range(m):
    labelst[i] = max(np.where(Yt.iloc[i] >= partitiont)[0]) + 1

d = 2
alpha = 0.1


V_CIR = CIR(fg, labels, bg, labelst, alpha, d)
X_CIR = X @ V_CIR


print("Training Linear Model")
model_CIR_LM = LinearRegression().fit(X_CIR, Y)
print("coefficients: ", model_CIR_LM.coef_)
print("intercept: ", model_CIR_LM.intercept_)
print("R-squared: ", model_CIR_LM.score(X_CIR, Y))

Y_pred = model_CIR_LM.predict(X_CIR)
MSE_CIR_LM = mean_squared_error(Y, Y_pred)

print("This is MSE: ", MSE_CIR_LM)
