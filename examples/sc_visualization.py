import numpy as np
import pandas as pd
import time
from scipy.linalg import eigh
import matplotlib.pyplot as plt

import cir
from cir import CIR
from importlib import reload
reload(cir)


data = pd.read_csv('pbmc_1_counts.csv')
data = data.iloc[:, 1:]
data = data.transpose()

cell_type = pd.read_csv('pbmc_1_cell_type.csv')
cell_type = cell_type.iloc[:, 1].values

# foreground data
fg = data.dropna()

p = 100     # p can vary from 100 to 500
# select the top p highly variable genes
col_var = np.var(fg, axis=0)
col_var_sorted_idx = np.argsort(-col_var)
fg = fg.iloc[:, col_var_sorted_idx[:p]]

# foreground label: cell types
Y = cell_type
Y = pd.Categorical(Y)
Y = Y.rename_categories({'B cell': '0', 'CD4 T cell': '1', 'CD8 T cell': '2', 'NK cell': '3',
                         'Plasma cell': '4', 'cDC': '5', 'cMono': '6', 'ncMono': '7', 'pDC': '8'})
Y = Y.astype(float)
labels = np.unique(Y)      # set of unique foreground labels
L = len(labels)         # number of foreground classes/slices
n = fg.shape[0]  # foreground sample size
X = fg - np.mean(fg, axis=0)
X = X.values

# background data
bg = fg
m, p = bg.shape  # background sample size
Yt = 3 * np.ones((m, 1))  # background labels
Yt = np.random.randint(1, 10, size=(m, 1))


# tuning parameter
alpha = 1.5
d = 2

# CIR
print("CIR......")
V_CIR = CIR(fg, Y, bg, Yt, alpha, d)
X_CIR = X @ V_CIR

colors = [[228/255, 26/255, 28/255],
          [55/255, 126/255, 184/255],
          [77/255, 175/255, 74/255],
          [152/255, 78/255, 163/255],
          [255/255, 127/255, 0/255],
          [255/255, 255/255, 51/255],
          [166/255, 86/255, 40/255],
          [247/255, 129/255, 191/255],
          [153/255, 153/255, 153/255]]


markers = ['o', 's', 'p', 'o', 's', 'p', 'o', 's', 'p']

for l in range(L):
    idx = np.where(Y == labels[l])[0]
    X_curr = X_CIR[idx, :]
    plt.scatter(X_curr[:, 0], X_curr[:, 1], s=100,
                c=colors[l], marker=markers[l])
# plt.hold(False)
plt.title('CIR', fontsize=32)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.show()
