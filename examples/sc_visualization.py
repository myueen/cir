import numpy as np
import pandas as pd
import time
import scipy
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import Lasso
from sklearn.manifold import TSNE
import umap

import contrastive_inverse_regression
from contrastive_inverse_regression import CIR


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

print(Y)


# background data
bg = fg
m, p = bg.shape  # background sample size
Yt = 3 * np.ones((m, 1))  # background labels
Yt = np.random.randint(1, 10, size=(m, 1))


# tuning parameter
alpha = 1.5
d = 2


# np.random.seed(24)      # add seed


# CIR
print("CIR......")
V_CIR = CIR(fg, Y, bg, Yt, alpha, d)
X_CIR = X @ V_CIR


val = fg.values
print(val.shape)
print(Y.shape)

# SIR
print("SIR...")
Sigma_XX = X.T @ X / n
Sigma_X = np.zeros((p, p))
for l in range(L):
    print(l)
    X_curr = fg.values[Y == l]
    n_curr = X_curr.shape[0]
    Sigma_X += n_curr * np.outer(X_curr.mean(axis=0) - fg.values.mean(
        axis=0), X_curr.mean(axis=0) - fg.values.mean(axis=0))


Sigma_X /= n
eigvals, eigvecs = scipy.linalg.eig(Sigma_XX, Sigma_X)
V_SIR = eigvecs[:, :d]
X_SIR = X @ V_SIR


# PCA
print("PCA...")
pca = PCA(n_components=d)
X_PCA = pca.fit_transform(X)


# CPCA
print("CPCA...")
alpha_CPCA = 2
cov_fg = np.cov(fg.values, rowvar=False)
cov_bg = np.cov(bg.values, rowvar=False)
eigvals, eigvecs = eigh(cov_fg - alpha_CPCA * cov_bg)
V_CPCA = eigvecs[:, :d]
X_CPCA = X @ V_CPCA


# t-SNE
print("t-SNE...")
tsne = TSNE(n_components=d)
X_tSNE = tsne.fit_transform(fg.values)


# UMAP
print("UMAP...")
umap_model = umap.UMAP(n_components=d)
X_UMAP = umap_model.fit_transform(fg.values)


# LDA
print("LDA...")
lda = LDA(n_components=d)
X_LDA = lda.fit_transform(X, Y)


# LASSO
print("LASSO...")
lasso = Lasso(alpha=0.1)
lasso.fit(X, Y)
selected_features = np.where(lasso.coef_ != 0)[0][:d]
X_LASSO = X[:, selected_features]


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


fig, axs = plt.subplots(2, 4, figsize=(15, 15))


# Function to plot scatter subplots
def plot_scatter(ax, X, title, fontsize=32, ylim=None):
    for l in range(L):
        X_curr = X[Y == labels[l]]
        ax.scatter(X_curr[:, 0], X_curr[:, 1], 100, color=colors[l],
                   marker=markers[l], label=f'Label {labels[l]}', edgecolors='w')
    ax.set_title(title, fontsize=fontsize)
    if ylim:
        ax.set_ylim(ylim)


# Plotting each subplot
plot_scatter(axs[0, 0], X_PCA, 'PCA')
plot_scatter(axs[0, 1], X_CPCA, 'CPCA')
plot_scatter(axs[0, 2], X_LDA, 'LDA')
plot_scatter(axs[0, 3], X_LASSO, 'LASSO', ylim=(-250, 250))
plot_scatter(axs[1, 0], X_SIR, 'SIR')
plot_scatter(axs[1, 1], X_CIR, 'CIR', fontsize=22)
plot_scatter(axs[1, 2], X_tSNE, 'tSNE')
plot_scatter(axs[1, 3], X_UMAP, 'UMAP')


plt.tight_layout()
plt.show()


print('Silhouette scores for raw, PCA, CPCA, LDA, LASSO, SIR, CIR, tSNE, UMAP')
silhouette_scores = [
    silhouette_score(X_PCA, Y),
    silhouette_score(X_CPCA, Y),
    silhouette_score(X_LDA, Y),
    silhouette_score(X_LASSO, Y),
    silhouette_score(X_SIR, Y),
    silhouette_score(X_CIR, Y),
    silhouette_score(X_tSNE, Y),
    silhouette_score(X_UMAP, Y),
]


methods = ['PCA', 'CPCA', 'LDA', 'LASSO', 'SIR', 'CIR', 'tSNE', 'UMAP']
silhouette_df = pd.DataFrame({
    'Method': methods,
    'Silhouette Score': silhouette_scores,
    # 'Calinski-Harabasz Score': calinski_harabasz_scores
})


print(silhouette_df)
