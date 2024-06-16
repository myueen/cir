from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import umap
import numpy as np
import pandas as pd
import time
import matplotlib
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.metrics import silhouette_score
from scipy.linalg import eig
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import contrastive_inverse_regression
from contrastive_inverse_regression import CIR


data = pd.read_csv(
    'Data_Cortex_Nuclear.csv')


# foreground data
fg = data.dropna()


# foreground label
Y = fg['class']
Y = pd.Categorical(Y)
Y = Y.rename_categories({'c-CS-m': '0', 'c-CS-s': '1', 'c-SC-m': '2', 'c-SC-s': '3',
                        't-CS-m': '4', 't-CS-s': '5', 't-SC-m': '6', 't-SC-s': '7'})
Y = Y.astype(float)
labels = np.unique(Y)      # set of unique foreground labels
L = len(labels)         # number of foreground classes/slices
fg = fg.iloc[:, 1:78]
n = fg.shape[0]  # foreground sample size
X = fg - np.mean(fg, axis=0)
X = X.values


# background data
bg = data[data['Genotype'] == 'Control'].copy()
bg = bg.dropna()


# background label
Yt = bg['Behavior']
Yt = pd.Categorical(Yt)
Yt = Yt.rename_categories({'C/S': '0', 'S/C': '1'})


Yt = Yt.astype(float)
labelst = np.unique(Yt)   # set of unique background labels
Lt = len(labelst)       # number of background classes/slices
bg = bg.iloc[:, 1:78]
m, p = bg.shape


# tuning parameter alpha
alpha = 0.0001
# reduce dimension from 77 to 2 for visualization
d = 2


# np.random.seed(22)      # add seed


# CIR
print("CIR......")
start_time = time.time()


V_CIR = CIR(fg, Y, bg, Yt, alpha, d)
X_CIR = X @ V_CIR


end_time = time.time()


# SIR
print("SIR...")
Sigma_XX = X.T @ X / n
Sigma_X = np.zeros((p, p))
for l in labels:
    X_curr = fg.values[Y == l]
    n_curr = X_curr.shape[0]
    Sigma_X += n_curr * np.outer(X_curr.mean(axis=0) - fg.values.mean(
        axis=0), X_curr.mean(axis=0) - fg.values.mean(axis=0))


Sigma_X /= n
eigvals, eigvecs = eig(Sigma_XX, Sigma_X)
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
eigvals, eigvecs = eig(cov_fg - alpha_CPCA * cov_bg)
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


# # LASSO
print("LASSO...")
lasso = Lasso(alpha=0.1)
lasso.fit(X, Y)
selected_features = np.where(lasso.coef_ != 0)[0][:d]
X_LASSO = X[:, selected_features]


# colors credit to Color Brewer
colors = [[228/255, 26/255, 28/255],
          [55/255, 126/255, 184/255],
          [77/255, 175/255, 74/255],
          [152/255, 78/255, 163/255],
          [255/255, 127/255, 0/255],
          [255/255, 255/255, 51/255],
          [166/255, 86/255, 40/255],
          [247/255, 129/255, 191/255]]


markers = ['o', '^', 's', 'p', 'o', '^', 's', 'p']


fig, axs = plt.subplots(2, 4, figsize=(15, 15))


# Function to plot scatter subplots
def plot_scatter(ax, X, title, fontsize=32):
    for l in range(L):
        X_curr = X[Y == labels[l]]
        ax.scatter(X_curr[:, 0], X_curr[:, 1], 100, color=colors[l],
                   marker=markers[l], label=f'Label {labels[l]}', edgecolors='w')
    ax.set_title(title, fontsize=fontsize)


# Plotting each subplot
plot_scatter(axs[0, 0], X_PCA, 'PCA')
plot_scatter(axs[0, 1], X_CPCA, 'CPCA')
plot_scatter(axs[0, 2], X_LDA, 'LDA')
plot_scatter(axs[0, 3], X_LASSO, 'LASSO')
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
