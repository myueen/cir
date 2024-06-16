import numpy as np
import pandas as pd
import time
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

import contrastive_inverse_regression
from contrastive_inverse_regression import CIR

# first unzip 'pbmc_1_counts.csv.zip'
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

# Dimensionality settings
Ds = range(2, 11)
Accuracy_KNN = []
Accuracy_Tree = []
Accuracy_boosting = []
Accuracy_NN = []

for d in Ds:
    print(f'd = {d}')

    # PCA
    print('PCA......')
    pca = PCA(n_components=d)
    X_PCA = pca.fit_transform(X)

    # CPCA
    print('CPCA......')
    cov_fg = np.cov(fg, rowvar=False)
    cov_bg = np.cov(bg, rowvar=False)
    eigvals, eigvecs = np.linalg.eig(cov_fg - alpha * cov_bg)
    idx = eigvals.argsort()[::-1][:d]
    V_CPCA = eigvecs[:, idx]
    X_CPCA = X @ V_CPCA

    # LDA
    print('LDA......')
    lda = LDA(n_components=d)
    X_LDA = lda.fit_transform(X, Y)

    # LASSO
    print('LASSO......')
    lasso = LassoCV().fit(X, Y)
    B = lasso.coef_
    selected_idx = np.argsort(np.abs(B))[-d:]
    X_LASSO = X[:, selected_idx]

    # SIR
    print('SIR......')
    Sigma_XX = np.cov(X, rowvar=False)
    Sigma_X = np.zeros((p, p))
    for label in labels:
        X_curr = X[Y == label]
        n_curr = X_curr.shape[0]
        mean_diff = np.mean(X_curr, axis=0) - np.mean(X, axis=0)
        Sigma_X += np.outer(mean_diff, mean_diff) * n_curr
    Sigma_X /= n
    eigvals, eigvecs = np.linalg.eig(Sigma_XX, Sigma_X)
    idx = eigvals.argsort()[::-1][:d]
    V_SIR = eigvecs[:, idx]
    X_SIR = X @ V_SIR

    # CIR
    print('CIR......')
    # Assuming CIR is a defined function
    V_CIR = CIR(fg, Y, bg, Yt, alpha, d)
    X_CIR = X @ V_CIR

    T = 10
    for t in range(T):
        print(f't = {t}')

        # KNN
        print('Training KNN')
        knn = KNeighborsClassifier()
        Accuracy_KNN.append([
            cross_val_score(knn, fg, Y, cv=10).mean(),
            cross_val_score(knn, X_PCA, Y, cv=10).mean(),
            cross_val_score(knn, X_CPCA, Y, cv=10).mean(),
            cross_val_score(knn, X_LDA, Y, cv=10).mean(),
            cross_val_score(knn, X_LASSO, Y, cv=10).mean(),
            cross_val_score(knn, X_SIR, Y, cv=10).mean(),
            cross_val_score(knn, X_CIR, Y, cv=10).mean()
        ])

        # Decision Tree
        print('Training Decision Tree')
        tree = DecisionTreeClassifier()
        Accuracy_Tree.append([
            cross_val_score(tree, fg, Y, cv=10).mean(),
            cross_val_score(tree, X_PCA, Y, cv=10).mean(),
            cross_val_score(tree, X_CPCA, Y, cv=10).mean(),
            cross_val_score(tree, X_LDA, Y, cv=10).mean(),
            cross_val_score(tree, X_LASSO, Y, cv=10).mean(),
            cross_val_score(tree, X_SIR, Y, cv=10).mean(),
            cross_val_score(tree, X_CIR, Y, cv=10).mean()
        ])

        # Boosting
        print('Training Boosting')
        boosting = AdaBoostClassifier()
        Accuracy_boosting.append([
            cross_val_score(boosting, fg, Y, cv=10).mean(),
            cross_val_score(boosting, X_PCA, Y, cv=10).mean(),
            cross_val_score(boosting, X_CPCA, Y, cv=10).mean(),
            cross_val_score(boosting, X_LDA, Y, cv=10).mean(),
            cross_val_score(boosting, X_LASSO, Y, cv=10).mean(),
            cross_val_score(boosting, X_SIR, Y, cv=10).mean(),
            cross_val_score(boosting, X_CIR, Y, cv=10).mean()
        ])

        # Neural Network
        print('Training Neural Network')
        nn = MLPClassifier(max_iter=1000)
        Accuracy_NN.append([
            cross_val_score(nn, fg, Y, cv=10).mean(),
            cross_val_score(nn, X_PCA, Y, cv=10).mean(),
            cross_val_score(nn, X_CPCA, Y, cv=10).mean(),
            cross_val_score(nn, X_LDA, Y, cv=10).mean(),
            cross_val_score(nn, X_LASSO, Y, cv=10).mean(),
            cross_val_score(nn, X_SIR, Y, cv=10).mean(),
            cross_val_score(nn, X_CIR, Y, cv=10).mean()
        ])

# Convert lists to arrays for easy manipulation
Accuracy_KNN = np.array(Accuracy_KNN)
Accuracy_Tree = np.array(Accuracy_Tree)
Accuracy_boosting = np.array(Accuracy_boosting)
Accuracy_NN = np.array(Accuracy_NN)

# Calculate mean accuracy
Accuracy_KNN_mean = Accuracy_KNN.mean(axis=0)

# Plotting KNN accuracy

plt.figure()
for j in range(Accuracy_KNN_mean.shape[1]):
    plt.plot(Ds, Accuracy_KNN_mean[:, j], linewidth=2)
plt.legend(['Raw', 'PCA', 'CPCA', 'LDA', 'LASSO', 'SIR', 'CIR'], fontsize=10)
plt.title('KNN Accuracy')
plt.show()
