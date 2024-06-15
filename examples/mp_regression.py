import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from cir import CIR


data = pd.read_csv('Data_Cortex_Nuclear.csv')


# foreground data
fg = data.dropna()

# foreground label
Y = fg['class']
Y = pd.Categorical(Y)
Y = Y.rename_categories({'c-CS-m': '0', 'c-CS-s': '1', 'c-SC-m': '2', 'c-SC-s': '3',
                         't-CS-m': '4', 't-CS-s': '5', 't-SC-m': '6', 't-SC-s': '7'})
Y = Y.astype(float)
labels = np.unique(Y)      # set of unique foreground labels

# foreground slices
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

# tuning parameter
alpha = 0.0002
d = 2

print("CIR......")
accurary_group = []


for i in range(1):
    V_CIR = CIR(fg, Y, bg, Yt, alpha, d)
    X_CIR = X @ V_CIR
    mdl_CIR_KNN = KNeighborsClassifier(n_neighbors=1)
    # kf = KFold(n_splits=10)
    # kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cross_val_results = cross_val_score(
        mdl_CIR_KNN, X_CIR, Y, cv=10, scoring='accuracy')
    print(cross_val_results)
    Accuracy_CIR_KNN = np.mean(cross_val_results)
    accurary_group.append(Accuracy_CIR_KNN)

print("this is CIR_KNN ", accurary_group)
print("This is mean ", np.mean(accurary_group))
print("This is standard deviation", np.std(accurary_group))


accurary_group_tree = []

for i in range(1):
    V_CIR = CIR(fg, Y, bg, Yt, alpha, d)
    X_CIR = X @ V_CIR
    mdl_CIR_Tree = DecisionTreeClassifier()
    # kf = KFold(n_splits=10)
    # kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cross_val_results = cross_val_score(
        mdl_CIR_Tree, X_CIR, Y, cv=10, scoring='accuracy')
    print(cross_val_results)
    Accuracy_CIR_Tree = cross_val_results.mean()
    accurary_group_tree.append(Accuracy_CIR_Tree)

print("this is CIR_Tree ", accurary_group_tree)
print("This is mean ", np.mean(accurary_group_tree))
print("This is standard deviation ", np.std(accurary_group_tree))
