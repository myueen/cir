[![Python Package Index](https://img.shields.io/pypi/v/contrastive-inverse-regression.svg)](https://pypi.org/project/contrastive-inverse-regression)

contrastive_inverse_regression
======

This repository contains Matlab files and notebooks to reproduce results in clustering analysis and classification accuracy of various dimension reduction methods for biomedical data experiments. (see [manuscript](
https://doi.org/10.48550/arXiv.2305.12287)). 

This repository contains the main contrastive-inverse-regression (CIR) algorithm in both Matlab and python. But since clustering analysis and classification analysis are conducted in Matlab, it is our intent for the purpose of reproducibility to use the Matlab file (CIR.m) when running Matlab files in the experiments folder. The python package is available as an feasible alternative. 


Matlab files Dependency 
-----------------------
When running Matlab files, there some some functions listed below that must be installed in advance, and adding these installed file paths to the current directory where the Matlab files in *experiments* folder is executed on the local computer is also essential. 

- **dbindex** (Davies Bouldin index): [download here](https://www.mathworks.com/matlabcentral/fileexchange/118685-auto-cvi-tool-an-automatic-cluster-validity-index-toolbox?s_tid=srchtitle) (download all files)

- **UMAP** (uniform manifold approximation and projection): [download here](https://www.mathworks.com/matlabcentral/fileexchange/71902-uniform-manifold-approximation-and-projection-umap )(download all files)



Other Matlab files has been added to the repository for convinence and here are the references: 
- CHI.m (Calinski-Harabasz Criterion)
- [LDA](https://www.mathworks.com/matlabcentral/fileexchange/53151-linear-discriminant-analysis-lda-aka-fisher-discriminant-analysis-fda?s_tid=srchtitle) (Linear Discriminant Analysis): 
- SGPM (Oviedo, 2024)[2]


Data
----
For Mouse Protein analysis
- 'Data_Cortex_Nuclear.csv' is available in the repository. Reference on Github abidlabs/contrastive [here](https://github.com/abidlabs/contrastive/blob/master/experiments/datasets/Data_Cortex_Nuclear.csv).

For Single Cell RNA Sequencing analysis
- 'pbmc_1_counts.csv' and 'pbmc_1_cell_type.csv' both available in the repository 

For Plasma Retinol analysis
- 'Retinol.txt' is available in the repository

For COVID-19 analysis 
- The raw files 'PBMC_COVID.subsample500cells.covid.h5ad' for foreground and 'PBMC_COVID.subsample500cells.ctrl.h5ad' for background is available in [figshare](https://figshare.com/articles/dataset/Precise_disease-state_identification_with_healthy_single-cell_references_-_processed_datasets_and_models/21456645)
- This dataset is referenced [here](https://github.com/MarioniLab/oor_design_reproducibility/tree/master?tab=readme-ov-file)
- The preprocessed files (i.e., covid_preprocessed_fg,csv.zip) are available in the repository. 














Python package details
-----------------------
contrastive-inverse-regression is a python package provided the algorithm for contrastive inverse regression (CIR) for dimension reduction used in a supervised setting to find a low-dimensional representation by solving a nonconvex optimization problem on the Stiefel manifold. 


Installation
------------
Make sure you have numpy, pandas, and scipy install beforehand and the version of these packages compatible with cir. The easy way to install is using ``pip``:

```python

pip install contrastive-inverse-regression

```

Alternatively, you can also install by cloning this repository: 

```python

pip install git+https://github.com/myueen/cir.git

```

Dependencies
------------
- Python (>= 3.10.9)
- numpy (>= 1.24.3)
- pandas (>= 2.1.4)
- scipy (>= 1.9.3)

To run exmaple, matplotlib (>= 3.8.2) is required



<!-- Example
--------
The dataset for the following example is included in the example/dataset folder. 
```python
import contrastive_inverse_regression
from contrastive_inverse_regression import CIR
import pandas as pd

d = 2
alpha = 0.0001

# download the dataset and set it to the absolute path in your computer 
fg = pd.read_csv('../foregroundX.csv')
bg = pd.read_csv('../backgroundX.csv')
Y = pd.read_csv('../foregroundY.csv')
Yt = pd.read_csv('../backgroundY.csv')

fg = fg.iloc[0:, 1:]
bg = bg.iloc[0:, 1:]
Y = Y.iloc[0:, 1:]
Yt = Yt.iloc[0:, 1:]

V = CIR(fg, Y, bg, Yt, alpha, d) -->

```
Other detailed examples for employing CIR are provided. 

For the case of discrete foreground Y values, the mouse protein dataset  Data_Cortex_Nuclear.csv is used and the corresponding visualization in mp_regression.py and regression testing in mp_regression.py.

For the case of continuous foreground Y values, CIR is applied on the retinol dataset Retinol.txt and the corresponding regression is in plasma_regression.py. Continuous values are not usually for classification, hence visualization is not provided. 



Citing contrastive-inverse-regression
---------------------------------------
If you find this algorithm helpful in your research, please add the following bibtex citation in references.
```python

@phdthesis{hawke2023contrastive,
  title={Contrastive inverse regression for dimension reduction},
  author={Hawke, Sam and Luo, Hengrui and Li, Didong},
  journal={arXiv preprint arXiv:2305.12287},
  year={2023}
}
```

References
------------
.. [1] : Hawke, S., Luo, H., & Li, D. (2023)
        "Contrastive Inverse Regression for Dimension Reduction",
        Retrieved from https://arxiv.org/abs/2305.12287 

.. [2] Harry Oviedo (2024).
       SGPM for minimization over the Stiefel Manifold (https://www.mathworks.com/matlabcentral/fileexchange/73505-sgpm-for-minimization-over-the-stiefel-manifold), MATLAB Central File Exchange. Retrieved January 12, 2024.










