cir
======
cir is a python package provided the algorithm for contrastive inverse regression (CIR) for dimension reduction used in a supervised setting to find a low-dimensional representation by solving a nonconvex optimization problem on the Stiefel manifold. 


Example
--------
The dataset for the following example is included in the datasets_example folder. 
```python
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

    V = CIR(fg, Y, bg, Yt, alpha, d)

```
Other detailed examples for employing cir are provided. 

For the case of discrete foreground Y values, the mouse protein dataset  Data_Cortex_Nuclear.csv is used and the corresponding visualization in mp_regression.py and regression testing in mp_regression.py.

For the case of continuous foreground Y values, cir is applied on the retinol dataset Retinol.txt and the corresponding regression is in plasma_regression.py. Continuous values are not usually for classification, hence visualization is not provided. 


Dependencies
------------
- Python (>= 3.10.9)
- numpy (>= 1.24.3)
- pandas (>= 2.1.4)
- scipy (>= 1.9.3)

To run exmaple, matplotlib (>= 3.8.2) is required


References:
------------
.. [1] : Hawke, S., Luo, H., & Li, D. (2023)
        "Contrastive Inverse Regression for Dimension Reduction",
        Retrieved from https://arxiv.org/abs/2305.12287 

.. [2] Harry Oviedo (2024).
       SGPM for minimization over the Stiefel Manifold (https://www.mathworks.com/matlabcentral/fileexchange/73505-sgpm-for-minimization-over-the-stiefel-manifold), MATLAB Central File Exchange. Retrieved January 12, 2024.


Dataset Source: 
---------------

```python
    @article{abid2018exploring,
    title={Exploring patterns enriched in a dataset with contrastive principal component analysis},
    author={Abid, Abubakar and Zhang, Martin J and Bagaria, Vivek K and Zou, James},
    journal={Nature communications},
    volume={9},
    number={1},
    pages={2134},
    year={2018},
    }

```




