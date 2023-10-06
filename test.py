import numpy as np
import pandas as pd

# Given data
X = np.array([[10, 18, 5], [-8, 11, 12], [2, 17, 2]])
Y = np.array([1, 13, 20])
H = 3

# Center the data
X_centered = X - X.mean(axis=0)

# Define intervals
interval = np.linspace(Y.min(), Y.max(), num=H+1)

# Use digitize to find which interval each Y value belongs to
# Subtract 1 to match zero-based indexing
interval_indices = np.digitize(Y, interval) - 1

# Create an array to store the mask for each interval
masks = [(interval_indices == i) for i in range(H)]

# Calculate mh using vectorized operations
mh = np.array([X_centered[mask].mean(axis=0) if np.any(mask)
              else np.full(X.shape[1], np.nan) for mask in masks])

# Create a DataFrame for mh
mh_df = pd.DataFrame(mh)

print(mh_df)
