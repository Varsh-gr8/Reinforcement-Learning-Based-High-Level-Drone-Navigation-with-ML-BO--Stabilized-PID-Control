import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Path to your CSV file
csv_path = r"C:\Users\varsh\OneDrive\Documents\Drone_navi\data\pid_data.csv"

# Load CSV data
df = pd.read_csv(csv_path)

# List of features to check for multicollinearity
features = [
    "x", "y", "z", "Roll", "Pitch", "Yaw",
    "vx", "vy", "vz", "wx", "wy", "wz",
    "Target_x", "Target_y", "Target_z", "Error_Dist"
]

# Calculate the correlation matrix
corr_matrix = df[features].corr()

# Print the correlation matrix
print("Correlation Matrix:")
print(corr_matrix)

# Plot the correlation matrix using seaborn heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Feature Correlation Matrix")
plt.show()
pca = PCA(n_components=1)
combined_feature = pca.fit_transform(df[["vx", "vy"]])
df["v_xy"] = combined_feature

# You might then drop vx and vy:
df_reduced = df.drop(columns=["vx", "vy"])
print("New features after PCA combination:")
print(df_reduced.head())