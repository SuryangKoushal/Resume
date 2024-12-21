import pandas as pd

# Read the original data
df = pd.read_csv('restaurant_sales.csv')

# Create a copy to modify
df_missing = df.copy()

# Set random values to NaN for each feature column (except daily_revenue)
import numpy as np
np.random.seed(42)  # for reproducibility

# List of columns where we want to introduce missing values
columns_to_modify = [col for col in df.columns if col != 'daily_revenue']

# Randomly set ~15% of values to NaN in each column
for col in columns_to_modify:
    mask = np.random.random(len(df)) < 0.15
    df_missing.loc[mask, col] = np.nan

# Save to CSV
df_missing.to_csv('restaurant_sales_with_missing.csv', index=False)

# Verify the result
print("Preview of the data with missing values:")
print(df_missing.head())
print("\nMissing values count per column:")
print(df_missing.isnull().sum())