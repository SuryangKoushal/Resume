# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('restaurant_sales.csv')

# Step 2: Cleaning Data
## Handling missing values
"""for col in df.columns:
    if col in ['temperature', 'precipitation', 'price_point', 'neighborhood_income', 'local_event_size', 'staff_count', 'marketing_spend']:
        df[col].fillna(df[col].mean(), inplace=True)  # Replace missing numeric values with mean
    elif col in ['day_of_week', 'season']:
        df[col].fillna(df[col].mode()[0], inplace=True)  # Replace missing categorical values with mode"""

# Step 2: Handling outliers using Z-score normalization
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
outliers = (z_scores > 3).any(axis=1)  # Identifies rows where any value exceeds 3 standard deviations
df = df[~outliers]  # Remove outliers
print(f"Outliers removed: {outliers.sum()} rows")

# Step 2: Handling inconsistencies
## Standardize column names (e.g., lowercase, replace spaces with underscores)
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# Step 3: Data Transformation
## Feature Scaling
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('daily_revenue')
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Ensure is_holiday values are lowercase before mapping
#df['is_holiday'] = df['is_holiday'].map({'TRUE': 1, 'FALSE': 0}).astype(int)

## Encode 'is_holiday' to binary values: True->1, False->0
#df['is_holiday'] = df['is_holiday'].map({'TRUE': 1, 'FALSE': 0}).astype(int)
repl = {'TRUE':'1', 'FALSE':'0'}
df['is_holiday'] = df['is_holiday'].replace(repl, regex=True)

## Confirm that day_of_week and season remain as integer types
df['day_of_week'] = df['day_of_week'].astype(int)
df['season'] = df['season'].astype(int)

# Step 4: Data Reduction
## Dimensionality Reduction using PCA
pca = PCA(n_components=0.95)  # Keep 95% of the explained variance
df_reduced = pca.fit_transform(df)
df_reduced = pd.DataFrame(df_reduced, columns=[f'PC{i+1}' for i in range(df_reduced.shape[1])])

# Step 5: Data Splitting
## Separate target and features (assuming 'daily_revenue' as target)
target = 'daily_revenue'
if target in df.columns:
    X = df.drop(columns=[target])
    y = df[target]
else:
    X = df_reduced  # In case of PCA-reduced data
    y = df['daily_revenue'] if 'daily_revenue' in df.columns else None

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save cleaned and prepared data for further use
df_reduced.to_csv('restaurant_sales_cleaned_reduced.csv', index=False)
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("Data preparation completed. Files saved successfully.")

# Configure plot aesthetics
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# 1. Distribution of the target variable (daily_revenue)
plt.figure()
sns.histplot(df['daily_revenue'], kde=True, bins=30)
plt.title('Distribution of Daily Revenue')
plt.xlabel('Daily Revenue')
plt.ylabel('Frequency')
plt.show()

# 2. Pairplot of features to visualize correlations and patterns
sns.pairplot(df, diag_kind='kde', corner=True)
plt.suptitle("Pairplot of Features", y=1.02)
plt.show()

# 3. Correlation heatmap to identify relationships between features
plt.figure()
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()