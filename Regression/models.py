# Import necessary libraries
import pandas as pd
from scipy import stats
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = "restaurant_sales_with_missing.csv"
df = pd.read_csv(file_path)

# Identify and remove outliers using z-scores (> 3 standard deviations)
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
outliers = (z_scores > 3).any(axis=1)
df_no_outliers = df[~outliers]

# Standardize numeric features (excluding the target variable)
numeric_cols = df_no_outliers.select_dtypes(include=[np.number]).columns.drop('daily_revenue')
scaler = StandardScaler()
df_no_outliers[numeric_cols] = scaler.fit_transform(df_no_outliers[numeric_cols])

# Split features and target
X = df_no_outliers.drop(columns=['daily_revenue'])
y = df_no_outliers['daily_revenue']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models with optimized settings
models_optimized = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "XGBoost Regressor": XGBRegressor(objective='reg:squarederror', n_estimators=50, random_state=42, tree_method = 'exact')
}

# Store results for comparison
results_optimized = {}

# Train and evaluate each model
for name, model in models_optimized.items():
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Make predictions
    mse = mean_squared_error(y_test, y_pred)  # Calculate MSE
    r2 = r2_score(y_test, y_pred)  # Calculate R² score
    results_optimized[name] = {"MSE": mse, "R2": r2}  # Store results

# Display the results
for model_name, metrics in results_optimized.items():
    print(f"{model_name} - MSE: {metrics['MSE']:.2f}, R²: {metrics['R2']:.3f}")
