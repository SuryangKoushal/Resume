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
file_path1 = "X_train.csv"
X_train = pd.read_csv(file_path1)
file_path2 = "y_train.csv"
y_train = pd.read_csv(file_path2)
file_path3 = "X_test.csv"
X_test = pd.read_csv(file_path3)
file_path4 = "y_test.csv"
y_test = pd.read_csv(file_path4)

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
    model.fit(X_train, y_train)  # Train the model using the provided X_train and y_train
    y_pred = model.predict(X_test)  # Make predictions on the provided X_test
    mse = mean_squared_error(y_test, y_pred)  # Calculate MSE
    r2 = r2_score(y_test, y_pred)  # Calculate R² score
    results_optimized[name] = {"MSE": mse, "R2": r2}  # Store results

# Display the results
for model_name, metrics in results_optimized.items():
    print(f"{model_name} - MSE: {metrics['MSE']:.2f}, R²: {metrics['R2']:.3f}")
