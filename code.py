import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Define features
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
target = 'SalePrice'

# Check for missing columns in test data
missing_cols = set(features) - set(test_data.columns)
if missing_cols:
    print(f"Warning: The following columns are missing in the test data: {missing_cols}")
    for col in missing_cols:
        test_data[col] = 0  # Fill missing columns with 0

# Select features
X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Visualizations
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_train_pred, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Predicted vs Actual House Prices (Training Data)')
plt.tight_layout()
plt.show()

# Residual Plot
residuals = y_train - y_train_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_train_pred, residuals, alpha=0.5)
plt.plot([y_train_pred.min(), y_train_pred.max()], [0, 0], 'r--', lw=2)
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.tight_layout()
plt.show()

# Feature Importance Plot
feature_importance = pd.DataFrame({'feature': features, 'importance': abs(model.coef_)})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance')
plt.xlabel('Absolute Coefficient Value')
plt.tight_layout()
plt.show()

# Correlation Heatmap
correlation_matrix = train_data[features + [target]].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Print model performance metrics
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

print(f"Training RMSE: ${train_rmse:.2f}")
print(f"Training R-squared: {train_r2:.4f}")

# Create submission file
submission = pd.DataFrame({
    'Id': test_data.Id,
    'SalePrice': y_test_pred
})
submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")