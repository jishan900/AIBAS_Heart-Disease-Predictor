import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
TRAIN_PATH = Path("data/Cardiovascular_disease_dataset/training_data.csv")
TEST_PATH = Path("data/Cardiovascular_disease_dataset/test_data.csv")

OUT_DIR = Path("documentation/OLS_Results/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT_DIR / "currentOlsSolution.pkl"
SEED = 42
np.random.seed(SEED)

print("Loading datasets...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# For OLS regression, we'll use the continuous risk_score as target variable
# This provides a more appropriate continuous target for linear regression
if "risk_score" not in train_df.columns:
    raise ValueError("Column 'risk_score' not found. Please ensure data preprocessing created this column.")

# Drop columns that shouldn't be used as features
drop_cols = []
for c in ["id", "bmi", "cardio"]:  # Keep risk_score as target, drop cardio (binary)
    if c in train_df.columns:
        drop_cols.append(c)

if drop_cols:
    train_df = train_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

# Target variable (continuous risk score)
y_train = train_df["risk_score"].values
y_test = test_df["risk_score"].values

# Features
feature_cols = [c for c in train_df.columns if c != "risk_score"]
X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values

# Add constant term for intercept
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features used: {feature_cols}")
print(f"Target range - Train: [{y_train.min():.3f}, {y_train.max():.3f}]")
print(f"Target range - Test: [{y_test.min():.3f}, {y_test.max():.3f}]")

# Fit OLS model
print("\nFitting OLS model...")
ols_model = sm.OLS(y_train, X_train_const).fit()

# Save model
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(ols_model, f)

print(f"Model saved to: {MODEL_PATH}")

# Predictions
y_train_pred = ols_model.predict(X_train_const)
y_test_pred = ols_model.predict(X_test_const)

# Calculate performance metrics
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Residuals for diagnostics
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

# Statistical tests
durbin_watson_stat = durbin_watson(train_residuals)

# Heteroscedasticity tests
try:
    bp_stat, bp_pvalue, _, _ = het_breuschpagan(train_residuals, X_train_const)
    white_stat, white_pvalue, _, _ = het_white(train_residuals, X_train_const)
except:
    bp_stat = bp_pvalue = white_stat = white_pvalue = np.nan

# Compile results
results = {
    "model_summary": str(ols_model.summary()),
    "features_used": feature_cols,
    "dropped_columns": drop_cols,
    "training_metrics": {
        "mse": float(train_mse),
        "rmse": float(train_rmse),
        "mae": float(train_mae),
        "r2": float(train_r2),
        "adj_r2": float(ols_model.rsquared_adj)
    },
    "test_metrics": {
        "mse": float(test_mse),
        "rmse": float(test_rmse),
        "mae": float(test_mae),
        "r2": float(test_r2)
    },
    "diagnostic_tests": {
        "durbin_watson": float(durbin_watson_stat),
        "breusch_pagan_stat": float(bp_stat) if not np.isnan(bp_stat) else None,
        "breusch_pagan_pvalue": float(bp_pvalue) if not np.isnan(bp_pvalue) else None,
        "white_test_stat": float(white_stat) if not np.isnan(white_stat) else None,
        "white_test_pvalue": float(white_pvalue) if not np.isnan(white_pvalue) else None
    },
    "model_file": str(MODEL_PATH)
}

# Save results
with open(OUT_DIR / "ols_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Save detailed summary
with open(OUT_DIR / "ols_summary.txt", "w") as f:
    f.write("OLS Regression Results\n")
    f.write("=" * 50 + "\n\n")
    f.write(str(ols_model.summary()))
    f.write("\n\n" + "=" * 50 + "\n")
    f.write("Performance Metrics\n")
    f.write("=" * 50 + "\n")
    f.write(f"Training R²: {train_r2:.4f}\n")
    f.write(f"Training Adj R²: {ols_model.rsquared_adj:.4f}\n")
    f.write(f"Training RMSE: {train_rmse:.4f}\n")
    f.write(f"Training MAE: {train_mae:.4f}\n\n")
    f.write(f"Test R²: {test_r2:.4f}\n")
    f.write(f"Test RMSE: {test_rmse:.4f}\n")
    f.write(f"Test MAE: {test_mae:.4f}\n\n")
    f.write("Diagnostic Tests\n")
    f.write("=" * 20 + "\n")
    f.write(f"Durbin-Watson: {durbin_watson_stat:.4f} (2.0 = no autocorrelation)\n")
    if not np.isnan(bp_pvalue):
        f.write(f"Breusch-Pagan p-value: {bp_pvalue:.4f} (>0.05 = homoscedastic)\n")
    if not np.isnan(white_pvalue):
        f.write(f"White test p-value: {white_pvalue:.4f} (>0.05 = homoscedastic)\n")

print("\n" + "="*50)
print("OLS MODEL RESULTS")
print("="*50)
print(f"Training R²: {train_r2:.4f}")
print(f"Training Adj R²: {ols_model.rsquared_adj:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Durbin-Watson: {durbin_watson_stat:.4f}")

# Create visualizations
plt.style.use('default')
fig_size = (12, 10)

# 1. Diagnostic Plots (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=fig_size)

# Residuals vs Fitted
axes[0,0].scatter(y_train_pred, train_residuals, alpha=0.6, s=20)
axes[0,0].axhline(y=0, color='red', linestyle='--')
axes[0,0].set_xlabel('Fitted Values')
axes[0,0].set_ylabel('Residuals')
axes[0,0].set_title('Residuals vs Fitted Values')
axes[0,0].grid(True, alpha=0.3)

# Q-Q Plot for normality
stats.probplot(train_residuals, dist="norm", plot=axes[0,1])
axes[0,1].set_title('Q-Q Plot (Normality Check)')
axes[0,1].grid(True, alpha=0.3)

# Scale-Location Plot (sqrt of standardized residuals vs fitted)
standardized_residuals = train_residuals / np.std(train_residuals)
sqrt_abs_residuals = np.sqrt(np.abs(standardized_residuals))
axes[1,0].scatter(y_train_pred, sqrt_abs_residuals, alpha=0.6, s=20)
axes[1,0].set_xlabel('Fitted Values')
axes[1,0].set_ylabel('√|Standardized Residuals|')
axes[1,0].set_title('Scale-Location Plot')
axes[1,0].grid(True, alpha=0.3)

# Residuals vs Leverage (Cook's distance)
influence = ols_model.get_influence()
leverage = influence.hat_matrix_diag
axes[1,1].scatter(leverage, standardized_residuals, alpha=0.6, s=20)
axes[1,1].axhline(y=0, color='red', linestyle='--')
axes[1,1].set_xlabel('Leverage')
axes[1,1].set_ylabel('Standardized Residuals')
axes[1,1].set_title('Residuals vs Leverage')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "diagnostic_plots.png", dpi=200, bbox_inches='tight')
plt.close()

# 2. Scatter Plots (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=fig_size)

# Training: Actual vs Predicted
axes[0,0].scatter(y_train, y_train_pred, alpha=0.6, s=20)
axes[0,0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0,0].set_xlabel('Actual Risk Score')
axes[0,0].set_ylabel('Predicted Risk Score')
axes[0,0].set_title(f'Training: Actual vs Predicted (R² = {train_r2:.3f})')
axes[0,0].grid(True, alpha=0.3)

# Test: Actual vs Predicted
axes[0,1].scatter(y_test, y_test_pred, alpha=0.6, s=20, color='orange')
axes[0,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0,1].set_xlabel('Actual Risk Score')
axes[0,1].set_ylabel('Predicted Risk Score')
axes[0,1].set_title(f'Test: Actual vs Predicted (R² = {test_r2:.3f})')
axes[0,1].grid(True, alpha=0.3)

# Training Residuals Distribution
axes[1,0].hist(train_residuals, bins=50, alpha=0.7, density=True, color='skyblue')
axes[1,0].set_xlabel('Residuals')
axes[1,0].set_ylabel('Density')
axes[1,0].set_title('Training Residuals Distribution')
axes[1,0].grid(True, alpha=0.3)

# Test Residuals Distribution
axes[1,1].hist(test_residuals, bins=50, alpha=0.7, density=True, color='lightcoral')
axes[1,1].set_xlabel('Residuals')
axes[1,1].set_ylabel('Density')
axes[1,1].set_title('Test Residuals Distribution')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "scatter_plots.png", dpi=200, bbox_inches='tight')
plt.close()

# 3. Feature Importance Plot
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': ols_model.params[1:],  # Exclude intercept
    'p_value': ols_model.pvalues[1:],
    'abs_coefficient': np.abs(ols_model.params[1:])
}).sort_values('abs_coefficient', ascending=True)

plt.figure(figsize=(10, 8))
colors = ['red' if p < 0.05 else 'gray' for p in feature_importance['p_value']]
plt.barh(range(len(feature_importance)), feature_importance['coefficient'], color=colors)
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.xlabel('Coefficient Value')
plt.title('OLS Feature Coefficients (Red = Significant p<0.05)')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "feature_importance.png", dpi=200, bbox_inches='tight')
plt.close()

print(f"\nAll outputs saved to: {OUT_DIR.resolve()}")
print("\nFiles created:")
print("- currentOlsSolution.pkl (trained model)")
print("- ols_results.json (metrics)")
print("- ols_summary.txt (detailed summary)")
print("- diagnostic_plots.png (4 diagnostic plots)")
print("- scatter_plots.png (actual vs predicted + residuals)")
print("- feature_importance.png (coefficient plot)")

print("\nOLS model training completed successfully!")