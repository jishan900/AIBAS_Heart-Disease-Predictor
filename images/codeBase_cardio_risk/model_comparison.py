import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import tensorflow as tf

# Paths
OLS_RESULTS_PATH = Path("documentation/OLS_Results/ols_results.json")
ANN_RESULTS_PATH = Path("documentation/ANN_Results/final_metrics.txt")
ANN_MODEL_PATH = Path("documentation/ANN_Results/currentAiSolution.keras")
OLS_MODEL_PATH = Path("documentation/OLS_Results/currentOlsSolution.pkl")

TEST_PATH = Path("data/Cardiovascular_disease_dataset/test_data.csv")
OUT_DIR = Path("documentation/Model_Comparison/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading test data and models...")
test_df = pd.read_csv(TEST_PATH)

# Load OLS results
with open(OLS_RESULTS_PATH, 'r') as f:
    ols_results = json.load(f)

# Load ANN results
with open(ANN_RESULTS_PATH, 'r') as f:
    ann_content = f.read()
    # Extract JSON part
    json_start = ann_content.find('{')
    json_end = ann_content.find('}', json_start) + 1
    ann_results = json.loads(ann_content[json_start:json_end])

# Load OLS model for predictions
with open(OLS_MODEL_PATH, 'rb') as f:
    ols_model = pickle.load(f)

# Prepare test data
# For OLS: use risk_score as target
# For ANN: use cardio as target

# OLS prediction (continuous risk score)
drop_cols_ols = ["id", "bmi", "cardio"] if "id" in test_df.columns else ["bmi", "cardio"]
test_df_ols = test_df.drop(columns=[c for c in drop_cols_ols if c in test_df.columns])
y_test_ols = test_df_ols["risk_score"].values
X_test_ols = test_df_ols.drop(columns=["risk_score"]).values

# Add constant for OLS
import statsmodels.api as sm
X_test_ols_const = sm.add_constant(X_test_ols)
y_pred_ols = ols_model.predict(X_test_ols_const)

# ANN prediction (use existing results from final_metrics.txt)
# We'll use the reported accuracy instead of recomputing
ann_accuracy = ann_results["test_accuracy"]

# Calculate metrics
ols_r2 = r2_score(y_test_ols, y_pred_ols)
ols_rmse = np.sqrt(mean_squared_error(y_test_ols, y_pred_ols))

# Create comparison report
comparison_results = {
    "model_comparison": {
        "OLS_Regression": {
            "task": "Continuous risk score prediction (0-1)",
            "target_variable": "risk_score",
            "model_type": "Linear Regression",
            "test_r2": float(ols_r2),
            "test_rmse": float(ols_rmse),
            "training_r2": ols_results["training_metrics"]["r2"],
            "adj_r2": ols_results["training_metrics"]["adj_r2"],
            "features_count": len(ols_results["features_used"]),
            "interpretability": "High - coefficients show direct feature impact"
        },
        "ANN_Classification": {
            "task": "Binary heart disease classification (0/1)",
            "target_variable": "cardio",
            "model_type": "Artificial Neural Network",
            "test_accuracy": float(ann_accuracy),
            "training_accuracy": ann_results["final_train_accuracy"],
            "validation_accuracy": ann_results["final_val_accuracy"],
            "features_count": len(ann_results["features_used"]),
            "interpretability": "Low - black box model"
        }
    },
    "key_differences": {
        "target_variables": "OLS predicts continuous risk (0-1), ANN predicts binary disease (0/1)",
        "performance_metrics": "OLS uses R²/RMSE, ANN uses accuracy/precision/recall",
        "model_complexity": "OLS is linear and interpretable, ANN is non-linear and complex",
        "overfitting_risk": "OLS shows potential overfitting (R²=0.9999), ANN more reasonable (71% accuracy)"
    },
    "recommendations": {
        "use_ols_when": "Need interpretable model, continuous risk assessment, feature importance analysis",
        "use_ann_when": "Need binary classification, can accept black box, have large datasets",
        "concerns": "OLS R² of 0.9999 suggests overfitting or data leakage - investigate risk_score calculation"
    }
}

# Save comparison results
with open(OUT_DIR / "model_comparison.json", "w") as f:
    json.dump(comparison_results, f, indent=2)

# Create comparison report
report_content = f"""
MODEL COMPARISON REPORT
======================

TASK COMPARISON:
- OLS Model: Predicts continuous cardiovascular risk score (0-1 scale)
- ANN Model: Predicts binary heart disease presence (0/1 classification)

PERFORMANCE SUMMARY:
===================

OLS Regression Results:
- Target: Continuous risk_score
- Test R²: {ols_r2:.4f}
- Test RMSE: {ols_rmse:.4f}
- Training R²: {ols_results['training_metrics']['r2']:.4f}
- Adjusted R²: {ols_results['training_metrics']['adj_r2']:.4f}

ANN Classification Results:
- Target: Binary cardio disease
- Test Accuracy: {ann_accuracy:.4f} ({ann_accuracy*100:.1f}%)
- Training Accuracy: {ann_results['final_train_accuracy']:.4f}
- Validation Accuracy: {ann_results['final_val_accuracy']:.4f}

FEATURE USAGE:
=============
Both models use {len(ols_results['features_used'])} features:
{', '.join(ols_results['features_used'])}

MODEL CHARACTERISTICS:
=====================

OLS Strengths:
+ Highly interpretable coefficients
+ Fast training and prediction
+ Statistical significance testing
+ Perfect fit to risk score (R² ≈ 1.0)

OLS Concerns:
- Extremely high R^2 (0.9999) suggests potential overfitting
- May have data leakage in risk_score calculation
- Linear assumptions may not hold

ANN Strengths:
+ Handles non-linear relationships
+ Good generalization (71% accuracy)
+ Robust to feature interactions
+ More realistic performance metrics

ANN Limitations:
- Black box model (low interpretability)
- Requires more computational resources
- Prone to overfitting with small datasets

RECOMMENDATIONS:
===============
1. Investigate OLS overfitting - R^2 of 0.9999 is suspiciously high
2. Check risk_score calculation for potential data leakage
3. Consider regularized regression (Ridge/Lasso) for OLS
4. Use OLS for interpretable risk assessment
5. Use ANN for binary diagnostic decisions
6. Both models complement each other for comprehensive analysis

DIAGNOSTIC INSIGHTS:
===================
- OLS Durbin-Watson: {ols_results['diagnostic_tests']['durbin_watson']:.4f} (close to 2.0 = good)
- Both models use same preprocessed features
- Different targets make direct comparison challenging
- Consider ensemble approach combining both predictions
"""

with open(OUT_DIR / "comparison_report.txt", "w", encoding='utf-8') as f:
    f.write(report_content)

# Create visualization comparing predictions
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# OLS: Actual vs Predicted Risk Score
axes[0,0].scatter(y_test_ols, y_pred_ols, alpha=0.6, s=20)
axes[0,0].plot([y_test_ols.min(), y_test_ols.max()], [y_test_ols.min(), y_test_ols.max()], 'r--', lw=2)
axes[0,0].set_xlabel('Actual Risk Score')
axes[0,0].set_ylabel('Predicted Risk Score')
axes[0,0].set_title(f'OLS: Risk Score Prediction (R² = {ols_r2:.4f})')
axes[0,0].grid(True, alpha=0.3)

# ANN: Accuracy Bar (using reported results)
ann_train_acc = ann_results["final_train_accuracy"]
ann_val_acc = ann_results["final_val_accuracy"]

x_pos = [0, 1, 2]
accuracies = [ann_train_acc, ann_val_acc, ann_accuracy]
labels = ['ANN Train', 'ANN Val', 'ANN Test']

bars = axes[0,1].bar(x_pos, accuracies, color=['lightblue', 'lightgreen', 'lightcoral'])
axes[0,1].set_ylabel('Accuracy')
axes[0,1].set_title('ANN Model Performance')
axes[0,1].set_xticks(x_pos)
axes[0,1].set_xticklabels(labels)
axes[0,1].set_ylim(0, 1)
axes[0,1].grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, accuracies):
    axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{value:.3f}', ha='center', va='bottom')

# Risk Score vs Cardio Relationship
axes[1,0].boxplot([test_df[test_df['cardio']==0]['risk_score'], 
                   test_df[test_df['cardio']==1]['risk_score']], 
                  labels=['No Disease', 'Disease'])
axes[1,0].set_ylabel('Risk Score')
axes[1,0].set_title('Risk Score Distribution by Cardio Disease')
axes[1,0].grid(True, alpha=0.3)

# Model Performance Comparison (Bar Chart)
metrics = ['OLS R²', 'ANN Accuracy']
values = [ols_r2, ann_accuracy]
colors = ['skyblue', 'lightcoral']

bars = axes[1,1].bar(metrics, values, color=colors)
axes[1,1].set_ylabel('Performance Score')
axes[1,1].set_title('Model Performance Comparison')
axes[1,1].set_ylim(0, 1)
axes[1,1].grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, values):
    axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{value:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(OUT_DIR / "model_comparison_plots.png", dpi=200, bbox_inches='tight')
plt.close()

print("\n" + "="*60)
print("MODEL COMPARISON COMPLETED")
print("="*60)
print(f"OLS R² Score: {ols_r2:.4f}")
print(f"ANN Accuracy: {ann_accuracy:.4f} ({ann_accuracy*100:.1f}%)")
print(f"\nFiles created in {OUT_DIR}:")
print("- model_comparison.json")
print("- comparison_report.txt") 
print("- model_comparison_plots.png")
print("\nNote: OLS R² of 0.9999 suggests potential overfitting - investigate risk_score calculation")