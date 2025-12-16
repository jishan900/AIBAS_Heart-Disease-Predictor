import pandas as pd
from pathlib import Path

DATA_DIR = Path("/Users/jishan/Documents/AIBAS_Heart-Disease-Predictor/data/Cardiovascular_disease_dataset")
JOINT_PATH = DATA_DIR / "joint_data_collection.csv"
SEED = 42

if not JOINT_PATH.exists():
    raise FileNotFoundError(f"Missing: {JOINT_PATH}")

df = pd.read_csv(JOINT_PATH, sep=";")

print("Missing values per column (before cleaning):")
print(df.isna().sum())
print("Total missing values:", df.isna().sum().sum())
print("-" * 50)

df = df.drop_duplicates().dropna()

target_col = "cardio" if "cardio" in df.columns else None

numeric_cols = df.select_dtypes(include="number").columns.tolist()
feature_numeric_cols = [c for c in numeric_cols if c != target_col]


df_clean = df.copy()

for col in feature_numeric_cols:
    q1 = df_clean[col].quantile(0.25)
    q3 = df_clean[col].quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        continue
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    df_clean = df_clean[(df_clean[col] >= low) & (df_clean[col] <= high)]


for col in feature_numeric_cols:
    mn = df_clean[col].min()
    mx = df_clean[col].max()
    if mx == mn:
        df_clean[col] = 0.0
    else:
        df_clean[col] = (df_clean[col] - mn) / (mx - mn)


df_clean["bmi"] = df_clean["weight"] / ((df_clean["height"] + 1e-9) ** 2)

# Normalize bmi too
bmi_min = df_clean["bmi"].min()
bmi_max = df_clean["bmi"].max()
if bmi_max == bmi_min:
    df_clean["bmi"] = 0.0
else:
    df_clean["bmi"] = (df_clean["bmi"] - bmi_min) / (bmi_max - bmi_min)


df_clean["risk_score"] = (
    0.25 * df_clean["age"] +
    0.25 * df_clean["ap_hi"] +
    0.15 * df_clean["cholesterol"] +
    0.10 * df_clean["gluc"] +
    0.10 * df_clean["bmi"] +
    0.05 * df_clean["smoke"] +
    0.05 * df_clean["alco"] -
    0.05 * df_clean["active"]
).clip(0, 1)

print("Missing values per column (after cleaning):")
print(df_clean.isna().sum())
print("Total missing values after cleaning:", df_clean.isna().sum().sum())
print("-" * 50)


df_clean.to_csv(DATA_DIR / "joint_data_collection.csv", index=False)

# Split dataset (80:20)
df_shuffled = df_clean.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
split_idx = int(0.8 * len(df_shuffled))

train_df = df_shuffled.iloc[:split_idx].copy()
test_df = df_shuffled.iloc[split_idx:].copy()

train_df.to_csv(DATA_DIR / "training_data.csv", index=False)
test_df.to_csv(DATA_DIR / "test_data.csv", index=False)

# Activation data (1 row from test)
activation_df = test_df.sample(n=1, random_state=SEED)
activation_df.to_csv(DATA_DIR / "activation_data.csv", index=False)

print("Created:")
print("joint_data_collection.csv (cleaned + normalized + risk_score)")
print("training_data.csv")
print("test_data.csv")
print("activation_data.csv")
print(f"Rows: joint={len(df_clean)}, train={len(train_df)}, test={len(test_df)}")
print("Targets available:")
print(" - Cardio (0/1) for classification")
print(" - Risk_score (0-1) for OLS regression")
