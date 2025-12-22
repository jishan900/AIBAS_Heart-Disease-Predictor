import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)

import tensorflow as tf
from tensorflow import keras


# Paths
TRAIN_PATH = Path("/Users/jishan/Documents/AIBAS_Heart-Disease-Predictor/data/Cardiovascular_disease_dataset/training_data.csv")
TEST_PATH  = Path("/Users/jishan/Documents/AIBAS_Heart-Disease-Predictor/data/Cardiovascular_disease_dataset/test_data.csv")

OUT_DIR = Path("/Users/jishan/Documents/AIBAS_Heart-Disease-Predictor/documentation/ANN_Results/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT_DIR / "currentAiSolution.keras" 
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load dataset
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

# Drop columns 
drop_cols = []
for c in ["id", "bmi", "risk_score"]:
    if c in train_df.columns:
        drop_cols.append(c)

if drop_cols:
    train_df = train_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

# Target variable 
if "cardio" not in train_df.columns:
    raise ValueError("Column 'cardio' not found in training_data.csv")

y_train = train_df["cardio"].astype(int).values
y_test  = test_df["cardio"].astype(int).values

# Features
X_train = train_df.drop(columns=["cardio"]).values.astype("float32")
X_test  = test_df.drop(columns=["cardio"]).values.astype("float32")

n_features = X_train.shape[1]

# Build ANN model 
model = keras.Sequential([
    keras.layers.Input(shape=(n_features,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Save history, best checkpoint, and early stop
csv_logger = keras.callbacks.CSVLogger(OUT_DIR / "training_history.csv", append=False)
checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
)
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=50,
    restore_best_weights=True,
)

# Train 
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    callbacks=[csv_logger, checkpoint, early_stop],
    verbose=1,
)

model.save(MODEL_PATH)

# Evaluate 
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4)

final_info = {
    "epochs_trained": int(len(history.history["loss"])),
    "final_train_loss": float(history.history["loss"][-1]),
    "final_train_accuracy": float(history.history["accuracy"][-1]),
    "final_val_loss": float(history.history["val_loss"][-1]),
    "final_val_accuracy": float(history.history["val_accuracy"][-1]),
    "test_accuracy": float(acc),
    "features_used": [c for c in train_df.columns if c != "cardio"],
    "dropped_columns": drop_cols,
    "model_file": str(MODEL_PATH),
}

with open(OUT_DIR / "final_metrics.txt", "w") as f:
    f.write("Final Metrics\n")
    f.write(json.dumps(final_info, indent=2))
    f.write("\n\nClassification Report\n")
    f.write(report)

# Training and Validation curves
plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "curve_train_val_loss.png", dpi=200)
plt.close()

plt.figure()
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "curve_train_val_accuracy.png", dpi=200)
plt.close()

# Confusion matrix
plt.figure()
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0, 1], ["0", "1"])
plt.yticks([0, 1], ["0", "1"])
plt.xlabel("Predicted")
plt.ylabel("True")
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.tight_layout()
plt.savefig(OUT_DIR / "diagnostic_confusion_matrix.png", dpi=200)
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "diagnostic_roc.png", dpi=200)
plt.close()

# Precision-Recall curve
prec, rec, _ = precision_recall_curve(y_test, y_prob)
plt.figure()
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig(OUT_DIR / "diagnostic_pr.png", dpi=200)
plt.close()

# Scatter plot
# Predicted probability vs true label (with small jitter so points are visible)
jitter = (np.random.rand(len(y_test)) - 0.5) * 0.05
plt.figure()
plt.scatter(y_prob, y_test + jitter, s=8)
plt.xlabel("Predicted P(cardio=1)")
plt.ylabel("True label (jittered)")
plt.title("Scatter: Predicted Probability vs True Label")
plt.tight_layout()
plt.savefig(OUT_DIR / "scatter_pred_prob.png", dpi=200)
plt.close()

print("DONE")
print("Model saved as:", MODEL_PATH)
print("Outputs saved in:", OUT_DIR.resolve())
print("Test accuracy:", acc)
print("Dropped columns:", drop_cols)
