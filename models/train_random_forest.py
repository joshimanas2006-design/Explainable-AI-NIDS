import os
import time
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import joblib


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = "preprocessing/processed/UNSW_NB15_train_processed.csv"
TEST_PATH = "preprocessing/processed/UNSW_NB15_test_processed.csv"

MODEL_DIR = "models/saved"
REPORT_DIR = "reports"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "random_forest.pkl"
)

REPORT_PATH = os.path.join(
    REPORT_DIR,
    "random_forest_report.txt"
)


# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================

print("=" * 70)
print("             RANDOM FOREST - NIDS")
print("=" * 70)

print("\n[1] Loading processed datasets...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Training data:", train_df.shape)
print("Testing data :", test_df.shape)


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

print("\n[2] Separating features and target...")

X_train = train_df.drop(columns=["label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# ============================================================
# 3. CREATE RANDOM FOREST MODEL
# ============================================================

print("\n[3] Creating Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 4. TRAIN MODEL
# ============================================================

print("\n[4] Training model...")
print("This may take some time...")

start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print(
    f"Training completed in "
    f"{training_time:.2f} seconds."
)


# ============================================================
# 5. MAKE PREDICTIONS
# ============================================================

print("\n[5] Making predictions...")

y_pred = model.predict(X_test)

# Probability of Attack (class 1)
y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# 6. CALCULATE EVALUATION METRICS
# ============================================================

print("\n[6] Calculating evaluation metrics...")

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("             RANDOM FOREST RESULTS")
print("=" * 70)

print(
    f"\nAccuracy  : {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Precision : {precision:.4f} "
    f"({precision * 100:.2f}%)"
)

print(
    f"Recall    : {recall:.4f} "
    f"({recall * 100:.2f}%)"
)

print(
    f"F1-Score  : {f1:.4f} "
    f"({f1 * 100:.2f}%)"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "Attack"
        ],
        zero_division=0
    )
)


# ============================================================
# 8. SAVE MODEL
# ============================================================

print("\n[7] Saving trained model...")

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)

print("Model saved to:")
print(MODEL_PATH)


# ============================================================
# 9. SAVE REPORT
# ============================================================

print("\n[8] Saving evaluation report...")

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "RANDOM FOREST - NIDS\n"
    )

    file.write(
        "=" * 50 + "\n\n"
    )

    file.write(
        f"Training samples: {len(X_train)}\n"
    )

    file.write(
        f"Testing samples: {len(X_test)}\n"
    )

    file.write(
        f"Processed features: {X_train.shape[1]}\n\n"
    )

    file.write(
        f"Number of trees: "
        f"{model.n_estimators}\n"
    )

    file.write(
        f"Training time: "
        f"{training_time:.2f} seconds\n\n"
    )

    file.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Precision: {precision:.4f}\n"
    )

    file.write(
        f"Recall: {recall:.4f}\n"
    )

    file.write(
        f"F1-Score: {f1:.4f}\n"
    )

    file.write(
        f"ROC-AUC: {roc_auc:.4f}\n\n"
    )

    file.write(
        "Confusion Matrix:\n"
    )

    file.write(
        str(cm) + "\n\n"
    )

    file.write(
        "Classification Report:\n"
    )

    file.write(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Normal",
                "Attack"
            ],
            zero_division=0
        )
    )


print("\nReport saved to:")
print(REPORT_PATH)


# ============================================================
# 10. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("             RANDOM FOREST COMPLETE ✓")
print("=" * 70)