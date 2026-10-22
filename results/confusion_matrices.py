import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ============================================================
# CONFIGURATION
# ============================================================

TEST_PATH = "preprocessing/processed/UNSW_NB15_test_processed.csv"

MODEL_DIR = "models/saved"
OUTPUT_DIR = "results/confusion_matrices"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 70)
print("              CONFUSION MATRIX ANALYSIS")
print("=" * 70)

print("\n[1] Loading processed test dataset...")

test_df = pd.read_csv(TEST_PATH)

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

print("Test data:", X_test.shape)


# ============================================================
# MODELS
# ============================================================

models = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}


# ============================================================
# GENERATE CONFUSION MATRICES
# ============================================================

print("\n[2] Generating confusion matrices...")

for model_name, model_file in models.items():

    print("\nProcessing:", model_name)

    model_path = os.path.join(
        MODEL_DIR,
        model_file
    )

    model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("Confusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Normal",
            "Attack"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    display.plot(
        ax=ax,
        values_format="d"
    )

    ax.set_title(
        f"{model_name} - Confusion Matrix"
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        safe_name + "_confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved to:",
        output_path
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("          CONFUSION MATRIX ANALYSIS COMPLETE ✓")
print("=" * 70)

print("\nResults saved in:")
print(OUTPUT_DIR)