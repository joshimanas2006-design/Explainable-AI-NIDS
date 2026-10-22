import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/saved/xgboost.pkl"

TEST_PATH = (
    "preprocessing/processed/"
    "UNSW_NB15_test_processed.csv"
)

OUTPUT_DIR = "xai/results"

SUMMARY_BAR_PATH = os.path.join(
    OUTPUT_DIR,
    "shap_feature_importance_bar.png"
)

SUMMARY_DOT_PATH = os.path.join(
    OUTPUT_DIR,
    "shap_summary.png"
)

INDIVIDUAL_PATH = os.path.join(
    OUTPUT_DIR,
    "shap_individual_prediction.png"
)

FEATURE_IMPORTANCE_PATH = os.path.join(
    OUTPUT_DIR,
    "feature_importance.csv"
)

EXPLANATION_PATH = os.path.join(
    OUTPUT_DIR,
    "individual_explanation.txt"
)


# ============================================================
# 1. LOAD MODEL
# ============================================================

print("=" * 70)
print("             SHAP - EXPLAINABLE AI ANALYSIS")
print("=" * 70)

print("\n[1] Loading trained XGBoost model...")

model = joblib.load(MODEL_PATH)

print("XGBoost model loaded successfully.")


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print("\n[2] Loading processed test dataset...")

test_df = pd.read_csv(TEST_PATH)

X_test = test_df.drop(columns=["label"])
y_test = test_df["label"]

print("Test data:", X_test.shape)


# ============================================================
# 3. SAMPLE TEST DATA FOR SHAP
# ============================================================

print("\n[3] Preparing SHAP sample...")

# SHAP can be computationally expensive on the entire
# 82,332-record test set.
#
# We therefore use a representative random sample.

SAMPLE_SIZE = 2000

sample_size = min(
    SAMPLE_SIZE,
    len(X_test)
)

X_sample = X_test.sample(
    n=sample_size,
    random_state=42
)

y_sample = y_test.loc[X_sample.index]

print(
    f"Using {len(X_sample):,} test records "
    "for SHAP analysis."
)


# ============================================================
# 4. CREATE SHAP EXPLAINER
# ============================================================

print("\n[4] Creating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(model)

print("SHAP TreeExplainer created successfully.")


# ============================================================
# 5. CALCULATE SHAP VALUES
# ============================================================

print("\n[5] Calculating SHAP values...")
print("This may take some time...")

shap_values = explainer.shap_values(X_sample)

print("SHAP values calculated successfully.")


# ============================================================
# 6. HANDLE SHAP OUTPUT FORMAT
# ============================================================

# For binary XGBoost classification, SHAP normally returns
# a matrix with:
#
# rows    = samples
# columns = features
#
# Newer SHAP versions may return an Explanation object
# depending on the model/output configuration.

if isinstance(shap_values, list):
    shap_matrix = shap_values[-1]
else:
    shap_matrix = shap_values

if hasattr(shap_matrix, "values"):
    shap_matrix = shap_matrix.values

shap_matrix = np.asarray(shap_matrix)

print(
    "SHAP matrix shape:",
    shap_matrix.shape
)


# ============================================================
# 7. GLOBAL FEATURE IMPORTANCE
# ============================================================

print("\n[6] Calculating global feature importance...")

mean_abs_shap = np.abs(
    shap_matrix
).mean(axis=0)

feature_importance = pd.DataFrame(
    {
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs_shap
    }
)

feature_importance = feature_importance.sort_values(
    by="mean_abs_shap",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 8. DISPLAY TOP FEATURES
# ============================================================

print("\n" + "=" * 70)
print("          TOP 20 IMPORTANT FEATURES")
print("=" * 70)

print(
    feature_importance.head(20).to_string(
        index=False
    )
)


# ============================================================
# 9. SAVE FEATURE IMPORTANCE
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

feature_importance.to_csv(
    FEATURE_IMPORTANCE_PATH,
    index=False
)

print(
    "\nFeature importance saved to:"
)

print(FEATURE_IMPORTANCE_PATH)


# ============================================================
# 10. SHAP BAR PLOT
# ============================================================

print("\n[7] Creating SHAP bar plot...")

plt.figure()

shap.summary_plot(
    shap_matrix,
    X_sample,
    plot_type="bar",
    show=False,
    max_display=20
)

plt.tight_layout()

plt.savefig(
    SUMMARY_BAR_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "SHAP bar plot saved to:"
)

print(SUMMARY_BAR_PATH)


# ============================================================
# 11. SHAP SUMMARY / BEESWARM PLOT
# ============================================================

print("\n[8] Creating SHAP summary plot...")

plt.figure()

shap.summary_plot(
    shap_matrix,
    X_sample,
    show=False,
    max_display=20
)

plt.tight_layout()

plt.savefig(
    SUMMARY_DOT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "SHAP summary plot saved to:"
)

print(SUMMARY_DOT_PATH)


# ============================================================
# 12. FIND AN ATTACK SAMPLE
# ============================================================

print("\n[9] Selecting an individual attack record...")

attack_indices = X_sample.index[
    y_sample == 1
]

if len(attack_indices) == 0:

    print(
        "No attack record found in SHAP sample."
    )

else:

    attack_index = attack_indices[0]

    attack_position = X_sample.index.get_loc(
        attack_index
    )

    X_attack = X_sample.loc[
        [attack_index]
    ]

    attack_shap = shap_matrix[
        attack_position
    ]


    # ========================================================
    # 13. INDIVIDUAL SHAP EXPLANATION
    # ========================================================

    print(
        "\n[10] Creating individual attack explanation..."
    )

    individual_importance = pd.DataFrame(
        {
            "feature": X_attack.columns,
            "shap_value": attack_shap,
            "feature_value": X_attack.iloc[0].values
        }
    )

    individual_importance[
        "absolute_shap"
    ] = np.abs(
        individual_importance["shap_value"]
    )

    individual_importance = (
        individual_importance
        .sort_values(
            by="absolute_shap",
            ascending=False
        )
        .reset_index(drop=True)
    )


    # ========================================================
    # 14. DISPLAY INDIVIDUAL EXPLANATION
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "        INDIVIDUAL ATTACK EXPLANATION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nOriginal label: "
        f"{y_test.loc[attack_index]}"
    )

    prediction = model.predict(
        X_attack
    )[0]

    probability = model.predict_proba(
        X_attack
    )[0][1]

    print(
        f"Model prediction: "
        f"{prediction}"
    )

    print(
        f"Attack probability: "
        f"{probability:.4f}"
    )

    print(
        "\nTop 15 contributing features:"
    )

    print(
        individual_importance.head(15).to_string(
            index=False
        )
    )


    # ========================================================
    # 15. SAVE INDIVIDUAL EXPLANATION
    # ========================================================

    with open(
        EXPLANATION_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "INDIVIDUAL SHAP EXPLANATION\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Test record index: "
            f"{attack_index}\n"
        )

        file.write(
            f"Original label: "
            f"{y_test.loc[attack_index]}\n"
        )

        file.write(
            f"Model prediction: "
            f"{prediction}\n"
        )

        file.write(
            f"Attack probability: "
            f"{probability:.4f}\n\n"
        )

        file.write(
            "Top contributing features:\n\n"
        )

        file.write(
            individual_importance.head(15).to_string(
                index=False
            )
        )

    print(
        "\nIndividual explanation saved to:"
    )

    print(
        EXPLANATION_PATH
    )


    # ========================================================
    # 16. INDIVIDUAL SHAP WATERFALL
    # ========================================================

    print(
        "\n[11] Creating individual SHAP waterfall..."
    )

    try:

        explanation = explainer(
            X_attack
        )

        shap.plots.waterfall(
            explanation[0],
            max_display=15,
            show=False
        )

        plt.tight_layout()

        plt.savefig(
            INDIVIDUAL_PATH,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "Individual SHAP plot saved to:"
        )

        print(
            INDIVIDUAL_PATH
        )

    except Exception as error:

        print(
            "\nCould not create waterfall plot."
        )

        print(
            "Reason:",
            error
        )


# ============================================================
# FINAL MESSAGE
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "          SHAP ANALYSIS COMPLETE ✓"
)

print(
    "=" * 70
)

print(
    "\nResults are available in:"
)

print(
    OUTPUT_DIR
)