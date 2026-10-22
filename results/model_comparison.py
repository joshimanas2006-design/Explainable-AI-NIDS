import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# MODEL RESULTS
# ============================================================

results = {
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost"
    ],

    "Accuracy": [
        0.8061,
        0.8618,
        0.8685,
        0.8732
    ],

    "Precision": [
        0.7494,
        0.8246,
        0.8142,
        0.8192
    ],

    "Recall": [
        0.9733,
        0.9512,
        0.9864,
        0.9876
    ],

    "F1-Score": [
        0.8468,
        0.8834,
        0.8920,
        0.8956
    ],

    "ROC-AUC": [
        0.9553,
        0.8542,
        0.9813,
        0.9846
    ]
}


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(results)

print("=" * 75)
print("              MODEL PERFORMANCE COMPARISON")
print("=" * 75)

print("\n")
print(df.to_string(index=False))


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# SAVE CSV
# ============================================================

csv_path = "results/model_comparison.csv"

df.to_csv(
    csv_path,
    index=False
)

print("\nComparison table saved to:")
print(csv_path)


# ============================================================
# SAVE TEXT REPORT
# ============================================================

report_path = "results/model_comparison_report.txt"

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "MODEL PERFORMANCE COMPARISON\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        df.to_string(index=False)
    )

    file.write("\n\n")

    file.write(
        "Metric definitions:\n"
    )

    file.write(
        "Accuracy  - Overall proportion of correct predictions.\n"
    )

    file.write(
        "Precision - Proportion of predicted attacks that were actual attacks.\n"
    )

    file.write(
        "Recall    - Proportion of actual attacks correctly detected.\n"
    )

    file.write(
        "F1-Score  - Harmonic mean of precision and recall.\n"
    )

    file.write(
        "ROC-AUC   - Ability of the model to distinguish the two classes.\n"
    )


print("\nReport saved to:")
print(report_path)


# ============================================================
# BAR CHART
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC"
]

for metric in metrics:

    plt.figure(figsize=(9, 6))

    plt.bar(
        df["Model"],
        df[metric]
    )

    plt.title(
        f"{metric} Comparison of NIDS Models"
    )

    plt.ylabel(metric)

    plt.ylim(
        0,
        1
    )

    plt.xticks(
        rotation=20
    )

    plt.tight_layout()

    filename = (
        "results/"
        + metric.lower().replace("-", "_")
        + "_comparison.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"{metric} chart saved to:"
    )

    print(filename)


# ============================================================
# COMBINED METRICS CHART
# ============================================================

plt.figure(figsize=(11, 7))

x = range(len(df["Model"]))

width = 0.15

for i, metric in enumerate(metrics):

    values = df[metric]

    positions = [
        value + (i - 2) * width
        for value in x
    ]

    plt.bar(
        positions,
        values,
        width=width,
        label=metric
    )

plt.xticks(
    list(x),
    df["Model"],
    rotation=20
)

plt.ylabel(
    "Score"
)

plt.title(
    "Performance Comparison of Machine Learning Models"
)

plt.ylim(
    0,
    1
)

plt.legend()

plt.tight_layout()

combined_path = (
    "results/model_performance_comparison.png"
)

plt.savefig(
    combined_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nCombined comparison chart saved to:")
print(combined_path)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("             MODEL COMPARISON COMPLETE ✓")
print("=" * 75)