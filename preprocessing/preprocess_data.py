import os
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = "dataset/UNSW_NB15_training-set.csv"
TEST_PATH = "dataset/UNSW_NB15_testing-set.csv"

OUTPUT_DIR = "preprocessing/processed"
PREPROCESSOR_PATH = "models/saved/preprocessor.pkl"

TRAIN_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "UNSW_NB15_train_processed.csv"
)

TEST_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "UNSW_NB15_test_processed.csv"
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 70)
print("        UNSW-NB15 DATA PREPROCESSING")
print("=" * 70)

print("\n[1] Loading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Training dataset: {train_df.shape}")
print(f"Testing dataset : {test_df.shape}")


# ============================================================
# 2. REMOVE DUPLICATES
# ============================================================

print("\n[2] Checking duplicate records...")

train_duplicates = train_df.duplicated().sum()
test_duplicates = test_df.duplicated().sum()

print(f"Training duplicates: {train_duplicates}")
print(f"Testing duplicates : {test_duplicates}")

train_df = train_df.drop_duplicates().reset_index(drop=True)
test_df = test_df.drop_duplicates().reset_index(drop=True)


# ============================================================
# 3. DEFINE TARGET
# ============================================================

print("\n[3] Defining target variable...")

TARGET_COLUMN = "label"

y_train = train_df[TARGET_COLUMN].copy()
y_test = test_df[TARGET_COLUMN].copy()


# ============================================================
# 4. REMOVE COLUMNS NOT USED AS FEATURES
# ============================================================

print("\n[4] Removing non-feature columns...")

COLUMNS_TO_REMOVE = [
    "id",
    "label",
    "attack_cat"
]

X_train = train_df.drop(columns=COLUMNS_TO_REMOVE)
X_test = test_df.drop(columns=COLUMNS_TO_REMOVE)

print("Removed columns:", COLUMNS_TO_REMOVE)
print("Remaining features:", X_train.shape[1])


# ============================================================
# 5. IDENTIFY FEATURE TYPES
# ============================================================

print("\n[5] Identifying numerical and categorical features...")

categorical_features = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nCategorical features:")
for feature in categorical_features:
    print(" -", feature)

print("\nNumber of categorical features:",
      len(categorical_features))

print("\nNumber of numerical features:",
      len(numerical_features))


# ============================================================
# 6. CREATE PREPROCESSING PIPELINES
# ============================================================

print("\n[6] Creating preprocessing pipelines...")


# Numerical preprocessing:
# Missing values -> median
# Scaling -> MinMaxScaler

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            MinMaxScaler()
        )
    ]
)


# Categorical preprocessing:
# Missing values -> most frequent
# Encoding -> One-Hot Encoding

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# Combine both pipelines

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 7. FIT ONLY ON TRAINING DATA
# ============================================================

print("\n[7] Fitting preprocessing pipeline on training data...")

X_train_processed = preprocessor.fit_transform(X_train)

print("Training preprocessing completed.")
os.makedirs(
    os.path.dirname(PREPROCESSOR_PATH),
    exist_ok=True
)

joblib.dump(
    preprocessor,
    PREPROCESSOR_PATH
)

print("\nPreprocessor saved to:")
print(PREPROCESSOR_PATH)


# ============================================================
# 8. TRANSFORM TEST DATA
# ============================================================

print("\n[8] Transforming testing data...")

X_test_processed = preprocessor.transform(X_test)

print("Testing preprocessing completed.")


# ============================================================
# 9. GET FEATURE NAMES
# ============================================================

print("\n[9] Generating processed feature names...")

feature_names = preprocessor.get_feature_names_out()

print("Number of processed features:",
      len(feature_names))


# ============================================================
# 10. CONVERT TO DATAFRAMES
# ============================================================

print("\n[10] Creating processed DataFrames...")

X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names
)


# ============================================================
# 11. ADD TARGET COLUMN
# ============================================================

X_train_processed["label"] = y_train.values
X_test_processed["label"] = y_test.values


# ============================================================
# 12. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 13. SAVE PROCESSED DATA
# ============================================================

print("\n[11] Saving processed datasets...")

X_train_processed.to_csv(
    TRAIN_OUTPUT,
    index=False
)

X_test_processed.to_csv(
    TEST_OUTPUT,
    index=False
)

print("\nTraining file saved to:")
print(TRAIN_OUTPUT)

print("\nTesting file saved to:")
print(TEST_OUTPUT)


# ============================================================
# 14. FINAL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("             PREPROCESSING COMPLETE")
print("=" * 70)

print("\nOriginal training shape:", train_df.shape)
print("Original testing shape :", test_df.shape)

print("\nProcessed training shape:",
      X_train_processed.shape)

print("Processed testing shape :",
      X_test_processed.shape)

print("\nTraining labels:")
print(y_train.value_counts())

print("\nTesting labels:")
print(y_test.value_counts())

print("\n✓ No test-data fitting was performed.")
print("✓ Training data was used to fit preprocessing.")
print("✓ Testing data was only transformed.")
print("✓ id and attack_cat were excluded.")
print("✓ label is the prediction target.")

print("\nPreprocessing pipeline completed successfully!")