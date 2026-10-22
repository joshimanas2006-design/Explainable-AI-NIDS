import pandas as pd

# ============================================
# UNSW-NB15 DATASET INSPECTION
# ============================================

TRAIN_PATH = "dataset/UNSW_NB15_training-set.csv"
TEST_PATH = "dataset/UNSW_NB15_testing-set.csv"

print("=" * 60)
print("       UNSW-NB15 DATASET INSPECTION")
print("=" * 60)

# Load datasets
print("\nLoading training dataset...")
train_df = pd.read_csv(TRAIN_PATH)

print("Loading testing dataset...")
test_df = pd.read_csv(TEST_PATH)

# ------------------------------------------------
# BASIC INFORMATION
# ------------------------------------------------

print("\n" + "=" * 60)
print("1. DATASET SIZE")
print("=" * 60)

print(f"Training rows : {train_df.shape[0]:,}")
print(f"Training cols : {train_df.shape[1]:,}")

print(f"Testing rows  : {test_df.shape[0]:,}")
print(f"Testing cols  : {test_df.shape[1]:,}")

# ------------------------------------------------
# COLUMN NAMES
# ------------------------------------------------

print("\n" + "=" * 60)
print("2. COLUMN NAMES")
print("=" * 60)

for i, column in enumerate(train_df.columns, start=1):
    print(f"{i:2}. {column}")

# ------------------------------------------------
# DATA TYPES
# ------------------------------------------------

print("\n" + "=" * 60)
print("3. DATA TYPES")
print("=" * 60)

print(train_df.dtypes)

# ------------------------------------------------
# MISSING VALUES
# ------------------------------------------------

print("\n" + "=" * 60)
print("4. MISSING VALUES")
print("=" * 60)

missing_values = train_df.isnull().sum()

print(missing_values[missing_values > 0])

if missing_values.sum() == 0:
    print("No missing values found in training dataset.")

# ------------------------------------------------
# DUPLICATES
# ------------------------------------------------

print("\n" + "=" * 60)
print("5. DUPLICATE RECORDS")
print("=" * 60)

print("Training duplicates:", train_df.duplicated().sum())
print("Testing duplicates :", test_df.duplicated().sum())

# ------------------------------------------------
# TARGET COLUMNS
# ------------------------------------------------

print("\n" + "=" * 60)
print("6. TARGET / LABEL INFORMATION")
print("=" * 60)

if "label" in train_df.columns:
    print("\nBinary Label Distribution:")
    print(train_df["label"].value_counts())

    print("\nBinary Label Percentage:")
    print(train_df["label"].value_counts(normalize=True) * 100)

if "attack_cat" in train_df.columns:
    print("\nAttack Category Distribution:")
    print(train_df["attack_cat"].value_counts())

# ------------------------------------------------
# CATEGORICAL COLUMNS
# ------------------------------------------------

print("\n" + "=" * 60)
print("7. CATEGORICAL COLUMNS")
print("=" * 60)

categorical_columns = train_df.select_dtypes(
    include=["object"]
).columns.tolist()

for column in categorical_columns:
    print(column)

# ------------------------------------------------
# NUMERICAL COLUMNS
# ------------------------------------------------

print("\n" + "=" * 60)
print("8. NUMERICAL COLUMNS")
print("=" * 60)

numerical_columns = train_df.select_dtypes(
    include=["number"]
).columns.tolist()

for column in numerical_columns:
    print(column)

# ------------------------------------------------
# FIRST 5 RECORDS
# ------------------------------------------------

print("\n" + "=" * 60)
print("9. FIRST 5 TRAINING RECORDS")
print("=" * 60)

print(train_df.head())

print("\n" + "=" * 60)
print("DATASET INSPECTION COMPLETE ✓")
print("=" * 60)