"""
Task 01 — Clean & Prepare a Dataset
Dataset: Titanic passenger manifest (891 passengers, Kaggle "train.csv")
Source:  https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

Usage:   python clean_titanic.py
"""
import json
import sys

import numpy as np
import pandas as pd

RAW_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/titanic_raw.csv"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "data/titanic_clean.csv"


def quality_report(df: pd.DataFrame) -> dict:
    """Summarise data quality so before/after can be compared."""
    text_cols = df.select_dtypes(include=["object", "string", "str"]).columns
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_col": {c: int(n) for c, n in df.isna().sum().items() if n > 0},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ignoring_id": int(df.drop(columns=["PassengerId"]).duplicated().sum()),
        "zero_fares": int((df["Fare"] == 0).sum()),
        "untrimmed_text": int(sum((df[c].dropna() != df[c].dropna().str.strip()).sum() for c in text_cols)),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }


# ── 1. Load ───────────────────────────────────────────────────────────────
df = pd.read_csv(RAW_PATH)
before = quality_report(df)
print(f"Loaded {len(df)} rows × {df.shape[1]} columns from {RAW_PATH}")
log = []

# ── 2. Identify missing values ────────────────────────────────────────────
print("\nMissing values (before):")
print(df.isna().sum()[df.isna().sum() > 0].to_string())

# ── 3. Normalise text, then remove duplicates ─────────────────────────────
# Trim whitespace / collapse internal spaces / lowercase categoricals first,
# so near-duplicates that differ only in formatting are caught too.
for c in ["Name", "Ticket", "Cabin"]:
    df[c] = df[c].str.strip().str.replace(r"\s+", " ", regex=True)
for c in ["Sex", "Embarked"]:
    df[c] = df[c].str.strip()
df["Sex"] = df["Sex"].str.lower()
df["Embarked"] = df["Embarked"].str.upper()

n0 = len(df)
df = df.drop_duplicates()                                                   # exact copies
df = df.drop_duplicates(subset=[c for c in df.columns if c != "PassengerId"])  # same person, new ID
log.append(f"Duplicates removed: {n0 - len(df)} (checked exact rows and rows identical except PassengerId)")

# ── 4. Handle incorrect / inconsistent values ─────────────────────────────
# 4a. Titles: extract from Name and merge spelling variants / rare titles.
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
title_map = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs", "Lady": "Rare", "the Countess": "Rare",
             "Capt": "Rare", "Col": "Rare", "Don": "Rare", "Dr": "Rare", "Major": "Rare",
             "Rev": "Rare", "Sir": "Rare", "Jonkheer": "Rare", "Dona": "Rare"}
df["Title"] = df["Title"].replace(title_map)
log.append("Title variants unified: Mlle/Ms → Miss, Mme → Mrs, 10 rare honorifics → Rare")

# 4b. Fare = 0 is not a real ticket price -> treat as missing, impute below.
df["FareWasZero"] = df["Fare"] == 0
log.append(f"Fares of 0 flagged as invalid: {int(df['FareWasZero'].sum())}")
df.loc[df["FareWasZero"], "Fare"] = np.nan

# 4c. Range checks (would catch typos such as negative ages or Age=800).
bad_age = ~df["Age"].between(0, 100) & df["Age"].notna()
bad_family = (df["SibSp"] < 0) | (df["Parch"] < 0)
df.loc[bad_age, "Age"] = np.nan
log.append(f"Out-of-range ages set to missing: {int(bad_age.sum())}; negative family counts: {int(bad_family.sum())}")

# 4d. Valid category domains.
assert set(df["Sex"].unique()) <= {"male", "female"}
assert set(df["Pclass"].unique()) <= {1, 2, 3}
assert set(df["Survived"].unique()) <= {0, 1}

# ── 5. Handle missing values ──────────────────────────────────────────────
# Age: median by Title × Pclass (a "Master" is a boy, so a global median would be wrong).
df["AgeWasMissing"] = df["Age"].isna()
df["Age"] = df["Age"].fillna(df.groupby(["Title", "Pclass"])["Age"].transform("median"))
df["Age"] = df["Age"].fillna(df["Age"].median())
log.append(f"Age imputed for {int(df['AgeWasMissing'].sum())} passengers (median by Title × Pclass), flagged in AgeWasMissing")

# Fare: median by Pclass for the invalid zeros.
df["Fare"] = df["Fare"].fillna(df.groupby("Pclass")["Fare"].transform("median")).round(4)
log.append("Zero fares replaced with median fare of their passenger class")

# Embarked: both missing passengers (Icard & Stone) shared ticket 113572;
# Southampton is also the most common port.
n_emb = int(df["Embarked"].isna().sum())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
log.append(f"Embarked filled for {n_emb} passengers with the most common port (S)")

# Cabin: 77% missing — too sparse to impute. Keep the info as features instead.
df["HasCabin"] = df["Cabin"].notna()
df["Deck"] = df["Cabin"].str[0].fillna("Unknown")
df["Cabin"] = df["Cabin"].fillna("Unknown")
log.append("Cabin missing values labelled 'Unknown'; Deck and HasCabin columns derived")

# ── 6. Convert data types ─────────────────────────────────────────────────
df = df.astype({
    "PassengerId": "int32",
    "Survived": "bool",
    "Pclass": pd.CategoricalDtype([1, 2, 3], ordered=True),
    "Sex": "category",
    "Embarked": "category",
    "Title": "category",
    "Deck": "category",
    "SibSp": "int8",
    "Parch": "int8",
    "Age": "float32",
    "Fare": "float64",
    "Name": "string",
    "Ticket": "string",
    "Cabin": "string",
})
log.append("Types converted: Survived → bool, Pclass → ordered category, Sex/Embarked/Title/Deck → category, "
           "SibSp/Parch → int8, text → string")

# ── 7. Save ───────────────────────────────────────────────────────────────
df.to_csv(OUT_PATH, index=False)
after = quality_report(df)

print("\nCleaning steps:")
for line in log:
    print(" -", line)
print("\nBEFORE vs AFTER")
for k in ["rows", "columns", "missing_total", "duplicate_rows", "duplicate_ignoring_id", "zero_fares", "untrimmed_text"]:
    print(f"  {k:<24}{before[k]:>8}  →  {after[k]}")
print(f"\nSaved cleaned dataset to {OUT_PATH}")

with open("data/quality_report.json", "w") as f:
    json.dump({"before": before, "after": after, "steps": log}, f, indent=2)
