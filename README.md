# Task 01 — Clean & Prepare a Dataset

Cleaning the **Titanic passenger dataset** (891 passengers) with Python and pandas so it is ready for analysis and modelling.

- **Source:** [datasciencedojo/datasets · titanic.csv](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv) (Kaggle Titanic training set)
- **Tools:** Python, pandas, NumPy, Jupyter

## Project structure

```
├── data_cleaning.ipynb      # step-by-step notebook with outputs
├── clean_titanic.py         # same pipeline as a single script
├── requirements.txt
└── data/
    ├── titanic_raw.csv      # original dataset
    ├── titanic_clean.csv    # cleaned dataset (output)
    └── quality_report.json  # before/after quality metrics
```

## Steps

1. **Load** the raw CSV with pandas and profile it (`info()`, missing counts, duplicates).
2. **Identify missing values:** Age (177), Cabin (687) and Embarked (2).
3. **Remove duplicates:** text is normalised first, then the data is checked for exact duplicate rows and for rows that are identical except for PassengerId. None were found.
4. **Fix incorrect or inconsistent values:**
   - Titles are taken from names, and variants are merged: Mlle and Ms become Miss, Mme becomes Mrs, and rare honorifics become Rare.
   - 15 fares of £0 are flagged as invalid.
   - Range checks run on Age and the family counts, and domain checks run on Sex, Pclass and Survived.
5. **Handle missing values:**
   - Age: filled with the median for the passenger's Title and Pclass, and flagged in `AgeWasMissing`.
   - Fare: each £0 fare is replaced with the median fare of its class, and flagged in `FareWasZero`.
   - Embarked: filled with the most common port, `S`.
   - Cabin: labelled `Unknown`. The new `HasCabin` and `Deck` columns keep the useful information.
6. **Convert data types:**
   - `Survived` becomes `bool` and `Pclass` an ordered category.
   - `Sex`, `Embarked`, `Title` and `Deck` become `category`.
   - `SibSp` and `Parch` become `int8`, and the text columns become `string`.
7. **Save** the result to `data/titanic_clean.csv`.

## Before / after data quality

| Check | Before | After |
|---|---:|---:|
| Rows | 891 | 891 |
| Columns | 12 | 17 |
| Missing cells | 866 | **0** |
| Duplicate rows (exact) | 0 | 0 |
| Duplicate rows (ignoring PassengerId) | 0 | 0 |
| Invalid fares (£0) | 15 | **0** |
| Text with stray whitespace | 2 | **0** |

| Column | Missing before | Missing after |
|---|---:|---:|
| Age | 177 (19.9%) | 0 |
| Cabin | 687 (77.1%) | 0 |
| Embarked | 2 (0.2%) | 0 |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook data_cleaning.ipynb   # or: python clean_titanic.py
```

## Notes on decisions

- **Why impute Age by Title and Pclass?** Title is a strong signal for age. For example, "Master" was used for boys. A single overall median would give children an adult age.
- **Why not impute Cabin?** 77% of the values are missing, so any guess would be mostly invented. Whether a passenger had a cabin, and which deck it was on, are kept as features instead.
- **Why treat £0 fares as invalid?** No passenger ticket cost nothing. The `FareWasZero` flag means this choice can be undone in later analysis.
- **Flags instead of silent changes.** The `AgeWasMissing` and `FareWasZero` columns record which values were filled in, so later analysis can tell real values from imputed ones.
