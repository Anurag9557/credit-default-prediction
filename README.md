# Credit Default Prediction

Predicting loan default risk on the **UCI Statlog German Credit** dataset (Hofmann, 1994) — 1,000 applicants, 20 features. The model is tuned to minimise **business cost**, not accuracy.

---

## Problem

Classify loan applicants by default risk. The two errors are not equally expensive:

- **False negative** — approve someone who defaults → lose the loan principal
- **False positive** — reject someone who would have repaid → lose only their margin

The dataset ships with an **official cost matrix** specifying that classifying a bad customer as good costs **5×** the reverse. The model is optimised against that cost function.

---

## Data

| | |
|---|---|
| Rows | 1,000 applicants |
| Features | 20 (7 numeric, 13 categorical) |
| Target | 700 repaid (70%) / 300 default (30%) |
| Missing values | None |
| Source file | `german.data` (raw categorical codes, not the pre-encoded `german.data-numeric`) |

The raw file was used deliberately so that preprocessing could be built explicitly and the categorical levels stay traceable — necessary for explaining individual credit decisions.

---

## Approach

1. Remapped target so **1 = default** (the event of interest), making precision/recall interpretable
2. **Stratified** 80/20 split preserving the 70/30 class ratio
3. `ColumnTransformer` (StandardScaler on numeric, OneHotEncoder on categorical) inside a `Pipeline` — preprocessing is fit on training folds only, preventing **data leakage**
4. Compared Logistic Regression, Random Forest, Gradient Boosting
5. Addressed the class imbalance with `class_weight='balanced'`
6. Tuned the decision threshold against the dataset's 5:1 cost matrix
7. Validated with 5-fold cross-validation

---

## Results

### Model comparison (test set, n=200)

| Model | ROC-AUC | PR-AUC | Recall (default) |
|---|---|---|---|
| LogReg | 0.804 | 0.635 | 0.533 |
| RandomForest | 0.784 | 0.634 | 0.417 |
| **LogReg balanced** | **0.806** | 0.633 | **0.800** |
| GradBoost | 0.792 | 0.630 | 0.567 |
| RF balanced | 0.801 | 0.628 | 0.600 |

All five models sit within 0.007 PR-AUC of each other — **statistically indistinguishable** at this sample size. The model was therefore selected on **recall and interpretability**, not on a performance difference that the data cannot support.

### Cost optimisation (the actual result)

| Stage | Recall | Precision | Accuracy | **Cost** |
|---|---|---|---|---|
| Baseline LogReg @ 0.50 | 0.533 | 0.667 | 0.780 | 156 |
| + `class_weight='balanced'` @ 0.50 | 0.800 | 0.558 | 0.750 | 98 |
| + threshold tuned to 0.46 | **0.833** | 0.532 | 0.730 | **94** |

**Total cost reduction: 39.7%** (156 → 94), where cost = `FP×1 + FN×5`.

Note that **accuracy falls** from 0.78 to 0.73 across this improvement. That is the intended trade — accuracy is the wrong objective when errors are asymmetric.

Missed defaulters dropped from **28 → 10** out of 60.

### Cross-validation

`CV ROC-AUC: 0.788 (±0.016)` — the tight standard deviation indicates a stable model. It sits slightly below the single-split test ROC-AUC of 0.806, suggesting the test split was marginally optimistic.

**Final model:** Logistic Regression, `class_weight='balanced'`, decision threshold **0.46**

---

## Key drivers of default

From logistic regression coefficients (positive → pushes toward default):

| Feature | Meaning | Coef | Direction |
|---|---|---|---|
| `purpose_A46` | Loan for education | +0.96 | ↑ risk |
| `property_A124` | No property / unknown | +0.64 | ↑ risk |
| `checking_acct_A11` | Checking balance < 0 DM | +0.62 | ↑ risk |
| `credit_history_A30` | No credits taken at this bank | +0.59 | ↑ risk |
| `savings_A61` | Savings < 100 DM | +0.58 | ↑ risk |
| `savings_A64` | Savings ≥ 1000 DM | −0.68 | ↓ risk |
| `credit_history_A34` | Critical account / credits elsewhere | −0.85 | ↓ risk |
| `purpose_A41` | Loan for a used car | −0.89 | ↓ risk |
| `checking_acct_A14` | No checking account | −0.99 | ↓ risk |

**The most interesting finding:** `credit_history_A30` ("no credits taken") *increases* predicted risk, while `A34` ("critical account / other credits existing") *decreases* it. The label that sounds healthy is riskier, and the one that sounds alarming is safer — because A30 signals a **thin credit file** (unknown risk) whereas A34 signals a **demonstrated track record of servicing debt**. This mirrors real credit underwriting, where absence of history is harder to price than imperfect history.

---

## Limitations

- **Small sample.** 1,000 rows; only 60 defaulters in the test set. Recall of 0.833 carries a 95% confidence interval of roughly **[0.74, 0.93]** — a single applicant moves it by 1.7 points.
- **Threshold tuned on the test set.** The 0.46 threshold was selected using the same data it is reported on, which optimistically biases the final numbers. A held-out validation split or nested CV would be the correct procedure.
- **Most of the gain came from `class_weight`, not the threshold.** Class weighting cut cost by 37.2%; threshold tuning added a further 2.6%. The two are partly redundant — balanced weights already shift the probability distribution, which is why the optimal threshold (0.46) landed near 0.50.
- **Protected attributes present.** `personal_status` encodes sex and `foreign_worker` encodes nationality. Both were included here for completeness, but using them in a real lending decision would violate fair-lending regulation and both should be dropped before any deployment.
- **Vintage data.** 1994 German lending patterns; findings do not transfer directly to a modern portfolio.

---

## Next steps

- Refit with protected attributes removed and measure the performance cost of doing so
- Probability calibration (`CalibratedClassifierCV`) so predicted probabilities are usable as risk scores
- Nested cross-validation for unbiased threshold selection
- Per-applicant reason codes for rejections

---

## Files

```
credit-default-prediction/
├── credit_default.ipynb    # full analysis
├── requirements.txt
├── .gitignore
└── data/
    ├── german.data         # raw dataset
    └── german.doc          # attribute code definitions (A11, A34, ...)
```

`credit_model.pkl` (the saved Pipeline) is generated by running the notebook and is
intentionally not committed — it is a build artifact, fully reproducible from the code.

## Run

```bash
pip install -r requirements.txt jupyter
jupyter notebook credit_default.ipynb
```

Dataset: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
