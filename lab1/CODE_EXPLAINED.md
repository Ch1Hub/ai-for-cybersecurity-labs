# CODE_EXPLAINED — Detailed walkthrough of `lab01_unsw_nb15.ipynb`

This file explains **every important block** of the notebook so you can defend it in the consultation.
Notebook has 28 cells in 9 sections (§0–§9 below). All randomness fixed with `RANDOM_STATE = 42`.

---

## §0 — Setup & imports (cells 0–1)

```python
RANDOM_STATE = 42; np.random.seed(RANDOM_STATE)
from sklearn... import train_test_split, StandardScaler, OneHotEncoder, SimpleImputer,
    ColumnTransformer, Pipeline, LogisticRegression, RandomForestClassifier, metrics...
```
- **Why these imports:** `SimpleImputer` (NaN/Inf requirement), `StandardScaler` (lab mandates scaling;
  essential for LogReg distance-based optimization, harmless for RF), `OneHotEncoder` (proto/service/state
  are nominal — no order, so no LabelEncoder), `ColumnTransformer` (different transforms per column type),
  `Pipeline` (**anti-leakage**: `fit` learns imputation means/scaler params/encoder categories on train only,
  `transform` merely applies them to val/test), `train_test_split(stratify=...)` (keeps rare Worms in val).
- `sns.set_theme`, `plt.rcParams["figure.dpi"]` — readable figures for the report.

## §1 — Loading (cell 2)

```python
train_df = pd.read_csv("UNSW_NB15_training-set.csv")  # 175341 rows
test_df  = pd.read_csv("UNSW_NB15_testing-set.csv")    # 82332 rows
assert list(train_df.columns) == list(test_df.columns)
```
- Two files kept **separate from the start**; test never concatenated before splitting (classic leakage trap).
- `assert` fails fast if sources mismatch (e.g. wrong mirror). Our files verified:
  45 cols `id,dur,proto,...,attack_cat,label`, shapes exactly matching UNSW spec.

## §2 — EDA (cells 3–4)

- `dtypes` → separates numeric (39) vs object (proto/service/state).
- `isna().sum()` + `np.isinf(...)` → lab explicitly requires NaN/Inf handling; counts are 0 here but the
  cleaning code still implements it (robustness + shows method).
- `label.value_counts()` → 68% attack train / 55% test (moderate binary imbalance).
- `attack_cat.value_counts()` → 10 classes, Generic 40000 … Worms 130 (430:1 — the lab's real challenge).
- Plots: label countplot + attack_cat **log-scale** countplot (without log, Worms/Shellcode bars are invisible),
  printed `proto/service/state` cardinalities → motivates `min_frequency` grouping (133 protos, 3 singleton states).

## §3 — Cleaning + preprocessor + validation split (cells 5–6)

```python
drop: id
CAT_COLS = ["proto","service","state"]
clean_frame(): strip+lowercase categoricals; '-'/'nan'/'' -> 'missing'; Inf -> NaN
```
- `id` dropped: row identifier, zero generalization value, memorizable by trees.
- `service='-'` means "no service" (53% of rows) — kept as `missing` category, NOT dropped, NOT NaN-imputed away.
- Lowercasing prevents `TCP` vs `tcp` splits; `replace(Inf→NaN)` lets the imputer handle extreme sensor values.

```python
preprocessor = ColumnTransformer([
  ("num", Pipeline([SimpleImputer(median), StandardScaler()]), NUM_COLS),
  ("cat", Pipeline([SimpleImputer("missing"), OneHotEncoder(handle_unknown="ignore", min_frequency=100)]), CAT_COLS)])
X_tr, X_val, ... = train_test_split(..., test_size=0.20, stratify=y_multi_full, random_state=42)
```
- Numeric median (robust to outliers like huge `sbytes`); `StandardScaler` → zero-mean/unit-variance (LogReg converges ~10× faster).
- `OneHotEncoder(handle_unknown="ignore")` → test-only proto values produce all-zero rows instead of crashing.
  `min_frequency=100` → protos/states rarer than 100 rows collapse to one `infrequent` column (cuts ~150 sparse
  cols, reduces RF overfitting on singletons like `state=no`).
- `stratify=attack_cat` (finest label) guarantees Worms in val (13 samples); stratifying only on binary `label`
  could yield 0 Worms in val. Same indices reused for Exp1/Exp2 → fair comparison.
- Imbalance print: binary counts + multi max/min ratio (≈430:1) → justifies Exp3.

## §4 — Metric helpers (cell 7)

- `scores_binary`: accuracy/precision/recall/F1 + `roc_auc_score` (needs `predict_proba[:,1]`).
- `scores_multi`: accuracy + **macro** (unweighted mean over 10 classes — punishes ignoring Worms) + **weighted**
  (support-weighted — reflects overall traffic) + `roc_auc_score(multi_class="ovr", average="macro")` via
  `label_binarize`. Both reported because the lab asks for both; their gap IS the imbalance story.
- `plot_cm` wraps `ConfusionMatrixDisplay` — diagonal = correct; off-diagonal shows confused pairs
  (e.g. Exploits↔Fuzzers in Exp1).

## §5 — Experiment 1, multi-class (cells 8–10)

```python
exp1_configs = {"LogReg C=0.5/1.0/5.0": ..., "RF d=None/100, d=20/200, d=15/100": ...}
for name, clf: pipe = Pipeline([("prep", clone(preprocessor)), ("clf", clf)]); pipe.fit(X_tr, y_multi_tr)
```
- `clone(preprocessor)` per model → each fit learns its own scaler/encoder (no cross-contamination).
- `C` = inverse regularization for LogReg (small C = stronger penalty); `max_iter=500` (multinomial lbfgs on
  140k×200 needs iterations; 100 default would not converge).
- RF grid: trees 100→200, depth None (memorize) vs 20/15 (regularize), `min_samples_split` 2 vs 5,
  `n_jobs=-1` (all cores). Deliberately **manual validation search**, not `GridSearchCV` — matches the lab's
  "simple search on the validation set" and is 3× faster.
- Selection on `val F1-weighted` (not accuracy — accuracy ignores rare classes).
- Test eval: `predict` + `predict_proba` → `scores_multi` + `classification_report` (per-class precision/recall —
  the lab's "full classification report") + confusion matrix figure + OvR ROC-AUC (~0.94 both models: ranking
  is good even when hard classification errs).
- Our result: RF 0.757 acc / 0.468 macro-F1 / 0.776 weighted-F1 beats LogReg 0.699/0.336/0.708. Gap train−val ≈ +0.2
  for unbounded RF = overfitting signal; depth-20 closes it.

## §6 — Experiment 2, binary (cell 11)

- Same `make_pipe` pattern, `y=label`. Configs re-tuned (binary needs less capacity: C=1.0 wins for LogReg).
- `target_names=["Normal","Attack"]` in report; ROC curve plotted per model (FPR vs TPR, AUC 0.956/0.982).
- Comparison table Exp1 vs Exp2 built explicitly (lab requirement "compare performance with Exp1"):
  binary +12 pts F1 because 9 attack families collapse into one well-sampled class.

## §7 — Experiment 3, Normal-vs-Worms (cells 12–14)

```python
subset_nw(): filter attack_cat in ["Normal","Worms"]; y = (attack_cat=="Worms")
```
- Filtering **after** the split (per-partition) → val/test ratios stay honest; filtering before splitting then
  re-splitting would also be valid but this reuses §3 indices (cleaner lineage).
- Sizes: train 44,904 (104 Worms), val 11,226 (26), test 37,044 (44). Ratio printed (≈430:1).

```python
strategies = {
 "RF baseline": Pipeline(prep + RF),
 "RF balanced": Pipeline(prep + RF(class_weight="balanced")),
 "RF SMOTE": ImbPipeline(prep + SMOTE(k=3) + RF),   # imblearn
 "LogReg balanced": Pipeline(prep + LogReg(class_weight="balanced"))}
```
- `class_weight="balanced"` → loss weight ∝ 1/class-frequency (Worms errors count ~430× more; no data synthesis).
- `SMOTE(k_neighbors=3)` → synthesizes Worms points along segments between each Worms sample and its 3 nearest
  Worms neighbors. k=3 (not default 5) because only 104 Worms in train — k=5 would bridge across outliers.
- `ImbPipeline` (not sklearn `Pipeline`) → `SMOTE.fit_resample` runs **inside** `fit` on train only; `predict`
  path skips it. Applying SMOTE before `train_test_split` would leak synthetic val/test information (the #1
  imbalance leakage the professor checks).
- Judged on `minority_recall`/`minority_precision`/F1 + ROC overlay (not accuracy: 99.9% is meaningless with 44 Worms).
- Our numbers: baseline recall 0.796 → SMOTE 0.841 (+2 Worms), precision 0.897→0.725; `balanced` neutral here;
  LogReg recall 0.864 but precision 0.029 (unusable) — the intelligent verdict in the report follows directly.

## §8 — Synthesis (cell 15)

- Prints the three final tables, draws the best-RF-F1 bar chart (Exp1 0.776 / Exp2 0.893 / Exp3 0.843),
  saves `results_exp1_test.csv`, `results_exp2_test.csv`, `results_exp3_all.csv` (the report's tables are generated
  from these — no hand copying).
- Conclusion markdown anticipates consultation questions (see report §5–6).

## How to re-run
1. `pip install scikit-learn pandas numpy matplotlib seaborn imbalanced-learn`
2. Place the two CSVs beside the notebook (names as in §1).
3. Run all cells top→bottom (≈6–10 min on 8 cores; RF dominates). Expect identical numbers (`random_state=42`).
