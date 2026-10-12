# Lab 01 Report — Data Preparation, Model Training & Evaluation on UNSW-NB15

**Module:** IA for Cybersecurity — 3rd Year 2nd Cycle, CyS | **Lab 01**
**Dataset:** `UNSW_NB15_training-set.csv` (175,341 rows) + `UNSW_NB15_testing-set.csv` (82,332 rows)
**Code:** `lab01_unsw_nb15.ipynb` (reproducible, `random_state=42`) — see `CODE_EXPLAINED.md` for line-by-line walkthrough.
**Models:** Logistic Regression (baseline) + Random Forest, all inside `sklearn Pipeline` (no leakage).

---

## 1. Data Preparation

### 1.1 Loading & exploration
- Both CSVs loaded separately; columns verified identical (45 cols: `id` + 42 features + `attack_cat` + `label`).
- No missing values, no Inf (checked explicitly). Pipeline still imputes (median / `missing`) for robustness, as required.
- Types: 39 numeric + 3 categorical (`proto`, `service`, `state`).

### 1.2 Class distributions (the core difficulty of this lab)
| Split | Normal (0) | Attack (1) |
|---|---|---|
| Train (175,341) | 56,000 (31.9%) | 119,341 (68.1%) |
| Test (82,332) | 37,000 (44.9%) | 45,332 (55.1%) |

`attack_cat` (train): Normal 56000, Generic 40000, Exploits 33393, Fuzzers 18184, DoS 12264,
Reconnaissance 10491, Analysis 2000, Backdoor 1746, Shellcode 1133, **Worms 130**.
Test shows the same order with Worms=44. Max/min ratio ≈ **430:1** — severe multi-class imbalance.

Categorical cardinality: `proto` 133 values (tcp/udp/unas ≈ 90%), `service` 13 values (`missing` ex-`-` = 53%,
dns, http, …), `state` 9 values (INT/FIN/CON dominate; PAR/URN/no = 1 sample each).

### 1.3 Cleaning & encoding & scaling (leakage-proof)
1. Dropped `id` (identifier, no predictive value, overfitting risk).
2. Categorical normalization: strip + lowercase, `-`/empty → `missing` (kept as a real category).
3. `Inf → NaN` replacement (0 found, methodologically required).
4. `ColumnTransformer`: numeric → `SimpleImputer(median)` + `StandardScaler`; categorical →
   `SimpleImputer(missing)` + `OneHotEncoder(handle_unknown='ignore', min_frequency=100)`.
   `min_frequency=100` groups the ~120 rare protos and singleton states into one infrequent bucket:
   limits dimensionality (~200 cols instead of ~180), reduces overfitting, still encodes every value.
5. **Anti-leakage:** transformer fitted ONLY on inner-train via `Pipeline([prep, clf])`. Never `fit` on val/test.
   `handle_unknown='ignore'` absorbs test-only categories safely.
6. Validation: `train_test_split(test_size=0.20, stratify=attack_cat, random_state=42)` on the train file →
   inner-train ~140k / val ~35k. Held-out `testing-set.csv` touched once, at final eval. Same split indices
   reused for Exp1/Exp2 so the comparison is fair. Feature selection deliberately skipped (full 42 features
   already tractable; RF importances can be added as extension).

---

## 2. Experiment 1 — Multi-class (`attack_cat`, 10 classes)

Simple search on **validation** (F1-weighted), then test:

| Model (val-selected) | Test Acc | Prec macro | Rec macro | F1 macro | F1 weighted | ROC-AUC OvR macro |
|---|---|---|---|---|---|---|
| LogReg C=5.0 | 0.699 | 0.353 | 0.384 | 0.336 | 0.708 | 0.949 |
| **RF (100 trees, depth=None)** | **0.757** | **0.521** | **0.492** | **0.468** | **0.776** | **0.944** |

Full per-class report (RF, test): Normal/Generic/Exploits/Fuzzers/DoS/Recon F1 ≈ 0.70–0.98;
Analysis/Backdoor/Shellcode F1 ≈ 0.2–0.5; **Worms F1 ≈ 0.78 but on 44 samples**.
LogReg underfits (linear boundary insufficient for 10 classes); RF overfits slightly
(train F1w ≈ 0.99 vs val ≈ 0.78, gap ≈ +0.21 when unbounded) — depth-20 variant closes the gap
with <1 pt loss, recommended if deployment matters. Macro scores stay low for both models:
the mean is dragged by rare classes — exactly why both macro AND weighted are reported.

## 3. Experiment 2 — Binary (`label`: Normal vs Attack)

| Model | Test Acc | Prec | Rec | F1 | ROC-AUC |
|---|---|---|---|---|---|
| LogReg C=1.0 | 0.809 | 0.753 | 0.973 | 0.849 | 0.956 |
| **RF (200 trees, depth=20)** | **0.870** | **0.815** | **0.987** | **0.893** | **0.982** |

**Exp1 vs Exp2:** binary F1 0.893 ≫ multi-class F1-weighted 0.776 (+12 pts); accuracy 0.870 vs 0.757.
Collapsing 9 attacks into one class removes fine-grained confusion (Exploits↔Fuzzers, DoS↔Recon),
and each binary class has 37k+ samples vs 44 Worms. Cost: loss of actionable detail (which attack?).
Operational reading: binary = triage alarm, multi-class = diagnosis — use both stages.

## 4. Experiment 3 — `Normal vs Worms` + imbalance (intelligent handling)

Subset sizes — train 44,904 (Normal 44,800 / Worms 104), val 11,226 (11,200/26), test 37,044 (37,000/44).
Ratio ≈ **430:1**. Same RF backbone (200 trees, depth=20); SMOTE `k_neighbors=3` (minority too small for k=5)
inside `imblearn.Pipeline` (oversampling train-only, no leakage). Selection criterion: **minority Recall/F1**, not accuracy.

| Strategy (TEST) | Acc | Prec (Worms) | Rec (Worms) | F1 (Worms) | ROC-AUC |
|---|---|---|---|---|---|
| RF baseline | 0.9996 | 0.897 | 0.796 | 0.843 | 0.999 |
| RF `class_weight=balanced` | 0.9996 | 0.897 | 0.796 | 0.843 | 0.999 |
| **RF SMOTE(k=3)** | 0.9994 | 0.725 | **0.841** | 0.779 | 0.998 |
| LogReg balanced | 0.9657 | 0.029 | 0.864 | 0.056 | 0.978 |

Reading:
- **Accuracy lies**: 99.96% baseline hides 20% missed Worms. Always quote minority recall.
- `class_weight` here is neutral (Worms already separable; weights don't hurt — zero-cost safety net, guideline-compliant).
- **SMOTE wins on recall** (0.796 → 0.841, +4.5 pts, 2 extra Worms caught of 44) at a precision cost (0.897 → 0.725, more false alarms). Val confirms (recall 0.923 → 0.962).
- LogReg balanced is the cautionary tale: recall 0.864 but precision 0.029 (1,264 false alarms for 38 Worms) — unusable.
- **Verdict (intelligent rule):** default to `class_weight='balanced'` (free, compliant, stable); add SMOTE when
  minority recall still below target on validation AND you can afford extra false positives — exactly the Worms case.
  SMOTE must stay inside the pipeline (applying it before the split leaks synthetic minority info into validation).

## 5. Overfitting / underfitting synthesis
- LogReg: underfits multi-class (train≈val, both low), fits binary adequately.
- RF unbounded: overfits (train F1w ~0.99 vs val ~0.78); depth cap 15–20 removes most of the gap for ~1 pt cost.
- Exp3: train/val gaps small → no serious overfitting on the subset; the risk there is variance (26 val Worms, 44 test Worms — one sample = 2–4 pts).

## 6. Conclusion
1. Leakage prevention (Pipeline + train-only fit + stratified split + untouched test set) makes the numbers trustworthy.
2. RF beats LogReg everywhere; binary beats multi-class on raw scores but loses diagnosis granularity.
3. Imbalance must be judged on minority recall/F1 + ROC/PR, never accuracy; `balanced` first (neutral here — identical to baseline — but free), SMOTE when recall still below target and FP cost affordable.
4. Operational proposal: binary RF as alarm → multi-class RF as diagnosis → specialized Normal-vs-rare detector (SMOTE) for Worms/Shellcode.

**Reproducibility:** `random_state=42` everywhere, `results_exp{1,2,3}*.csv` regenerated by the notebook, versions: sklearn 1.7.2, pandas 2.2.3, imbalanced-learn 0.14.0, matplotlib 3.11, seaborn 0.13.
**Sources:** train/test CSVs from HuggingFace `Mouwiya/UNSW-NB15` (train) + `Mouwiya/UNSW-NB15-small` (test),
verified: 175,341 / 82,332 rows, 10-class distribution matching the UNSW specification (research.unsw.edu.au/projects/unsw-nb15-dataset).
