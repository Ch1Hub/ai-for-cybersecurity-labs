# AI for Cybersecurity Labs

Applied machine learning for detection engineering, from the final year of the cybersecurity engineering cycle at **ESI SBA**. Three complete labs cover supervised intrusion detection, a head-to-head comparison of ML and hand-written rules, and unsupervised anomaly detection.

**3 labs · scikit-learn · pandas · Jupyter · UNSW-NB15 · CICIDS2017**

## What is inside

| Lab | Topic | Deliverables |
| --- | --- | --- |
| [Lab 1](lab1/) | Intrusion detection on UNSW-NB15 | Preprocessing, model comparison, `CODE_EXPLAINED.md`, full report |
| [Lab 2](lab2/) | ML versus rule-based detection | Random forest and rule engine, confusion matrices, feature importance |
| [Lab 3](lab3/) | Unsupervised anomaly detection | CICIDS2017 curated features, synthetic traffic generator, full report |

Every lab includes a notebook, an executed report (PDF/HTML/Markdown) and the figures needed to read the results. The raw CICIDS2017 CSVs (1.2 GB) are not included; the curated feature set and the notebook that builds it are.

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install jupyter pandas scikit-learn matplotlib seaborn
jupyter notebook
```

The notebooks are self-contained and use relative data paths. Large training sets can be obtained from their public sources (UNSW-NB15, CICIDS2017).

## About the archive

Split from the larger [`CyberSec at ESI`](https://github.com/Ch1Hub/CyberSec-at-ESI) archive with dated history (October–November 2026).

---

*Published for study and portfolio purposes.*
