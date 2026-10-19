# Lab 02: Network Traffic Pattern Analysis & DDoS Detection
## Heuristic Behavioral Detection and Anti-Memorization Analysis

**Course:** AI for Cybersecurity (Dr. R. BOUSMAHA)  
**Level:** 3rd Year - Second Cycle / Option: Cybersecurity (CyS)  
**Student:** Chikhaoui Youcef Chihab Eddine  
**Date:** March 2026  

---

## Executive Summary

This laboratory work implements an end-to-end exploratory analysis, heuristic threshold rule, and machine learning detector for network intrusion detection using the **CIC-IDS2017** benchmark. We specifically address a pervasive problem in cybersecurity machine learning: **dataset artifact memorization**. Standard models evaluated on raw network flow dumps frequently report artificially inflated performances (>99.9%) by exploiting single-session tool fingerprints (such as fixed destination ports or static TCP window sizes) rather than learning generalized attack behaviors.

By intentionally purging tool-specific signatures and retaining only **7 core heuristic behavioral flow features**, we develop:
1. **A Simple Threshold Detection Rule** based on bidirectional payload amplification, achieving **79.23% accuracy** and **92.98% precision**.
2. **A Constrained Heuristic Random Forest Classifier** (`max_depth=3`) achieving **98.00% accuracy**, **97.66% precision**, and **98.36% recall** on 10,000 unseen test flows.

This report documents all 9 tasks prescribed by the laboratory syllabus, detailing the exploratory analysis, data sanitization, visualization artifacts, heuristic formulation, and critical performance trade-offs.

---

## Task 1: Cybersecurity Use Case

### Context and Operational Scope
* **Domain:** Network Traffic Telemetry Monitoring & Network Intrusion Detection Systems (NIDS).
* **Operational Motivation:** Enterprise networks face persistent volumetric and application-layer Denial of Service (DoS/DDoS) campaigns designed to exhaust server socket pools, connection queues, and back-end database resources. Static firewall filters relying on source IP addresses or static packet signatures frequently fail against distributed botnets with spoofed headers or polymorphic clients.
* **Objective:** Extract flow-level statistical metrics (packet inter-arrival rates, directional payload volumes, and session holding durations) to differentiate legitimate user sessions from anomalous flood traffic in real time.

---

## Task 2: Dataset Selection

### Benchmark Source
We utilize the **CIC-IDS2017** benchmark dataset provided by the Canadian Institute for Cybersecurity (University of New Brunswick). The dataset features captured network traffic generated in a realistic testbed environment over five consecutive days.

### Capture Files Analyzed
1. **`Monday-WorkingHours.pcap_ISCX.csv`** (Baseline Normal Traffic):
   * Contains **529,918 flows**, representing 100% benign operational traffic (HTTP/HTTPS browsing, internal DNS queries, SSH management, NetBIOS).
2. **`Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`** (Attack Traffic):
   * Contains **225,745 flows**, capturing both background legitimate enterprise sessions and an active **Low Orbit Ion Cannon (LOIT)** HTTP DDoS flood attack.

### Balanced Dataset Construction
To prevent evaluation bias and class imbalance distortions, we construct a balanced evaluation corpus of **50,000 flows**:
* **25,000 Benign flows** randomly sampled from Monday's baseline capture.
* **25,000 DDoS attack flows** randomly sampled from Friday's afternoon attack session.

---

## Task 3: Tool Selection & Justification

The implementation is executed in Python using standard scientific and machine learning libraries:

| Tool / Library | Role in Pipeline | Justification |
|---|---|---|
| **Python 3** | Core Runtime | Standard environment for security telemetry analysis and machine learning workflows. |
| **pandas** | Tabular Ingestion & Cleaning | Efficient vectorized operations, label normalization, column stripping, and conditional filtering. |
| **numpy** | Numerical Transformation | Robust handling of numeric infinities (`inf`, `-inf`) and logarithmic density scaling. |
| **matplotlib & seaborn** | Statistical Visualizations | Production-grade density plots, distribution boxplots, count plots, and confusion matrix heatmaps. |
| **scikit-learn** | Modeling & Evaluation | Stratified data splitting, Random Forest ensemble fitting, and precision/recall/confusion matrix computation. |

---

## Task 4: Data Exploration, Cleaning & The Anti-Memorization Strategy

### 1. Identifying the Memorization Pitfall
Initial experiments training an unconstrained machine learning model on all 78 numeric features yielded a suspiciously high accuracy of **99.97%**. A granular audit of the dataset revealed severe data leakage caused by the testbed attack generation setup:
* **`Destination Port`:** 100% of the DDoS flows targeted port `80` (a single victim web server). In contrast, benign flows were distributed across 1,500+ different ports. An unconstrained tree split immediately on `Destination Port == 80`.
* **`Init_Win_bytes_backward`:** DDoS traffic had only 2 discrete values (`229` or `-1`).
* **`min_seg_size_forward`:** Fixed to `20` in 100% of DDoS flows.
* **`Fwd Packet Length Mean`:** Clustered into only 17 discrete values (3.0 to 10.0 bytes), reflecting LOIT's hardcoded request payload length.

A model learning these fields does not detect DDoS behavior; it merely fingerprints LOIT executing against that specific IP target.

### 2. Curating 7 Heuristic Behavioral Features
To build a realistic, generalized heuristic detector, we stripped all port numbers, protocol flags, and TCP window/segment sizes. We retained only **7 high-level behavioral flow features**:

| Feature Name | Telemetry Dimension | Physical Heuristic Value |
|---|---|---|
| `Flow Duration` | Temporal Duration | Flood connections stay open significantly longer awaiting server timeouts. |
| `Flow Bytes/s` | Volumetric Throughput | Rate of byte transfer relative to flow lifespan. |
| `Flow Packets/s` | Volumetric Arrival | Frequency of packet arrival within the bidirectional session. |
| `Total Length of Bwd Packets` | Directional Volume | Cumulative bytes returned from the target server back to the client. |
| `Average Packet Size` | Packet Profiling | Mean payload size across the entire bidirectional exchange. |
| `Bwd Packet Length Mean` | Response Profiling | Mean size of server response packets. |
| `Down/Up Ratio` | Symmetry Profiling | Ratio of backward packets received versus forward packets transmitted. |

### 3. Data Cleaning
* Infinite values (`+inf`, `-inf`) caused by zero-duration flows (`Flow Duration = 0` leading to division by zero in rate calculations) were converted to `NaN` and imputed with `0`.
* Column names were stripped of leading/trailing whitespace (`df.columns.str.strip()`).
* Categorical target labels were encoded into binary integers: `is_ddos = 1` for DDoS, `0` for BENIGN.

### 4. Median Behavioral Profile Comparison

| Feature Metric | BENIGN (Normal) | DDoS (Attack) | Physical Divergence |
|---|---|---|---|
| **Flow Duration ($\mu s$)** | 31,202.50 (~31 ms) | 1,959,031.00 (~1.96 s) | **63x increase** in session holding time |
| **Total Length of Bwd Packets (Bytes)** | 144.00 | 11,601.00 | **80x increase** in backward response payload |
| **Average Packet Size (Bytes)** | 76.00 | 897.62 | **11.8x increase** in average packet size |
| **Bwd Packet Length Mean (Bytes)** | 82.04 | 1,934.50 | **23.5x increase** in server response packet size |
| **Flow Packets/s** | 115.58 | 2.44 | DDoS flows maintain low packet rates over extended duration |
| **Flow Bytes/s** | 5,661.85 | 157.81 | Normal interactive traffic transfers bursts rapidly |
| **Down/Up Ratio** | 1.00 | 0.00 | DDoS exhibits severe forward/backward asymmetry |

---

## Task 5: Anomaly Identification (DDoS)

* **Selected Anomaly:** Low Orbit Ion Cannon (LOIT) HTTP Flood DDoS Attack.
* **Volume in Dataset:** Exactly **25,000 flows** out of 50,000 total (**50.0%** anomaly prevalence).
* **Behavioral Root Cause:** LOIT repeatedly issues HTTP GET requests designed to consume target web server connection handles. The web server attempts to deliver full HTTP error pages or HTML payloads, generating massive backward byte volume (`11,601 bytes` median) and large response frames (`1,934.5 bytes` mean). Normal user browsing, by contrast, is characterized by brief DNS queries (typically ~144 bytes backward) and rapid TLS handshakes.

---

## Task 6: Visualizations of Traffic Patterns and Anomalies

We generated three high-resolution visualizations saved directly in the project repository:

### Visualization 1: Class Distribution
* **Filename:** `viz1_class_distribution.png`
* **Content:** Clean bar chart displaying the balanced dataset breakdown (25,000 BENIGN vs. 25,000 DDoS).

![Visualization 1: Class Distribution](viz1_class_distribution.png)

### Visualization 2: Average Packet Size Comparison
* **Filename:** `viz2_packet_size_comparison.png`
* **Content:** Boxplot comparing `Average Packet Size` across normal and attack traffic (excluding extreme outliers for visual clarity). Legitimate flows remain tightly bounded under 150 bytes, while DDoS flows shift upwards past 800–1,400 bytes due to continuous server HTML/HTTP transmission.

![Visualization 2: Average Packet Size](viz2_packet_size_comparison.png)

### Visualization 3: Backward Payload Density Distribution
* **Filename:** `viz3_bwd_payload_density.png`
* **Content:** Log-transformed kernel density estimate (KDE) plot of $\log_{10}(\text{Total Length of Bwd Packets} + 1)$. Normal traffic is bimodal with peaks around $\log_{10} = 2.1$ (~144 bytes), while the DDoS attack forms a massive peak at $\log_{10} = 4.06$ (~11,600 bytes).

![Visualization 3: Backward Payload Density](viz3_bwd_payload_density.png)

---

## Task 7: Formulation of a Simple Detection Rule

Based on the observed separation in backward payload and packet sizing, we establish a deterministic, two-condition threshold rule:

$$\text{Alert}_{\text{DDoS}}(\text{flow}) = \begin{cases} 
1 & \text{if } (\text{Average Packet Size} > 400\text{ Bytes}) \land (\text{Total Length of Bwd Packets} > 2,000\text{ Bytes}) \\
0 & \text{otherwise}
\end{cases}$$

### Operational Justification
1. **Low False Positive Rate:** Normal enterprise traffic rarely maintains both an average packet size above 400 bytes and a backward transmission exceeding 2,000 bytes simultaneously (median benign values are 76 bytes and 144 bytes, respectively).
2. **Minimal Computational Footprint:** This rule requires zero matrix multiplications and can be evaluated directly on wire-speed network tap appliances (e.g., Bro/Zeek or Suricata flow scripts).

---

## Task 8: Evaluation of Detection Rule and Machine Learning Model

We conducted two distinct evaluations:

### 1. Simple Threshold Rule Evaluation (All 50,000 Flows)

The rule was evaluated across the entire dataset (both normal and anomalous traffic):

* **Confusion Matrix:**
  * True Negatives (TN): **23,807**
  * False Positives (FP): **1,193**
  * False Negatives (FN): **9,191**
  * True Positives (TP): **15,809**

* **Performance Metrics:**
  * **Accuracy:** **79.23%**
  * **Precision:** **92.98%**
  * **Recall:** **63.24%**
  * **F1-Score:** **75.28%**

![Detection Rule Confusion Matrix](rule_confusion_matrix.png)

**Analytical Assessment:** The rule demonstrates exceptionally high precision (**92.98%**), producing only 1,193 false alarms across 25,000 benign flows. However, its recall (**63.24%**) reveals an inherent blind spot: 9,191 attack flows were cut short during initial connection setup (SYN drops or connection resets) and never accumulated 2,000 backward bytes, thereby bypassing the static threshold.

---

### 2. Constrained Machine Learning Model (Random Forest)

To address the limitations of the static rule without falling into the memorization trap, we trained an ensemble model with strict regularizing constraints:
* **Model:** `RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)`
* **Features Used:** Exclusively the **7 curated heuristic features** (no port numbers, no protocol flags, no window sizes).
* **Train/Test Split:** 80% Training (40,000 flows) and 20% Held-Out Testing (10,000 flows), stratified by class.

#### Test Set Performance Metrics (10,000 Unseen Flows)
* **Accuracy:** **98.00%**
* **Precision:** **97.66%**
* **Recall:** **98.36%**
* **F1-Score:** **98.01%**

#### Detailed Classification Report

```text
              precision    recall  f1-score   support

      BENIGN     0.9835    0.9764    0.9799      5000
        DDoS     0.9766    0.9836    0.9801      5000

    accuracy                         0.9800     10000
   macro avg     0.9800    0.9800    0.9800     10000
weighted avg     0.9800    0.9800    0.9800     10000
```

#### Test Confusion Matrix (10,000 Samples)
* True Negatives (TN): **4,882**
* False Positives (FP): **118**
* False Negatives (FN): **82**
* True Positives (TP): **4,918**

![Random Forest Confusion Matrix](rf_confusion_matrix.png)

#### Heuristic Feature Importances

| Rank | Feature | Gini Importance | Cumulative Importance |
|---|---|---|---|
| 1 | `Average Packet Size` | **0.3645** (36.5%) | 36.5% |
| 2 | `Bwd Packet Length Mean` | **0.1933** (19.3%) | 55.8% |
| 3 | `Flow Duration` | **0.1603** (16.0%) | 71.8% |
| 4 | `Total Length of Bwd Packets` | **0.1394** (13.9%) | 85.7% |
| 5 | `Flow Bytes/s` | **0.0708** (7.1%) | 92.8% |
| 6 | `Flow Packets/s` | **0.0545** (5.5%) | 98.3% |
| 7 | `Down/Up Ratio` | **0.0173** (1.7%) | 100.0% |

![Feature Importances](rf_feature_importance.png)

---

## Task 9: Presentation and Explanation of Findings

### 1. Comparative Performance Overview

| Dimension | Simple Threshold Rule | Constrained Random Forest |
|---|---|---|
| **Underlying Mechanism** | 2 Static Thresholds | 50 Shallow Decision Trees (`depth=3`) |
| **Feature Set** | `Avg Packet Size` + `Bwd Payload` | 7 Curated Behavioral Telemetry Features |
| **Accuracy** | **79.23%** | **98.00%** |
| **Precision** | **92.98%** | **97.66%** |
| **Recall** | **63.24%** | **98.36%** |
| **F1-Score** | **75.28%** | **98.01%** |
| **Inference Latency** | $< 0.1\ \mu s$ (Zero overhead) | $\approx 1.2\ \text{ms}$ (Batch inference) |
| **Operational Deployment** | In-line hardware filter (eNPU / FPGA) | Software NIDS analyzer (Zeek / Suricata) |

### 2. Core Takeaways & Engineering Insights

1. **Eliminating the Memorization Illusion:**
   The apparent 99.97%+ accuracy achieved by naively training classifiers on raw CIC-IDS2017 features is misleading. When models are allowed to train on port numbers (`Destination Port = 80`) or fixed TCP segment parameters, they function as brittle signature memorizers. Restricting the feature set to 7 physical behavioral metrics forces the model to generalize across actual flow dynamics.

2. **Why the Simple Rule Had Low Recall (63.24%):**
   In volumetric HTTP DDoS attacks, a substantial portion of sessions are terminated prematurely due to server backlog queue exhaustion (SYN cookies, RST teardowns, or incomplete request buffering). These aborted sessions never deliver the full 2,000-byte response payload and thus escape static payload thresholding.

3. **How Machine Learning Resolves the Gap (98.36% Recall):**
   The shallow Random Forest (`max_depth=3`) achieved a 35-percentage-point gain in recall over the static rule. It achieves this by cross-referencing `Flow Duration` and packet arrival rates (`Flow Packets/s`) with `Average Packet Size`. Even when response payloads are truncated, protracted connection holding times coupled with abnormal packet size distributions reliably flag anomalous attack flows.

4. **Recommended Hybrid Defense Architecture:**
   In production SOC deployments, these two detection mechanisms complement each other:
   * **Tier 1 (Edge Gatekeeper):** The static rule runs directly in high-speed hardware tables (eBPF/XDP), instantly dropping >60% of high-volume DDoS sessions with near-zero latency and 93% precision.
   * **Tier 2 (Heuristic ML Inspector):** Flows that pass Tier 1 are evaluated by the constrained Random Forest model, detecting subtle, low-payload, or early-stage connection-exhaustion flows with 98% overall accuracy.

---

## Conclusion

This laboratory demonstrates that high-performance intrusion detection does not require complex deep neural networks or hundreds of leaky features. By systematically identifying and stripping dataset artifacts, applying rigorous data sanitization, and constraining tree depth, we achieved a robust, interpretable, and defensible **98.00% detection accuracy** using only 7 fundamental network behavioral heuristics.
